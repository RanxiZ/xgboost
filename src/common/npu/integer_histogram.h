// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_INTEGER_HISTOGRAM_H_
#define XGBOOST_COMMON_NPU_INTEGER_HISTOGRAM_H_
#include "acl_ops.h"
#include "histogram_contract.h"
#include "integer_histogram_abi.h"
#include <cstdint>
#include <memory>
#include <limits>
#include <vector>

namespace xgboost::common::npu {
// Numerical contract follows gpu_hist/histogram.cu: selected GLOBAL row ids,
// skip missing entries, add quantized int64 gradient and Hessian independently.
// Feature-major local int64 bin ids are an NPU layout, not CUDA ELLPACK storage.
// Single-device, single-target, in-memory only; no page/distributed aggregation.
class IntegerHistogramBuilder {
  friend class DeviceRowPartitioner;
 public:
  // Opaque validated columns. Only this factory accepts mutable ownership;
  // consumers retain a read-only handle across trees and expose no tensor aliases.
  class ImmutableBins {
    friend class IntegerHistogramBuilder;
    std::vector<Tensor> columns_;
    std::vector<int64_t> bin_counts_;
    Runtime const* owner_;
    int64_t rows_;
    ImmutableBins(std::vector<Tensor>&& columns, std::vector<int64_t> counts,
                  Runtime const* owner, int64_t rows)
        : columns_{std::move(columns)}, bin_counts_{std::move(counts)},
          owner_{owner}, rows_{rows} {}
   public:
    ImmutableBins(ImmutableBins const&) = delete;
    ImmutableBins& operator=(ImmutableBins const&) = delete;
  };
 private:
  struct Data {
    std::shared_ptr<ImmutableBins const> bins;
    std::vector<Tensor> const& columns;
    Tensor gradient, hessian;
    std::vector<int64_t> const& bin_counts;
    Data(std::shared_ptr<ImmutableBins const> b, Tensor&& g, Tensor&& h)
        : bins{std::move(b)}, columns{bins->columns_}, gradient{std::move(g)},
          hessian{std::move(h)}, bin_counts{bins->bin_counts_} {}
  };
 public:
  struct Feature {
    int64_t bin_count;
    // A zero-bin feature has one zero placeholder element, never a valid bin.
    Tensor gradient, hessian;
  };
  class Histogram {
    friend class IntegerHistogramBuilder;
    friend class NumericalSplitStatistics;
    friend class NumericalSplitEvaluator;
    friend class DeviceRowPartitioner;
    friend class IntegerHistogramTestAccess;  // Defined only by the standalone test.
    std::shared_ptr<Data const> source_;
    Tensor membership_, root_gradient_, root_hessian_, packed_;
    std::vector<Feature> features_;
    int64_t rows_;
    Histogram(std::shared_ptr<Data const> source, Tensor&& membership, Tensor&& root_g,
              Tensor&& root_h, Tensor&& packed, std::vector<Feature>&& features, int64_t rows)
        : source_{std::move(source)}, membership_{std::move(membership)},
          root_gradient_{std::move(root_g)}, root_hessian_{std::move(root_h)},
          packed_{std::move(packed)}, features_{std::move(features)}, rows_{rows} {}
   public:
    Histogram(Histogram&&) = default;
    Histogram(Histogram const&) = delete;
    Histogram& operator=(Histogram const&) = delete;
    // Keep validated device results private: a const Tensor still exposes raw
    // mutable CANN handles. Future split evaluation must use a trusted internal
    // interface rather than publishing writable histogram buffers.
    std::size_t FeatureCount() const { return features_.size(); }
    int64_t FeatureBinCount(std::size_t feature) const { return features_.at(feature).bin_count; }
    int64_t Rows() const { return rows_; }
  };
 private:
  Ops& ops_;
  std::shared_ptr<Data const> data_;
  Buffer metadata_device_;  // Reused by every node built from this immutable layout.
  std::size_t build_launches_{0};
  void CheckVector(Tensor const& t, int64_t size) const {
    if (t.Owner() != &ops_.GetRuntime() || t.Type() != ACL_INT64 || t.Count() != size) {
      throw std::invalid_argument("NPU histogram tensor owner, dtype or shape mismatch");
    }
  }
  void RejectIf(Tensor const& condition, char const* message) {
    if (ops_.SumControl(ops_.Cast(condition, ACL_INT64)) != 0) throw std::invalid_argument(message);
  }
  int64_t MagnitudeTotal(Tensor const& values) {
    auto low=ops_.AndBits(values, UINT32_MAX);
    auto high=ops_.Shift(values,ops_.Constant(1,32),false);
    return CheckedMagnitudeTotal(ops_.SumControl(low),ops_.SumControl(high));
  }
  void ValidateChannel(Tensor const& values) {
    constexpr int64_t limit=(INT64_C(1)<<62)-1;
    RejectIf(ops_.Logic(ops_.Compare(values,-limit,-1),ops_.Compare(values,limit,1),false),
             "NPU histogram quantized element exceeds safety domain");
    auto zero=ops_.Constant(1,0); auto negative=ops_.Compare(values,0,-1);
    auto positive=ops_.Where(negative,zero,values);
    auto magnitude=ops_.Where(negative,ops_.Add(zero,values,-1),zero);
    // Independently bound each sign, so every unique-row subset and every
    // partial histogram sum fits int64 regardless of accumulation order.
    MagnitudeTotal(positive); MagnitudeTotal(magnitude);
  }
  int64_t HistogramSize() const {
    int64_t size = 0;
    for (auto bins : data_->bin_counts) {
      auto count = std::max<int64_t>(1, bins);
      if (count > (INT64_MAX - size) / 2) {
        throw std::overflow_error("NPU packed histogram size overflow");
      }
      size += 2 * count;
    }
    return size;
  }
  std::vector<Feature> FeatureViews(Tensor const& packed) const {
    std::vector<Feature> features;
    features.reserve(data_->bin_counts.size());
    int64_t offset = 0;
    for (auto bins : data_->bin_counts) {
      auto count = std::max<int64_t>(1, bins);
      features.push_back({bins, Tensor::Slice(packed, offset, count),
                          Tensor::Slice(packed, offset + count, count)});
      offset += 2 * count;
    }
    return features;
  }
  Tensor Zeros(int64_t n) {
    Tensor out{ops_.GetRuntime(),n,ACL_INT64}; out.Storage().Zero(); return out;
  }
 public:
  // Takes exclusive ownership of bin storage, validates once, and publishes only
  // an opaque immutable handle. Scales and gradient provenance remain per-tree.
  static std::shared_ptr<ImmutableBins const> BindBins(
      Ops& ops, std::vector<Tensor>&& columns, std::vector<int64_t> bin_counts) {
    auto n = columns.empty() ? 0 : columns.front().Count();
    ValidateHistogramLayout(n, bin_counts);
    if (columns.size() != bin_counts.size()) {
      throw std::invalid_argument("NPU histogram feature count mismatch");
    }
    for (std::size_t f = 0; f < columns.size(); ++f) {
      auto const& col = columns[f];
      if (col.Owner() != &ops.GetRuntime() || col.Type() != ACL_INT64 || col.Count() != n) {
        throw std::invalid_argument("NPU histogram tensor owner, dtype or shape mismatch");
      }
      auto invalid = ops.Logic(ops.Compare(col, -1, -1),
                               ops.Compare(col, bin_counts[f] - 1, 1), false);
      if (ops.SumControl(ops.Cast(invalid, ACL_INT64)) != 0) {
        throw std::invalid_argument("NPU histogram feature bin id out of range");
      }
    }
    return std::shared_ptr<ImmutableBins const>{
        new ImmutableBins{std::move(columns), std::move(bin_counts), &ops.GetRuntime(), n}};
  }
  IntegerHistogramBuilder(Ops& ops, std::shared_ptr<ImmutableBins const> bins,
                          Tensor&& gradient, Tensor&& hessian) : ops_{ops} {
    auto n = gradient.Count();
    if (!bins || bins->owner_ != &ops.GetRuntime() || bins->rows_ != n) {
      throw std::invalid_argument("NPU histogram immutable bins runtime or shape mismatch");
    }
    CheckVector(gradient, n); CheckVector(hessian, n);
    ValidateChannel(gradient); ValidateChannel(hessian);
    data_ = std::make_shared<Data>(std::move(bins), std::move(gradient), std::move(hessian));
  }
  // Compatibility entry point for independent component tests and owned input.
  IntegerHistogramBuilder(Ops& ops, std::vector<Tensor>&& columns, Tensor&& gradient,
                          Tensor&& hessian, std::vector<int64_t> bin_counts)
      : IntegerHistogramBuilder{ops, BindBins(ops, std::move(columns), std::move(bin_counts)),
                                std::move(gradient), std::move(hessian)} {}
  // Production root construction keeps the row sequence device-resident and
  // exposes no writable alias to the validated dataset or histogram buffers.
  Histogram BuildRoot() {
    auto ids = ops_.ExclusiveInt64Scan(ops_.Constant(data_->gradient.Count(), 1));
    return this->Build(&ids);
  }
  std::size_t BuildLaunches() const { return build_launches_; }
  // nullptr represents an empty node; Tensor itself forbids zero-length data.
  // Nonempty row ids must be unique. Repetition is rejected, not reweighted.
  Histogram Build(Tensor const* row_ids) {
    auto n=data_->gradient.Count();
    Tensor packed{ops_.GetRuntime(), HistogramSize(), ACL_INT64};
    auto features = FeatureViews(packed);
    if (!row_ids) {
      auto membership=Zeros(n);
      packed.Storage().Zero();
      return Histogram{data_,std::move(membership),Zeros(1),Zeros(1),
                       std::move(packed),std::move(features),0};
    }
    CheckVector(*row_ids,row_ids->Count());
    if (row_ids->Count()>n) throw std::invalid_argument("Too many NPU node row ids");
    RejectIf(ops_.Logic(ops_.Compare(*row_ids,0,-1),ops_.Compare(*row_ids,n-1,1),false),
             "NPU node row id out of range");
    auto counts=ops_.ScatterSum(*row_ids,ops_.Constant(row_ids->Count(),1),n);
    RejectIf(ops_.Compare(counts,1,1),"Duplicate NPU node row ids");
    auto g=ops_.Gather(data_->gradient,*row_ids), h=ops_.Gather(data_->hessian,*row_ids);
    auto root_g=ops_.Sum(g), root_h=ops_.Sum(h);
    auto feature_count=data_->columns.size();
    if (feature_count > static_cast<std::size_t>(INT32_MAX) ||
        feature_count > std::numeric_limits<std::size_t>::max() / sizeof(histogram::FeatureDescriptor)) {
      throw std::overflow_error("NPU histogram metadata size overflow");
    }
    auto row_count=static_cast<uint64_t>(row_ids->Count());
    auto feature_count_u64=static_cast<uint64_t>(feature_count);
    auto metadata_bytes=feature_count * sizeof(histogram::FeatureDescriptor);
    features.reserve(feature_count);
    std::vector<histogram::FeatureDescriptor> metadata;
    metadata.reserve(feature_count);
    for (std::size_t f=0;f<feature_count;++f) {
      auto bins=data_->bin_counts[f];
      // Feature tensors are bounded views into one contiguous allocation.
      metadata.push_back({
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data_->columns[f].Storage().Data())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(features[f].gradient.Storage().Data())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(features[f].hessian.Storage().Data())),
        static_cast<uint32_t>(bins), 0});
    }
    // One descriptor upload replaces the per-feature Gather/Where/Scatter
    // sequence. Row guards, root sums, channel validation and subset proofs
    // remain ACLNN operations exactly as before.
    if (!metadata_device_.Data()) metadata_device_ = ops_.GetRuntime().Allocate(metadata_bytes);
    metadata_device_.Upload(metadata.data(), metadata.size() * sizeof(metadata[0]));
    XgbNpuIntegerHistogramBuild(ops_.GetRuntime().Stream(), row_ids->Storage().Data(),
                                data_->gradient.Storage().Data(), data_->hessian.Storage().Data(),
                                metadata_device_.Data(), row_count, feature_count_u64);
    ++build_launches_;
    ops_.GetRuntime().Synchronize();
    // Keep exact device membership for the later subtraction precondition.
    return Histogram{data_,std::move(counts),std::move(root_g),std::move(root_h),
                     std::move(packed),std::move(features),row_ids->Count()};
  }
  Histogram SubtractSibling(Histogram const& parent, Histogram const& child) {
    if (parent.source_.get()!=data_.get() || child.source_.get()!=data_.get()) {
      throw std::invalid_argument("NPU histogram subtraction requires identical immutable source and scales");
    }
    if (child.rows_>parent.rows_) throw std::invalid_argument("NPU histogram child is larger than parent");
    auto membership=ops_.Add(parent.membership_,child.membership_,-1);
    RejectIf(ops_.Compare(membership,0,-1),"NPU histogram child is not a parent subset");
    // The contiguous layout turns 2 * feature_count launches into one exact
    // int64 subtraction, including zero-bin placeholders.
    auto packed = ops_.Add(parent.packed_, child.packed_, -1);
    auto features = FeatureViews(packed);
    // The subset proof makes every subtraction the sum over a disjoint sibling,
    // bounded by the previously checked source positive/negative magnitude sums.
    return Histogram{data_,std::move(membership),ops_.Add(parent.root_gradient_,child.root_gradient_,-1),
                     ops_.Add(parent.root_hessian_,child.root_hessian_,-1),std::move(packed),
                     std::move(features),parent.rows_-child.rows_};
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_INTEGER_HISTOGRAM_H_
