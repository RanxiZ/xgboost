// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_DEVICE_ROW_PARTITIONER_H_
#define XGBOOST_COMMON_NPU_DEVICE_ROW_PARTITIONER_H_
#include "integer_histogram.h"
#include "row_partition_contract.h"
#include <memory>
#include <utility>
namespace xgboost::common::npu {
// Numerical partitioning over one immutable dataset/runtime. The reference
// adapter owns independent row sets. The consuming training adapter alternates
// two full-row buffers and retains disjoint node ranges in either buffer.
// Source contract: gpu_hist/row_partitioner.cuh SortPositionBatch predicates,
// scans flags, scatters row ids, and downloads child counts. We preserve child
// membership, not its reverse-right physical order, and do not assign leaf ids.
class DeviceRowPartitioner {
  using Builder = IntegerHistogramBuilder;
  using Source = std::shared_ptr<Builder::Data const>;
  struct Arena {
    Tensor first, second;
    explicit Arena(Tensor&& root)
        : first{std::move(root)}, second{*first.Owner(), first.Count(), ACL_INT64} {}
    Tensor const& At(uint32_t slot) const { return slot == 0 ? first : second; }
  };
 public:
  // Opaque owning device row tensor. Even const Tensor exposes writable CANN
  // handles, so no Tensor, descriptor, storage, or host row export is public.
  // Empty rows have no tensor. Source ownership survives builder destruction;
  // Runtime and Ops must outlive the partitioner, row sets and histograms.
  class Rows {
    friend class DeviceRowPartitioner;
    friend class RowPartitionTestAccess;  // Defined only in standalone tests.
    Source source_;
    std::unique_ptr<Tensor> ids_;
    int64_t count_;
    std::shared_ptr<Arena> arena_;
    int64_t offset_{0};
    uint32_t slot_{0};
    Rows(Source source, std::unique_ptr<Tensor> ids, int64_t count,
         std::shared_ptr<Arena> arena = nullptr, int64_t offset = 0, uint32_t slot = 0)
        : source_{std::move(source)}, ids_{std::move(ids)}, count_{count},
          arena_{std::move(arena)}, offset_{offset}, slot_{slot} {}
   public:
    Rows(Rows&&) = default;
    Rows(Rows const&) = delete;
    Rows& operator=(Rows const&) = delete;
    int64_t Count() const { return count_; }
  };
  struct Result { Rows left, right; };
 private:
  Ops& ops_;
  Source source_;
  void CheckSource(Source const& source) const {
    if (!source || source.get() != source_.get()) {
      throw std::invalid_argument("NPU row partition requires identical immutable dataset provenance");
    }
  }
  void CheckRows(Rows const& rows) const {
    CheckSource(rows.source_);
    if (rows.count_ < 0 || rows.count_ > source_->gradient.Count() ||
        static_cast<bool>(rows.ids_) != (rows.count_ != 0) ||
        (rows.arena_ && (rows.slot_ > 1 || rows.offset_ < 0 ||
                        rows.offset_ > source_->gradient.Count() - rows.count_))) {
      throw std::invalid_argument("Invalid or moved-from NPU row set");
    }
  }
  void RejectIf(Tensor const& condition, char const* message) {
    if (ops_.SumControl(ops_.Cast(condition, ACL_INT64))) throw std::invalid_argument(message);
  }
  Tensor Sequence(int64_t count) { return ops_.ExclusiveInt64Scan(ops_.Constant(count, 1)); }
  Rows Empty() const { return Rows{source_, nullptr, 0}; }
  Rows Compact(Tensor const& ids, Tensor const& flag, int64_t count,
               std::shared_ptr<Arena> arena = nullptr, int64_t offset = 0, uint32_t slot = 0) {
    if (count == 0) return Rows{source_, nullptr, 0, std::move(arena), offset, slot};
    auto mask = ops_.Cast(flag, ACL_INT64);
    auto prefix = ops_.ExclusiveInt64Scan(mask);
    // Selected indices are unique [0,count). Unselected rows target a dedicated
    // zero-valued sink at m. Thus every ScatterAdd partial sum is either one
    // validated row id or zero: no overflow, float cast, or host index work.
    auto target = ops_.Where(flag, prefix, ops_.Constant(1, ids.Count()));
    auto values = ops_.Where(flag, ids, ops_.Constant(1, 0));
    auto packed = ops_.ScatterSum(target, values, ids.Count() + 1);
    if (!arena) {
      return Rows{source_, std::make_unique<Tensor>(ops_.Gather(packed, Sequence(count))), count};
    }
    auto out = std::make_unique<Tensor>(Tensor::Slice(arena->At(slot), offset, count));
    ops_.GatherInto(packed, Sequence(count), *out);
    return Rows{source_, std::move(out), count, std::move(arena), offset, slot};
  }
 public:
  explicit DeviceRowPartitioner(Ops& ops, Builder::Histogram const& context)
      : ops_{ops}, source_{context.source_} {
    if (!source_ || source_->gradient.Owner() != &ops_.GetRuntime()) {
      throw std::invalid_argument("NPU row partition histogram/runtime mismatch");
    }
  }
  // Root includes every dataset row (also zero-gradient/sampled-out rows).
  Rows Root() {
    auto n = source_->gradient.Count();
    return Rows{source_, std::make_unique<Tensor>(Sequence(n)), n};
  }
  // Production roots keep one full-row sequence and one destination buffer.
  // Descendants retain bounded views and alternate slots within their parent's
  // interval. A split consumes the parent so no live ancestor can be overwritten.
  Rows RootBuffered() {
    auto n = source_->gradient.Count();
    auto arena = std::make_shared<Arena>(Sequence(n));
    auto ids = std::make_unique<Tensor>(Tensor::Slice(arena->first, 0, n));
    return Rows{source_, std::move(ids), n, std::move(arena), 0, 0};
  }
  // Import only rows exactly matching an already validated histogram membership.
  // The row tensor is copied ON DEVICE so callers retain no writable row alias.
  // nullptr is the sole empty representation. No full row/bin downloads occur.
  Rows Capture(Builder::Histogram const& node, Tensor const* ids) {
    CheckSource(node.source_);
    auto count = ids ? ids->Count() : 0;
    if (count != node.rows_ || count > source_->gradient.Count()) {
      throw std::invalid_argument("NPU captured row count differs from histogram");
    }
    if (!ids) return Empty();
    if (ids->Owner() != &ops_.GetRuntime() || ids->Type() != ACL_INT64 || !ids->Descriptor()) {
      throw std::invalid_argument("NPU captured row tensor runtime/dtype mismatch");
    }
    auto n = source_->gradient.Count();
    RejectIf(ops_.Logic(ops_.Compare(*ids, 0, -1), ops_.Compare(*ids, n - 1, 1), false),
             "NPU captured row id out of range");
    auto membership = ops_.ScatterSum(*ids, ops_.Constant(count, 1), n);
    RejectIf(ops_.Compare(membership, 1, 1), "Duplicate NPU captured row ids");
    auto equal = ops_.Compare(ops_.Add(membership, node.membership_, -1), 0, 0);
    RejectIf(ops_.Not(equal), "NPU captured rows differ from histogram membership");
    return Rows{source_, std::make_unique<Tensor>(ops_.AddScalar(*ids, 0)), count};
  }
 private:
  Result PartitionImpl(Rows const& rows, NumericalRowSplit split, bool buffered) {
    CheckRows(rows);
    ValidateRowSplit(source_->bin_counts, split);
    auto arena = buffered ? rows.arena_ : nullptr;
    auto slot = buffered ? 1U - rows.slot_ : 0U;
    if (!rows.count_) {
      return {Rows{source_, nullptr, 0, arena, rows.offset_, slot},
              Rows{source_, nullptr, 0, arena, rows.offset_, slot}};
    }
    auto bins = ops_.Gather(source_->columns[split.feature], *rows.ids_);
    RejectIf(ops_.Logic(ops_.Compare(bins, -1, -1),
                        ops_.Compare(bins, source_->bin_counts[split.feature] - 1, 1), false),
             "NPU row partition bin outside immutable numerical source domain");
    auto missing = ops_.Compare(bins, -1, 0);
    auto numerical_left = ops_.Compare(bins, split.candidate, -1);
    auto left = split.missing_left ? ops_.Logic(missing, numerical_left, false)
                                  : ops_.Logic(ops_.Not(missing), numerical_left, true);
    auto count = ops_.SumControl(ops_.Cast(left, ACL_INT64));
    if (count < 0 || count > rows.count_) throw std::runtime_error("Invalid NPU child count");
    return {Compact(*rows.ids_, left, count, arena, rows.offset_, slot),
            Compact(*rows.ids_, ops_.Not(left), rows.count_ - count, arena,
                    rows.offset_ + count, slot)};
  }
 public:
  // Reference path: repeated partitioning of the same immutable row set is safe.
  Result Partition(Rows const& rows, NumericalRowSplit split) {
    return PartitionImpl(rows, split, false);
  }
  // Training path: disjoint descendant ranges can progress in any order,
  // including loss-guided growth and unsplit terminal siblings. Only children
  // survive success; metadata failures leave the original parent usable.
  Result PartitionConsume(Rows&& rows, NumericalRowSplit split) {
    CheckRows(rows);
    if (!rows.arena_) throw std::invalid_argument("NPU consuming partition requires buffered rows");
    auto out = PartitionImpl(rows, split, true);
    rows.ids_.reset(); rows.arena_.reset(); rows.source_.reset(); rows.count_ = -1;
    return out;
  }
  // Trusted adapter; never publishes row tensors or mutates the dataset.
  Builder::Histogram BuildHistogram(Builder& builder, Rows const& rows) {
    CheckRows(rows); CheckSource(builder.data_);
    if (&builder.ops_.GetRuntime() != &ops_.GetRuntime()) {
      throw std::invalid_argument("NPU child histogram runtime mismatch");
    }
    return builder.Build(rows.ids_.get());
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_DEVICE_ROW_PARTITIONER_H_