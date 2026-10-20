// Copyright 2026, XGBoost Contributors
#include "native_training.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include "../hist_util.h"
#include "../../data/simple_dmatrix.h"
#include "native_training_abi.h"

namespace xgboost::common::npu {
namespace {
std::shared_ptr<SparsePage const> SinglePage(Context const* cpu_ctx, DMatrix* dmat) {
  // NumBatches()==1 alone is insufficient: an external-memory matrix can have
  // one page. Accept the concrete in-memory representation, never ext pages
  // or quantile matrices whose raw numerical values are no longer available.
  if (!dynamic_cast<data::SimpleDMatrix*>(dmat) || !dmat->SingleColBlock()) {
    throw std::invalid_argument("grow_npu_hist requires an in-memory SimpleDMatrix with raw CSR");
  }
  auto batches = dmat->GetBatches<SparsePage>(cpu_ctx);
  auto it = batches.begin();
  if (it.AtEnd()) throw std::invalid_argument("NPU training has no CSR page");
  auto page = it.Page();
  ++it;
  if (!it.AtEnd() || !page || page->base_rowid != 0 ||
      page->Size() != dmat->Info().num_row_) {
    throw std::invalid_argument("NPU training requires one complete CSR page");
  }
  return page;
}
}  // namespace

void NativeTrainingData::Bind(Context const* cpu_ctx, DMatrix* dmat, bst_bin_t max_bin,
                              std::shared_ptr<Runtime> runtime, std::shared_ptr<Ops> ops) {
  if (!cpu_ctx || !cpu_ctx->IsCPU() || !dmat || !runtime || !ops ||
      &ops->GetRuntime() != runtime.get() || max_bin < 2) {
    throw std::invalid_argument("Invalid NPU training data binding");
  }
  auto const& info = dmat->Info();
  if (info.num_row_ <= 0 || info.num_row_ > INT32_MAX || info.num_col_ == 0 ||
      info.num_col_ > INT32_MAX) {
    throw std::invalid_argument("NPU training requires nonempty single-device scalar row data");
  }
  for (auto type : info.feature_types.ConstHostSpan()) {
    if (type != FeatureType::kNumerical) {
      throw std::invalid_argument("grow_npu_hist does not support categorical features");
    }
  }
  auto page = SinglePage(cpu_ctx, dmat);
  auto const& weights = info.weights_.ConstHostVector();
  // Compare ownership, not addresses: even an allocator reusing a destroyed
  // DMatrix/page address cannot satisfy this cache binding. A page is treated
  // as immutable, matching DMatrix's training data contract.
  bool same_source = !source_.expired() && !source_.owner_before(page) &&
                     !page.owner_before(source_);
  if (same_source && runtime_.get() == runtime.get() && nrows_ == info.num_row_ &&
      nfeatures_ == info.num_col_ && max_bin_ == max_bin && sketch_weights_ == weights) {
    return;
  }

  auto cuts = common::SketchOnDMatrix(cpu_ctx, dmat, max_bin);
  if (cuts.HasCategorical()) throw std::invalid_argument("NPU categorical cuts unsupported");
  auto ptrs = cuts.Ptrs();
  auto values = cuts.Values();
  if (ptrs.size() != static_cast<size_t>(info.num_col_) + 1 || ptrs.front() != 0 ||
      ptrs.back() != values.size()) {
    throw std::invalid_argument("Invalid NPU quantile cut layout");
  }
  std::vector<int64_t> counts;
  counts.reserve(info.num_col_);
  for (bst_feature_t f = 0; f < info.num_col_; ++f) {
    if (ptrs[f] > ptrs[f + 1]) throw std::invalid_argument("Invalid NPU cut offsets");
    counts.push_back(ptrs[f + 1] - ptrs[f]);
    for (uint32_t i = ptrs[f]; i < ptrs[f + 1]; ++i) {
      if (!std::isfinite(values[i]) ||
          (i > ptrs[f] && !(values[i - 1] < values[i]))) {
        throw std::invalid_argument("NPU requires strictly ordered numerical cuts");
      }
    }
  }

  auto view = page->GetView();
  std::vector<uint64_t> offsets(static_cast<size_t>(info.num_row_) + 1, 0);
  std::vector<native_abi::Entry> entries;
  entries.reserve(page->data.Size());
  for (size_t row = 0; row < view.Size(); ++row) {
    std::set<bst_feature_t> seen;
    for (auto const& e : view[row]) {
      if (e.index >= info.num_col_ || !seen.insert(e.index).second || std::isinf(e.fvalue)) {
        throw std::invalid_argument("NPU CSR feature is duplicated, out of range or infinite");
      }
      entries.push_back({e.index, e.fvalue});
    }
    offsets[row + 1] = entries.size();
  }
  auto d_offsets = runtime->AllocateElements<uint64_t>(offsets.size());
  auto d_entries = runtime->Allocate(std::max<size_t>(1, entries.size() * sizeof(native_abi::Entry)));
  auto d_ptrs = runtime->AllocateElements<uint32_t>(ptrs.size());
  auto d_cuts = runtime->Allocate(std::max<size_t>(1, values.size() * sizeof(float)));
  d_offsets.Upload(offsets.data(), offsets.size() * sizeof(uint64_t));
  if (!entries.empty()) d_entries.Upload(entries.data(), entries.size() * sizeof(native_abi::Entry));
  d_ptrs.Upload(ptrs.data(), ptrs.size() * sizeof(uint32_t));
  if (!values.empty()) d_cuts.Upload(values.data(), values.size() * sizeof(float));
  std::vector<Tensor> columns;
  std::vector<uint64_t> pointers;
  columns.reserve(info.num_col_); pointers.reserve(info.num_col_);
  for (bst_feature_t f = 0; f < info.num_col_; ++f) {
    columns.emplace_back(*runtime, info.num_row_, ACL_INT64);
    pointers.push_back(reinterpret_cast<uintptr_t>(columns.back().Storage().Data()));
  }
  auto d_columns = runtime->AllocateElements<uint64_t>(pointers.size());
  d_columns.Upload(pointers.data(), pointers.size() * sizeof(uint64_t));
  XgbNpuNativeBinRows(runtime->Stream(), d_offsets.Data(), d_entries.Data(), d_ptrs.Data(),
                      d_cuts.Data(), d_columns.Data(), info.num_row_, info.num_col_);
  runtime->Synchronize();

  // Validate bin ranges before publication. All later builders share this
  // opaque storage without a per-tree device copy or repeated column scan.
  auto bins = IntegerHistogramBuilder::BindBins(*ops, std::move(columns), counts);
  bins_.reset();
  ops_ = std::move(ops); runtime_ = std::move(runtime);
  bins_ = std::move(bins);
  cut_ptrs_ = std::move(ptrs); cuts_ = std::move(values); bin_counts_ = std::move(counts);
  source_ = page; nrows_ = info.num_row_; nfeatures_ = info.num_col_; max_bin_ = max_bin;
  sketch_weights_ = weights;
}

float NativeTrainingData::Threshold(uint32_t feature, uint32_t candidate) const {
  if (feature >= bin_counts_.size() || candidate >= bin_counts_[feature]) {
    throw std::out_of_range("Invalid NPU selected split index");
  }
  // Pinned GPU histogram candidate contract: exclusive prefix i has threshold
  // cut[i-1], and candidate 0 has -infinity. Raw prediction uses value < cut.
  return candidate == 0 ? -std::numeric_limits<float>::infinity()
                        : cuts_.at(cut_ptrs_[feature] + candidate - 1);
}

SampledNativeGradient SampleNativeGradient(NativeGradient const& input, float subsample,
                                           uint64_t seed) {
  if (!input.runtime || !input.ops || !input.gradient || !input.hessian || !input.nrows ||
      input.nrows > INT32_MAX || !std::isfinite(subsample) || subsample <= 0 || subsample > 1 ||
      &input.ops->GetRuntime() != input.runtime.get()) {
    throw std::invalid_argument("Invalid native NPU gradient/sampling input");
  }
  for (auto const* t : {input.gradient.get(), input.hessian.get()}) {
    if (t->Owner() != input.runtime.get() || t->Type() != ACL_FLOAT ||
        t->Count() != input.nrows || !t->Descriptor()) {
      throw std::invalid_argument("NPU gradient tensor owner, dtype or shape mismatch");
    }
  }
  SampledNativeGradient out{Tensor{*input.runtime, input.nrows, ACL_FLOAT},
                            Tensor{*input.runtime, input.nrows, ACL_FLOAT}};
  XgbNpuNativeSampleRows(input.runtime->Stream(), input.gradient->Storage().Data(),
                         input.hessian->Storage().Data(), out.gradient.Storage().Data(),
                         out.hessian.Storage().Data(), input.nrows, seed, subsample);
  input.runtime->Synchronize();
  return out;
}
}  // namespace xgboost::common::npu