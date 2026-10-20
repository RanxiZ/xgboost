// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_NATIVE_TRAINING_H_
#define XGBOOST_COMMON_NPU_NATIVE_TRAINING_H_

#include <cstdint>
#include <memory>
#include <vector>
#include "acl_ops.h"
#include "integer_histogram.h"
#include "native_prediction.h"
#include "xgboost/context.h"
#include "xgboost/data.h"

namespace xgboost::common::npu {
// Updater-local cache. Identity is the SparsePage shared ownership control block,
// never a raw DMatrix address. Expiration/replacement forces a fresh binding.
// Bin columns are validated once and shared read-only by successive trees.
// Rebinding never invalidates a histogram still retaining the previous bins.
class NativeTrainingData {
  std::shared_ptr<Runtime> runtime_;
  std::shared_ptr<Ops> ops_;
  std::weak_ptr<SparsePage const> source_;
  bst_idx_t nrows_{0};
  bst_feature_t nfeatures_{0};
  bst_bin_t max_bin_{0};
  std::vector<float> sketch_weights_;
  std::vector<uint32_t> cut_ptrs_;
  std::vector<float> cuts_;
  std::vector<int64_t> bin_counts_;
  std::shared_ptr<IntegerHistogramBuilder::ImmutableBins const> bins_;

 public:
  void Bind(Context const* cpu_ctx, DMatrix* dmat, bst_bin_t max_bin,
            std::shared_ptr<Runtime> runtime, std::shared_ptr<Ops> ops);
  std::shared_ptr<IntegerHistogramBuilder::ImmutableBins const> Bins() const {
    if (!bins_) throw std::logic_error("NPU training data not bound");
    return bins_;
  }
  std::vector<int64_t> const& BinCounts() const { return bin_counts_; }
  float Threshold(uint32_t feature, uint32_t candidate) const;
};

struct SampledNativeGradient {
  Tensor gradient;
  Tensor hessian;
};
// Always writes a new pair; never changes objective/prediction sidecar data.
SampledNativeGradient SampleNativeGradient(NativeGradient const& input, float subsample,
                                          uint64_t seed);
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_NATIVE_TRAINING_H_