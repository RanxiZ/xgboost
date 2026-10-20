// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_ROW_PARTITION_CONTRACT_H_
#define XGBOOST_COMMON_NPU_ROW_PARTITION_CONTRACT_H_
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace xgboost::common::npu {
// Numerical candidate-local exclusive-prefix index, NOT a raw threshold.
// Pinned CUDA evaluate_splits.cu uses -inf at i=0, cut[i-1] otherwise;
// updater_gpu_hist.cu compares reconstructed cut[bin] <= that threshold.
// With the source's ordered unique numerical cuts this is bin < i. Raw
// prediction instead uses raw_value < threshold (predict_fn.h). No raw-value,
// categorical, external-page, or inferred-cut routing is implemented here.
struct NumericalRowSplit {
  std::size_t feature;
  int64_t candidate;
  bool missing_left;
};
inline void ValidateRowSplit(std::vector<int64_t> const& bin_counts, NumericalRowSplit split) {
  if (split.feature >= bin_counts.size() || split.candidate < 0 ||
      split.candidate >= bin_counts[split.feature]) {
    throw std::invalid_argument("NPU row split requires an existing numerical candidate-local index");
  }
}
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_ROW_PARTITION_CONTRACT_H_