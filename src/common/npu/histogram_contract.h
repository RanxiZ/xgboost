// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_HISTOGRAM_CONTRACT_H_
#define XGBOOST_COMMON_NPU_HISTOGRAM_CONTRACT_H_
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <vector>

namespace xgboost::common::npu {
inline void ValidateHistogramLayout(int64_t rows, std::vector<int64_t> const& bins) {
  if (rows <= 0 || rows > INT32_MAX || bins.empty() || bins.size() > INT32_MAX) {
    throw std::invalid_argument("NPU histogram requires 1..INT32_MAX rows and features");
  }
  for (auto n : bins) {
    if (n < 0 || n > INT32_MAX) throw std::invalid_argument("Invalid NPU feature bin count");
  }
}
// Low/high unsigned-magnitude limbs were reduced separately. For <=INT32_MAX
// values, each <2^62, both reductions fit int64 in every order. This check
// reconstructs only a scalar summary, never histogram or row computations.
inline int64_t CheckedMagnitudeTotal(int64_t low, int64_t high) {
  if (low < 0 || high < 0) throw std::overflow_error("Negative NPU magnitude limb sum");
  auto hi = static_cast<uint64_t>(high) + (static_cast<uint64_t>(low) >> 32);
  if (hi > UINT32_C(0x7fffffff)) throw std::overflow_error("NPU magnitude total exceeds int64");
  return static_cast<int64_t>((hi << 32) | (static_cast<uint64_t>(low) & UINT32_MAX));
}
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_HISTOGRAM_CONTRACT_H_
