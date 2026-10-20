// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_QUANTIZER_SCALE_H_
#define XGBOOST_COMMON_NPU_QUANTIZER_SCALE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "../deterministic.cuh"

namespace xgboost::common::npu {
// Host-side scalar scale construction, as in CUDA BuildQuantiserFromPair.
// This does not compute the device bounds. Callers must identify the provenance
// of the four reduced bounds; supplied test bounds are not a device quantizer fit.
struct QuantizerScale {
  double to_fixed;
  double to_floating;
  int exponent;  // to_fixed == 2^exponent

  static QuantizerScale FromExponent(int k) {
    // Both reciprocal factors must be normal, finite binary64 powers of two.
    if (k < -1022 || k > 1022) throw std::invalid_argument("Invalid NPU scale exponent");
    return {std::ldexp(1.0, k), std::ldexp(1.0, -k), k};
  }
  static QuantizerScale FromBounds(double positive, double negative, std::size_t rows) {
    if (!std::isfinite(positive) || !std::isfinite(negative) || positive < 0 || negative < 0 ||
        rows == 0 || rows > (std::size_t{1} << 31) - 1) {
      throw std::invalid_argument("Invalid NPU quantizer bounds or row count");
    }
    auto bound = positive > negative ? positive : negative;
    // The shared helper uses frexp(delta). A nonfinite delta has no useful
    // exponent; reject it explicitly instead of accepting an unrelated factor.
    auto denominator = 1.0 - 2.0 * static_cast<double>(rows) * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(bound / denominator)) {
      throw std::overflow_error("NPU quantizer bound correction overflows binary64");
    }
    auto rounding = xgboost::common::CreateRoundingFactor<double>(bound, rows);
    double to_float = rounding / static_cast<double>(std::int64_t{1} << 62);
    double to_fixed = 1.0 / to_float;
    if (!std::isfinite(to_fixed) || !std::isfinite(to_float) || to_float <= 0 || to_fixed <= 0) {
      throw std::overflow_error("NPU quantizer scale is not finite and positive");
    }
    int e;
    auto mantissa = std::frexp(to_fixed, &e);
    if (mantissa != 0.5) throw std::runtime_error("NPU quantizer scale is not a power of two");
    QuantizerScale result{to_fixed, to_float, e - 1};
    result.Validate();
    return result;
  }
  void Validate() const {
    auto expected = FromExponent(exponent);
    if (to_fixed != expected.to_fixed || to_floating != expected.to_floating) {
      throw std::invalid_argument("Inconsistent NPU quantizer factors");
    }
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_QUANTIZER_SCALE_H_
