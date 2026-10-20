// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_FIXED_QUANTIZER_H_
#define XGBOOST_COMMON_NPU_FIXED_QUANTIZER_H_

#include "acl_ops.h"
#include "quantizer_scale.h"
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace xgboost::common::npu {
struct QuantizedChannel {
  Tensor values;              // Remains on device; no full-row download.
  int64_t root;               // Small tree-control summary.
  int64_t positive_magnitude;
  int64_t negative_magnitude;
};
// Exact float32 -> int64 conversion for supplied binary64 power-of-two scales.
// CUDA widens float32 and multiplies by a power of two before truncation. The
// equivalent IEEE-754 significand/exponent shifts avoid requiring FP64 hardware
// and preserve subnormal input bits even if floating kernels flush denormals.
// This is NOT a complete fitted quantizer: device bound reduction and fitting
// are deliberately separate. Double-input GradientPairPrecise is unsupported.
class FixedQuantizer {
  Ops& ops_;
  int64_t MagnitudeSum(Tensor const& values) {
    auto low = ops_.AndBits(values, UINT32_MAX);
    auto high = ops_.Shift(values, ops_.Constant(1, 32), false);
    auto low_sum = ops_.SumControl(low);
    auto high_sum = ops_.SumControl(high);
    // With at most 2^31-1 rows and per-element magnitude < 2^62, both
    // independent limb reductions fit int64, regardless of reduction order.
    auto hi = static_cast<uint64_t>(high_sum) + (static_cast<uint64_t>(low_sum) >> 32);
    if (low_sum < 0 || high_sum < 0 || hi > UINT32_C(0x7fffffff)) {
      throw std::overflow_error("NPU quantized channel magnitude sum exceeds int64");
    }
    return static_cast<int64_t>((hi << 32) | (static_cast<uint64_t>(low_sum) & UINT32_MAX));
  }
 public:
  explicit FixedQuantizer(Ops& ops) : ops_{ops} {}
  QuantizedChannel Convert(Tensor const& input, QuantizerScale scale, bool hessian) {
    scale.Validate();
    if (input.Type() != ACL_FLOAT || input.Count() > INT32_MAX) {
      throw std::invalid_argument("NPU quantizer requires 1..2^31-1 float32 values");
    }
    auto bits = ops_.FloatBits(input);
    auto absolute_bits = ops_.AndBits(bits, INT32_MAX);
    auto exponent = ops_.Shift(absolute_bits, ops_.Constant(1, 23), false);
    auto fraction = ops_.AndBits(absolute_bits, (INT64_C(1) << 23) - 1);
    auto subnormal = ops_.Compare(exponent, 0, 0);
    auto zero = ops_.Constant(1, 0);
    auto one = ops_.Constant(1, 1);
    auto implicit_bit = ops_.Where(subnormal, zero, ops_.Constant(1, INT64_C(1) << 23));
    auto mantissa = ops_.Add(fraction, implicit_bit);
    auto normal_exponent = ops_.Where(subnormal, one, exponent);
    auto shift = ops_.AddScalar(normal_exponent, scale.exponent - 150);
    auto left_count = ops_.Clamp(shift, 0, 62);
    auto max_mantissa = ops_.Shift(ops_.Constant(1, (INT64_C(1) << 62) - 1), left_count, false);
    auto too_wide = ops_.Compare(ops_.Add(mantissa, max_mantissa, -1), 0, 1);
    auto nonzero = ops_.Compare(mantissa, 0, 1);
    auto oversized_shift = ops_.Logic(ops_.Compare(shift, 62, 1), nonzero, true);
    auto nonfinite = ops_.Compare(exponent, 255, 0);
    auto invalid = ops_.Logic(ops_.Logic(too_wide, oversized_shift, false), nonfinite, false);
    if (ops_.SumControl(ops_.Cast(invalid, ACL_INT64)) != 0) {
      throw std::invalid_argument("NPU quantizer input is nonfinite or exceeds CUDA safety magnitude");
    }
    // CANN 9.1.0 dispatches int64 LeftShift to the device's AI CPU. Form
    // 2^left_count with vector-core RightShift instead, then multiply exactly.
    // The safety check above guarantees that the integer product fits.
    auto complementary_shift = ops_.AddScalar(ops_.Add(zero, left_count, -1), 62);
    auto factor = ops_.Shift(ops_.Constant(1, INT64_C(1) << 62), complementary_shift, false);
    auto left = ops_.Mul(mantissa, factor);
    auto right_count = ops_.Clamp(ops_.Add(zero, shift, -1), 0, 62);
    auto right = ops_.Shift(mantissa, right_count, false);
    auto small = ops_.Where(ops_.Compare(shift, -62, -1), zero, right);
    auto magnitude0 = ops_.Where(ops_.Compare(shift, 0, -1), small, left);
    auto positive_bits = ops_.Compare(bits, 0, 1);
    auto rescue = ops_.Logic(positive_bits, ops_.Compare(magnitude0, 0, 0), true);
    auto magnitude = ops_.Where(rescue, hessian ? one : zero, magnitude0);
    auto negative = ops_.Compare(bits, 0, -1);
    auto positive_values = ops_.Where(negative, zero, magnitude);
    auto negative_values = ops_.Where(negative, magnitude, zero);
    auto positive_sum = MagnitudeSum(positive_values);
    auto negative_sum = MagnitudeSum(negative_values);
    auto signed_values = ops_.Where(negative, ops_.Add(zero, magnitude, -1), magnitude);
    auto root = ops_.SumControl(signed_values);
    if (root != positive_sum - negative_sum) {
      throw std::runtime_error("NPU integer root reduction differs from checked channel sums");
    }
    return {std::move(signed_values), root, positive_sum, negative_sum};
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_FIXED_QUANTIZER_H_
