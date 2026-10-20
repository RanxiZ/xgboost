// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_QUANTIZER_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_QUANTIZER_CASES_H_
// Test-only binary64 oracle and fixtures. No tree builder or backend.
#include "../../src/common/npu/quantizer_scale.h"
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace quantizer_test {
namespace npu = xgboost::common::npu;
inline void Require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
inline float FromBits(uint32_t bits) {
  float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
struct Case {
  std::string name;
  std::vector<float> input;
  npu::QuantizerScale scale;
  bool hessian{false}, reject{false};
};
struct Oracle {
  std::vector<int64_t> values;
  int64_t root{0}, positive{0}, negative{0};
};
inline Oracle Expected(Case const& c) {
  c.scale.Validate(); Oracle result;
  for (float v : c.input) {
    double scaled = static_cast<double>(v) * c.scale.to_fixed;
    if (!std::isfinite(scaled) || std::abs(scaled) >= std::ldexp(1.0, 62)) {
      throw std::invalid_argument("test input outside quantizer safety domain");
    }
    int64_t value = static_cast<int64_t>(scaled);
    if (v > 0 && c.hessian && value == 0) value = 1;
    result.values.push_back(value);
    auto& sum = value < 0 ? result.negative : result.positive;
    auto magnitude = value < 0 ? -value : value;
    if (sum > INT64_MAX - magnitude) throw std::overflow_error("test magnitude total exceeds int64");
    sum += magnitude;
  }
  result.root = result.positive - result.negative;
  return result;
}
inline std::vector<Case> Cases() {
  using S = npu::QuantizerScale;
  auto scale = [](int k) { return S::FromExponent(k); };
  std::vector<Case> cases{
    {"fixed_cuda_gradient", {-2,1,3,0,-1,2,4,0}, scale(58)},
    {"fixed_cuda_hessian", {1,1,2,0,1,3,2,0}, scale(58), true},
    {"fractional_signed_truncation", {-3.75f,-0.75f,0,0.75f,3.75f}, scale(0)},
    {"signed_zero", {0,FromBits(0x80000000)}, scale(62), true},
    {"tiny_positive_hessian", {FromBits(1),FromBits(0x007fffff),FromBits(0x00800000),
      0.25f,0,-0.25f,FromBits(0x80000001)}, scale(0), true},
    {"tiny_gradient_truncates", {FromBits(1),FromBits(0x80000001),0.25f,-0.25f}, scale(0)},
    {"subnormal_scaled_to_large_integer", {FromBits(1),FromBits(2),FromBits(0x007fffff),
      FromBits(0x00800000),FromBits(0x80000001),FromBits(0x807fffff)}, scale(180)},
    {"max_float_negative_scale", {std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),FromBits(1)}, scale(-100)},
    {"all_zero_fitted_bounds", {0,0,0,0}, S::FromBounds(0,0,4), true},
    {"smallest_subnormal_fitted_bounds", {FromBits(1),FromBits(1)},
      S::FromBounds(2.0*static_cast<double>(FromBits(1)),0,2), true},
    {"minimum_scale_gradient", {std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),FromBits(1),FromBits(0x80000001)}, scale(-1022)},
    {"minimum_scale_hessian", {1,-1,FromBits(1),FromBits(0x80000001),0}, scale(-1022), true},
    {"maximum_scale_signed_zero", {0,FromBits(0x80000000)}, scale(1022), true},
    {"reject_maximum_scale_subnormal", {FromBits(1)}, scale(1022), false, true},
    {"near_int64_positive_sum_limit", {FromBits(0x3fffffff),FromBits(0x3fffffff)}, scale(61)},
    {"near_int64_negative_sum_limit", {FromBits(0xbfffffff),FromBits(0xbfffffff)}, scale(61)},
    {"reject_nan", {FromBits(0x7fc00001)}, scale(0), false, true},
    {"reject_positive_infinity", {FromBits(0x7f800000)}, scale(0), false, true},
    {"reject_negative_infinity", {FromBits(0xff800000)}, scale(0), true, true},
    {"reject_per_element_magnitude", {2}, scale(61), false, true},
    {"reject_positive_sum_overflow", {1.5f,1.5f,1.5f,1.5f}, scale(61), false, true},
    {"reject_negative_sum_overflow", {-1.5f,-1.5f,-1.5f,-1.5f}, scale(61), false, true}};
  std::mt19937 random{20261016};
  for (bool hessian : {false,true}) {
    std::vector<float> input(8192);
    double positive=0, negative=0;
    for (auto& v : input) {
      uint32_t bits = ((random()%254)<<23) | (random()&0x7fffff);
      if (!hessian && (random()&1)) bits |= 0x80000000;
      v=FromBits(bits);
      if (v>0) positive+=static_cast<double>(v); else negative-=static_cast<double>(v);
    }
    // Test-only host bounds; production device fitting is not claimed here.
    cases.push_back({hessian?"random_test_bounds_hessian":"random_test_bounds_gradient",
      std::move(input),S::FromBounds(positive,negative,8192),hessian});
  }
  return cases;
}
inline void Passed(std::string const& name) {
  std::cout << "{\"test\":\"" << name << "\",\"status\":\"passed\"}\n";
}
}  // namespace quantizer_test
#endif  // XGBOOST_TESTS_NPU_PORT_QUANTIZER_CASES_H_
