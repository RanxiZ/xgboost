// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_DYNAMIC_QUANTIZER_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_DYNAMIC_QUANTIZER_CASES_H_
#include "quantizer_cases.h"
#include "../../src/common/npu/certified_scale.h"
#include <algorithm>

namespace dynamic_test {
namespace npu = xgboost::common::npu;
namespace qt = quantizer_test;
struct Case {
  std::string name;
  std::vector<float> input;
  bool hessian{false}, nonfinite{false}, ambiguous{false};
};
inline npu::SignificandBuckets Buckets(std::vector<float> const& input) {
  npu::SignificandBuckets bins{};
  for (float f : input) {
    uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
    auto e = (bits >> 23) & 255;
    if (e == 255) throw std::invalid_argument("Nonfinite test input");
    bins[(bits >> 31) * 256 + e] += (bits & 0x7fffff) + (e ? (1 << 23) : 0);
  }
  return bins;
}
// Test-only serial and pairwise binary64 nonnegative CUDA Clip-style sums.
inline std::pair<double,double> Sums(std::vector<float> const& input, bool balanced) {
  auto reduce = [&](bool negative) {
    std::vector<double> values;
    for (auto f : input) values.push_back(negative ? (f < 0 ? -static_cast<double>(f) : 0)
                                                  : (f > 0 ? static_cast<double>(f) : 0));
    if (!balanced) {
      double total = 0; for (auto v : values) total += v; return total;
    }
    while (values.size() > 1) {
      std::vector<double> next;
      for (std::size_t i = 0; i < values.size(); i += 2) {
        next.push_back(i + 1 < values.size() ? values[i] + values[i+1] : values[i]);
      }
      values = std::move(next);
    }
    return values.front();
  };
  return {reduce(false), reduce(true)};
}
inline std::vector<Case> Cases() {
  auto tiny = qt::FromBits(1);
  auto max = std::numeric_limits<float>::max();
  std::vector<Case> cases{
    {"cuda_gradient", {-2,1,3,0,-1,2,4,0}},
    {"cuda_hessian", {1,1,2,0,1,3,2,0}, true},
    {"signed_zero", {0,qt::FromBits(0x80000000)}, true},
    {"single_power_of_two", {1}},
    {"single_subnormal", {tiny}, true},
    {"single_max_float", {max}},
    {"single_negative_normal", {-1}},
    {"single_negative_subnormal", {-tiny}, true},
    {"single_negative_max_float", {-max}},
    {"reject_conservative_interval", {0x1.fffffep-1f,0x1.fffffep-25f,0x1.4p-49f}, false, false, true},
    {"tiny_hessian_rescue", {max,tiny,0}, true},
    {"negative_hessian_conversion", {-1,tiny,-tiny,0.25f}, true},
    {"reject_nan", {qt::FromBits(0x7fc00001)}, false, true},
    {"reject_positive_infinity", {qt::FromBits(0x7f800000)}, false, true},
    {"reject_negative_infinity", {qt::FromBits(0xff800000)}, true, true},
    {"reject_reduction_order_positive", {0x1.fffffep-1f,0x1.fffffep-25f,0x1.6p-50f,
                                         0x1p-55f,0x1p-55f}, false, false, true},
    {"reject_reduction_order_negative", {-0x1.fffffep-1f,-0x1.fffffep-25f,-0x1.6p-50f,
                                         -0x1p-55f,-0x1p-55f}, false, false, true}};
  std::vector<float> spectrum;
  for (uint32_t e = 0; e < 255; ++e) {
    spectrum.push_back(qt::FromBits((e << 23) | 0x7fffff));
    spectrum.push_back(qt::FromBits(0x80000000 | (e << 23) | 1));
  }
  cases.push_back({"all_finite_exponents", std::move(spectrum)});
  cases.push_back({"high_collision", std::vector<float>(131071,3.5f), true});
  for (auto const& c : qt::Cases()) {
    if (c.name.find("random_test_bounds_") == 0) {
      cases.push_back({c.hessian ? "random_hessian" : "random_gradient", c.input, c.hessian});
    }
  }
  return cases;
}
}  // namespace dynamic_test
#endif  // XGBOOST_TESTS_NPU_PORT_DYNAMIC_QUANTIZER_CASES_H_
