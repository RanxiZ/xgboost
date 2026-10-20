// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_CERTIFIED_SCALE_H_
#define XGBOOST_COMMON_NPU_CERTIFIED_SCALE_H_

#include "quantizer_scale.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>

namespace xgboost::common::npu {
// Fixed-size sufficient statistics, not per-row gradients. A float32 magnitude
// is m * 2^(max(e,1)-150). Each sign/exponent bin sums at most INT32_MAX
// unsigned 24-bit significands, hence is < 2^55 and has an exact int64 reduction.
using SignificandBuckets = std::array<int64_t, 512>;
struct BoundInterval { double lower, upper; };
struct ScaleCertificate {
  QuantizerScale scale;
  BoundInterval positive_exact, negative_exact;
  BoundInterval positive_tree, negative_tree;
};
class AmbiguousQuantizerScale : public std::runtime_error {
 public:
  AmbiguousQuantizerScale() : std::runtime_error(
      "NPU quantizer scale equality across CUDA reduction orders is uncertified; input rejected") {}
};

// CPU work is limited to assembling these 512 counters and selecting a scalar
// scale, corresponding to CUDA BuildQuantiserFromPair. No row data is consumed.
// We do NOT equate an exact sum with CUDA's FP64 tree reduction. Instead bound
// every nonnegative binary64 addition tree with gamma_(n-1), then require the
// shared CUDA scale formula to choose the same exponent over that interval.
// A conservative ambiguity is an explicit unsupported input, never a fallback.
inline ScaleCertificate CertifyScale(SignificandBuckets const& bins, std::size_t rows) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
                std::numeric_limits<float>::digits == 24);
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53 && FLT_EVAL_METHOD == 0);
  if (std::fegetround() != FE_TONEAREST) {
    throw std::invalid_argument("NPU scale certification requires round-to-nearest binary64");
  }
  if (rows == 0 || rows > INT32_MAX) throw std::invalid_argument("Invalid NPU bound row count");
  constexpr double inf = std::numeric_limits<double>::infinity();
  auto down = [](double x) { return std::nextafter(x, -std::numeric_limits<double>::infinity()); };
  auto up = [](double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); };
  auto max_bin = static_cast<int64_t>(rows) * ((INT64_C(1) << 24) - 1);
  for (std::size_t i = 0; i < bins.size(); ++i) {
    if (bins[i] < 0 || bins[i] > max_bin || (i % 256 == 255 && bins[i] != 0)) {
      throw std::invalid_argument("Invalid NPU significand bucket summary");
    }
  }
  auto assemble = [&](std::size_t offset) {
    BoundInterval sum{0, 0};
    for (int e = 0; e < 255; ++e) {
      auto value = bins[offset + e];
      if (value == 0) continue;
      double v = static_cast<double>(value);
      bool exact = static_cast<int64_t>(v) == value;  // v < 2^55; cast always safe.
      // ldexp is exact here: endpoints remain finite, normal binary64 values.
      int shift = std::max(e, 1) - 150;
      double lo = std::ldexp(exact ? v : down(v), shift);
      double hi = std::ldexp(exact ? v : up(v), shift);
      if (sum.upper == 0) sum = {lo, hi};
      else sum = {down(sum.lower + lo), up(sum.upper + hi)};
    }
    return sum;
  };
  // a and 1-a are exact for rows <= INT32_MAX. Round the division outward.
  double a = static_cast<double>(rows - 1) * std::ldexp(1.0, -53);
  double gamma = rows == 1 ? 0 : up(a / (1 - a));
  auto expand = [&](BoundInterval sum) {
    if (sum.upper == 0 || gamma == 0) return sum;
    return BoundInterval{std::max(0.0, down(sum.lower * down(1 - gamma))),
                         std::nextafter(sum.upper * up(1 + gamma), inf)};
  };
  auto positive = assemble(0), negative = assemble(256);
  auto positive_tree = expand(positive), negative_tree = expand(negative);
  auto lower = QuantizerScale::FromBounds(positive_tree.lower, negative_tree.lower, rows);
  auto upper = QuantizerScale::FromBounds(positive_tree.upper, negative_tree.upper, rows);
  if (lower.exponent != upper.exponent) throw AmbiguousQuantizerScale{};
  return {lower, positive, negative, positive_tree, negative_tree};
}
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_CERTIFIED_SCALE_H_
