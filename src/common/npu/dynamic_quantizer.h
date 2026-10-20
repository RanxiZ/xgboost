// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_DYNAMIC_QUANTIZER_H_
#define XGBOOST_COMMON_NPU_DYNAMIC_QUANTIZER_H_

#include "certified_scale.h"
#include "fixed_quantizer.h"

namespace xgboost::common::npu {
struct FittedQuantizedChannel {
  QuantizedChannel channel;
  ScaleCertificate certificate;
};
class DynamicQuantizer {
  Ops& ops_;
 public:
  explicit DynamicQuantizer(Ops& ops) : ops_{ops} {}
  SignificandBuckets ReduceBounds(Tensor const& input) {
    if (input.Type() != ACL_FLOAT || input.Count() > INT32_MAX) {
      throw std::invalid_argument("NPU dynamic quantizer requires 1..INT32_MAX float32 values");
    }
    // Bit decomposition avoids float32 overflow/FTZ and does not require FP64
    // NPU kernels. Invalid exponents are rejected before scatter is launched.
    auto bits = ops_.FloatBits(input);
    auto absolute = ops_.AndBits(bits, INT32_MAX);
    auto exponent = ops_.Shift(absolute, ops_.Constant(1, 23), false);
    auto invalid = ops_.Compare(exponent, 255, 0);
    if (ops_.SumControl(ops_.Cast(invalid, ACL_INT64)) != 0) {
      throw std::invalid_argument("NPU dynamic quantizer rejects NaN and infinity");
    }
    auto fraction = ops_.AndBits(absolute, (INT64_C(1) << 23) - 1);
    auto implicit = ops_.Where(ops_.Compare(exponent, 0, 0), ops_.Constant(1, 0),
                               ops_.Constant(1, INT64_C(1) << 23));
    auto mantissa = ops_.Add(fraction, implicit);
    auto offset = ops_.Where(ops_.Compare(bits, 0, -1), ops_.Constant(1, 256), ops_.Constant(1, 0));
    auto indices = ops_.Add(exponent, offset);  // exactly [0,254] or [256,510]
    auto summary = ops_.ScatterSum(indices, mantissa, 512);
    SignificandBuckets bins{};
    ops_.DownloadControl(summary, bins.data(), bins.size());
    return bins;
  }
  FittedQuantizedChannel Convert(Tensor const& input, bool hessian) {
    // CPU only assembles fixed-size bound metadata and selects the scalar scale,
    // as CUDA does. Ambiguous scale certificates fail before quantization.
    auto bins = ReduceBounds(input);
    auto certificate = CertifyScale(bins, static_cast<std::size_t>(input.Count()));
    FixedQuantizer fixed{ops_};
    auto values = fixed.Convert(input, certificate.scale, hessian);
    return {std::move(values), certificate};
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_DYNAMIC_QUANTIZER_H_
