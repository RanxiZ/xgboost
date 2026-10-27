// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_DYNAMIC_QUANTIZER_H_
#define XGBOOST_COMMON_NPU_DYNAMIC_QUANTIZER_H_

#include "certified_scale.h"
#include "fixed_quantizer.h"
#include "quantizer_bounds_abi.h"
#include <algorithm>
#include <array>
#include <memory>

namespace xgboost::common::npu {
struct FittedQuantizedChannel {
  QuantizedChannel channel;
  ScaleCertificate certificate;
};
class DynamicQuantizer {
  Ops& ops_;
  // Ping-pong bounds summaries (512 buckets + control word) reused across calls:
  // a fresh tensor would add an allocation, a descriptor build and an extra
  // stream synchronize on every reduction, all of which are pure host overhead.
  // The kernel clears the alternate buffer on device, so the host never memsets.
  std::unique_ptr<Tensor> summaries_[2];
  std::size_t summary_index_{0};
  Tensor& BoundsSummary() {
    if (!summaries_[0]) {
      for (auto& summary : summaries_) {
        summary = std::make_unique<Tensor>(ops_.GetRuntime(), quantizer::kBoundsOutputCount,
                                           ACL_INT64);
      }
      // Only the first buffer must start zeroed; each launch clears the other.
      summaries_[0]->Storage().Zero();
    }
    return *summaries_[summary_index_];
  }
 public:
  explicit DynamicQuantizer(Ops& ops) : ops_{ops} {}
  SignificandBuckets ReduceBounds(Tensor const& input) {
    if (input.Type() != ACL_FLOAT || input.Count() > INT32_MAX) {
      throw std::invalid_argument("NPU dynamic quantizer requires 1..INT32_MAX float32 values");
    }
    // Bit decomposition avoids float32 overflow/FTZ and does not require FP64
    // NPU kernels. One SIMT launch performs the whole decomposition: the kernel
    // keeps exact int64 bucket sums (int32 low word + carry counter in UB) and
    // reports NaN/Inf through a control word, replacing the twelve-aclnn
    // FloatBits/AndBits/Shift/Compare/SumControl/Where/Add/ScatterSum chain.
    constexpr auto kSummaryWords = static_cast<std::size_t>(quantizer::kBoundsOutputCount);
    Tensor& summary = BoundsSummary();
    Tensor& next = *summaries_[1 - summary_index_];
    XgbNpuDynamicQuantizeBounds(ops_.GetRuntime().Stream(), input.Storage().Data(),
                                static_cast<uint64_t>(input.Count()), summary.Storage().Data(),
                                next.Storage().Data());
    std::array<int64_t, kSummaryWords> raw{};
    ops_.DownloadControl(summary, raw.data(), raw.size());
    // Swap before any throw: the buffer just consumed holds stale atomics and
    // must not be reused until a later launch clears it.
    summary_index_ = 1 - summary_index_;
    if (raw[static_cast<std::size_t>(quantizer::kBoundsControlIndex)] != 0) {
      throw std::invalid_argument("NPU dynamic quantizer rejects NaN and infinity");
    }
    SignificandBuckets bins;
    std::copy_n(raw.begin(), bins.size(), bins.begin());
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
