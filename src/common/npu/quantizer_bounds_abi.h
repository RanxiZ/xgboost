// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_QUANTIZER_BOUNDS_ABI_H_
#define XGBOOST_COMMON_NPU_QUANTIZER_BOUNDS_ABI_H_
#include <cstdint>

namespace xgboost::common::npu::quantizer {
// Fused dynamic-quantizer bounds summary. One exact int64 sum per float32
// sign/exponent bucket (0..254 and 256..510; 255 and 511 stay zero), followed
// by one control word counting NaN/Inf inputs. A nonzero control word means the
// host must reject the input with the original invalid_argument message.
constexpr int64_t kSignificandBucketCount = 512;
constexpr int64_t kBoundsControlIndex = kSignificandBucketCount;
constexpr int64_t kBoundsOutputCount = kSignificandBucketCount + 1;
}  // namespace xgboost::common::npu::quantizer

// Single SIMT launch replacement for the twelve-aclnn bounds chain. output must
// point at kBoundsOutputCount zeroed int64 words; input is Count float32 values.
// next_summary is cleared on device by the same launch (ping-pong buffer).
extern "C" void XgbNpuDynamicQuantizeBounds(
    void* stream, void const* input, uint64_t count, void* output, void* next_summary);
#endif  // XGBOOST_COMMON_NPU_QUANTIZER_BOUNDS_ABI_H_
