// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_INTEGER_HISTOGRAM_ABI_H_
#define XGBOOST_COMMON_NPU_INTEGER_HISTOGRAM_ABI_H_
#include <cstdint>
#include <limits>

namespace xgboost::common::npu::histogram {
// One descriptor per feature. Addresses are device GM addresses and bins is
// zero for a zero-bin feature, whose output tensors still contain one placeholder.
struct alignas(8) FeatureDescriptor {
  uint64_t input_column;
  uint64_t output_gradient;
  uint64_t output_hessian;
  uint32_t bins;
  uint32_t padding;
};
static_assert(sizeof(FeatureDescriptor) == 32, "histogram feature descriptor ABI");
static_assert(alignof(FeatureDescriptor) == 8, "histogram feature descriptor alignment ABI");
}  // namespace xgboost::common::npu::histogram

// Builds all feature histograms in one SIMT launch. row_ids are global row ids;
// -1 is the only missing bin value in each input column. gradient and hessian
// are the original exact int64 channels. The launch has at most 16 blocks and
// each block processes feature indices in a block-strided loop.
extern "C" void XgbNpuIntegerHistogramBuild(
    void* stream, void const* row_ids, void const* gradient, void const* hessian,
    void const* feature_metadata, uint64_t row_count, uint64_t feature_count);
#endif  // XGBOOST_COMMON_NPU_INTEGER_HISTOGRAM_ABI_H_