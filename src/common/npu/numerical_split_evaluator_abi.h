// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_NUMERICAL_SPLIT_EVALUATOR_ABI_H_
#define XGBOOST_COMMON_NPU_NUMERICAL_SPLIT_EVALUATOR_ABI_H_
#include <cstddef>
#include "candidate_selection_abi.h"
namespace xgboost::common::npu::precise {
// The sole evaluator download, fixed size independent of bins/features.
// Float fields are IEEE binary32 bits to preserve signed zero exactly.
struct SplitSummary {
  Candidate best;
  NodeOutput node;
  int64_t gradient,hessian,left_gradient,left_hessian,right_gradient,right_hessian;
  uint32_t loss_change,error;
};
static_assert(sizeof(SplitSummary)==112,"split summary ABI");
// One upload: 32-byte header followed immediately by feature_count descriptors.
// Descriptors retain ORIGINAL sampled feature positions, not stripe order.
struct alignas(8) NodeBatchHeader {
  Parameters parameters;
  uint32_t feature_count, padding;
};
struct alignas(8) NodeBatchFeature {
  uint64_t gradient_address, hessian_address;
  uint64_t prefix_offset, candidate_offset;  // Element offsets, never byte offsets.
  uint32_t bin_count, feature;
};
static_assert(sizeof(NodeBatchHeader)==32 && alignof(NodeBatchHeader)==8 &&
              offsetof(NodeBatchHeader,feature_count)==24,"node batch header ABI");
static_assert(sizeof(NodeBatchFeature)==40 && alignof(NodeBatchFeature)==8 &&
              offsetof(NodeBatchFeature,bin_count)==32,"node batch feature ABI");
// Each feature reserves bins+1 entries in EACH prefix channel and bins entries
// in DirectionOutput scored[]. Packed uint32 status[] is [prefix:F, score:N,
// selected:F]. Fresh status storage is zeroed once before ANY kernel; producers
// store 1 only AFTER payload. Consumers reject !=1 with error bit 4 and never
// read unwritten payload. Same-stream kernel boundaries provide visibility.
constexpr uint32_t kBatchIncomplete=4;
XGB_NPU_PRECISE_INLINE bool ValidParameters(Parameters p) {
  return p.min_child_weight<0x7f800000U && p.reg_lambda<0x7f800000U &&
    p.reg_alpha<0x7f800000U && p.max_delta_step<0x7f800000U &&
    p.gradient_scale>=-1022 && p.gradient_scale<=1022 &&
    p.hessian_scale>=-1022 && p.hessian_scale<=1022;
}
}
extern "C" void XgbNpuSplitStatistics(void* stream,void const* bins,void const* parent,
                                       void* missing,void* left_missing_right,
                                       void* right_missing_right,void* left_missing_left,
                                       void* right_missing_left,uint32_t count);
extern "C" void XgbNpuSplitInitialize(void* stream,void const* gradient,void const* hessian,
                                     void const* parameters,void* summary);
extern "C" void XgbNpuSplitMerge(void* stream,void const* selected,void const* columns,void* summary);
extern "C" void XgbNpuSplitFinish(void* stream,void* summary);
// Always one launch per entry point, including feature_count==0. count is the
// validated header count, supplied separately only to choose the launch grid.
extern "C" void XgbNpuBatchedSplitPrefix(void* stream,void const* metadata,
    void* prefix_gradient,void* prefix_hessian,void* prefix_status,uint32_t count);
extern "C" void XgbNpuBatchedSplitScore(void* stream,void const* metadata,
    void const* gradient,void const* hessian,void const* prefix_gradient,
    void const* prefix_hessian,void const* prefix_status,void* scores,
    void* score_status,uint32_t count);
extern "C" void XgbNpuBatchedSplitSelect(void* stream,void const* metadata,
    void const* prefix_status,void const* scores,void const* score_status,
    void* selected,void* selected_status,uint32_t count);
extern "C" void XgbNpuBatchedSplitFinish(void* stream,void const* metadata,
    void const* gradient,void const* hessian,void const* prefix_gradient,
    void const* prefix_hessian,void const* prefix_status,void const* selected,
    void const* selected_status,void* summary);
#endif