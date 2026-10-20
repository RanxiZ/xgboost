// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_PRECISE_SCORING_ABI_H_
#define XGBOOST_COMMON_NPU_PRECISE_SCORING_ABI_H_
#include "precise_scoring.h"
namespace xgboost::common::npu::precise {
struct Input { int64_t left_gradient,left_hessian,right_gradient,right_hessian; Parameters parameters; };
static_assert(sizeof(Input)==56 && sizeof(Parameters)==24 && sizeof(Score)==16,"NPU scoring ABI layout mismatch");
}
// Device pointers only; caller owns private context, stream, buffers and sync.
// The wrapper launches an Ascend vector-core kernel, never host arithmetic.
extern "C" void XgbNpuPreciseScore(void* stream,void const* input,void* output,uint32_t count);
extern "C" void XgbNpuScoreColumns(void* stream,void const* lg,void const* lh,void const* rg,void const* rh,
                                   void const* parameters,void* output,void* errors,uint32_t count);
#endif  // XGBOOST_COMMON_NPU_PRECISE_SCORING_ABI_H_
