// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_SPLIT_FINALIZATION_H_
#define XGBOOST_COMMON_NPU_SPLIT_FINALIZATION_H_
#include "precise_scoring_abi.h"
namespace xgboost::common::npu::precise {
// Scalar stages surrounding CUDA's candidate reductions. The reductions and
// their CUB tie order are deliberately NOT implemented by this component.
struct NodeInput { int64_t gradient, hessian; Parameters parameters; };
struct NodeOutput { uint32_t weight, gain, error, padding; };
struct DirectionInput { Score missing_left, missing_right; };
struct DirectionOutput { Score score; uint32_t missing_left, error; };
struct GainInput { uint32_t selected_child_gain, parent_gain; };
struct GainOutput { uint32_t loss_change, error; };
static_assert(sizeof(NodeInput)==40 && sizeof(NodeOutput)==16 &&
              sizeof(DirectionInput)==32 && sizeof(DirectionOutput)==24 &&
              sizeof(GainInput)==8 && sizeof(GainOutput)==8,"split finalization ABI layout");
XGB_NPU_PRECISE_INLINE bool FiniteFloat(uint32_t v) { return (v&0x7fffffffU)<0x7f800000U; }
XGB_NPU_PRECISE_INLINE bool ValidGain(uint32_t v) { return FiniteFloat(v) || v==0xff800000U; }
XGB_NPU_PRECISE_INLINE bool GreaterGain(uint32_t a,uint32_t b) {
  if (a==0xff800000U) return false;
  if (b==0xff800000U) return true;
  if (!(a&0x7fffffffU) && !(b&0x7fffffffU)) return false;
  if ((a^b)&0x80000000U) return !(a&0x80000000U);
  return (a&0x80000000U)?a<b:a>b;
}
// LossChangeMissing selects right for equal gains, including signed zero and
// two invalid (-infinity) directions. Neither histograms nor scores leave NPU.
XGB_NPU_PRECISE_INLINE DirectionOutput ChooseDirection(Score left,Score right) {
  uint32_t error=left.error|right.error;
  if (!ValidGain(left.gain) || !ValidGain(right.gain) ||
      !FiniteFloat(left.left_weight) || !FiniteFloat(left.right_weight) ||
      !FiniteFloat(right.left_weight) || !FiniteFloat(right.right_weight)) error|=2;
  bool choose_left=GreaterGain(left.gain,right.gain);
  return {choose_left?left:right,static_cast<uint32_t>(choose_left),error};
}
// TreeEvaluator::CalcWeight/CalcGain for the parent and root. min_child_weight
// is a split validity condition, NOT a node-weight cutoff. No monotone bounds.
XGB_NPU_PRECISE_INLINE NodeOutput NodeQuantities(int64_t gradient,int64_t hessian,Parameters p) {
  Arithmetic a;
  U lambda=a.FromFloat(p.reg_lambda),alpha=a.FromFloat(p.reg_alpha),delta=a.FromFloat(p.max_delta_step);
  a.FromFloat(p.min_child_weight);
  if ((p.min_child_weight|p.reg_lambda|p.reg_alpha|p.max_delta_step)&0x80000000U ||
      p.gradient_scale < -1022 || p.gradient_scale > 1022 ||
      p.hessian_scale < -1022 || p.hessian_scale > 1022) a.error|=2;
  if (a.error) return {0,0,a.error,0};
  U g=a.FromInt(gradient,p.gradient_scale),h=a.FromInt(hessian,p.hessian_scale);
  if (a.error) return {0,0,a.error,0};
  auto child=WeightGain(a,g,h,lambda,alpha,delta);
  return {child.weight,child.gain,a.error,0};
}
// Call only AFTER raw child-gain selection. Subtracting parent gain before
// selection can round distinct raw gains to the same float and change a tie.
XGB_NPU_PRECISE_INLINE GainOutput SubtractParent(uint32_t selected,uint32_t parent) {
  if (!ValidGain(selected) || !FiniteFloat(parent)) return {0,2};
  if (selected==0xff800000U) return {selected,0};
  Arithmetic a;
  U s=a.FromFloat(selected),p=a.FromFloat(parent);
  auto result=a.ToFloat(a.Add(s,a.Neg(p)));
  return {result,a.error};
}
}  // namespace xgboost::common::npu::precise
extern "C" void XgbNpuNodeQuantities(void* stream,void const* input,void* output,uint32_t count);
extern "C" void XgbNpuChooseDirection(void* stream,void const* left,void const* right,void* output,uint32_t count);
extern "C" void XgbNpuSubtractParent(void* stream,void const* input,void* output,uint32_t count);
#endif  // XGBOOST_COMMON_NPU_SPLIT_FINALIZATION_H_
