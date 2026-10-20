// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_CANDIDATE_SELECTION_ABI_H_
#define XGBOOST_COMMON_NPU_CANDIDATE_SELECTION_ABI_H_
#include "split_finalization.h"
namespace xgboost::common::npu::precise {
// Pinned CCCL 2.8.2 contract; see tests/npu_port/candidate_selection_contract.json.
struct Candidate {
  Score score;
  uint32_t feature, candidate, missing_left, valid;
  uint32_t error, padding;
};
static_assert(sizeof(Candidate)==40,"candidate ABI");
XGB_NPU_PRECISE_INLINE Candidate EmptyCandidate() {
  return {{0,0,0xff800000U,0},UINT32_MAX,UINT32_MAX,0,0,0,0};
}
// Exact cub::ArgMax rule, not a left-preserving generic maximum. Signed zeros
// compare equal. The lane key resolves ties independently of tree shape.
XGB_NPU_PRECISE_INLINE Candidate LaneArgMax(Candidate a,Candidate b) {
  bool equal=!GreaterGain(a.score.gain,b.score.gain)&&!GreaterGain(b.score.gain,a.score.gain);
  return GreaterGain(b.score.gain,a.score.gain)||(equal&&b.candidate<a.candidate)?b:a;
}
// CUB shuffle-down steps 1,2,4,8,16, each reading the previous step's lanes.
XGB_NPU_PRECISE_INLINE Candidate WarpArgMax(Candidate* lanes) {
  for (uint32_t offset=1;offset<32;offset*=2) {
    for (uint32_t lane=0;lane+offset<32;++lane) lanes[lane]=LaneArgMax(lanes[lane],lanes[lane+offset]);
  }
  return lanes[0];
}
// CUDA's strict-greater chunk update and left-preserving across-feature op.
// Invalid (-inf) entries never replace the initial null split.
XGB_NPU_PRECISE_INLINE Candidate UpdateCandidate(Candidate left,Candidate right) {
  uint32_t errors=left.error|right.error;
  // CUDA initializes DeviceSplitCandidate::loss_chg to -FLT_MAX. Keep our
  // externally normalized invalid sentinel (-inf), but preserve that strict
  // initial comparison as well, including the exact -FLT_MAX edge case.
  if (right.valid && GreaterGain(right.score.gain,left.score.gain) &&
      (left.valid || GreaterGain(right.score.gain,0xff7fffffU))) left=right;
  left.error=errors;
  return left;
}
XGB_NPU_PRECISE_INLINE Candidate MakeCandidate(Score left,Score right,uint32_t feature,uint32_t index) {
  auto d=ChooseDirection(left,right);
  return {d.score,feature,index,d.missing_left,static_cast<uint32_t>(d.score.gain!=0xff800000U),d.error,0};
}
}  // namespace xgboost::common::npu::precise
// Device pointers only. count==0 is a no-op, including with null pointers.
extern "C" void XgbNpuSelectCandidates(void* stream,void const* left,void const* right,
                                      void* output,uint32_t count,uint32_t feature);
#endif