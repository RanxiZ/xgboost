// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_CANDIDATE_SELECTION_DEVICE_H_
#define XGBOOST_COMMON_NPU_CANDIDATE_SELECTION_DEVICE_H_
#include "candidate_selection_abi.h"
namespace xgboost::common::npu::precise {
// Address-space-aware helpers shared by independently compilable device units.
__simt_callee__ inline Score LoadSelectionScore(__gm__ Score const* s) {
  return {s->left_weight,s->right_weight,s->gain,s->error};
}
__simt_callee__ inline Candidate LoadCandidate(__gm__ Candidate const* c) {
  return {LoadSelectionScore(&c->score),c->feature,c->candidate,c->missing_left,c->valid,c->error,c->padding};
}
__simt_callee__ inline void StoreCandidate(__gm__ Candidate* c,Candidate v) {
  c->score.left_weight=v.score.left_weight;c->score.right_weight=v.score.right_weight;
  c->score.gain=v.score.gain;c->score.error=v.score.error;c->feature=v.feature;c->candidate=v.candidate;
  c->missing_left=v.missing_left;c->valid=v.valid;c->error=v.error;c->padding=0;
}
}
#endif