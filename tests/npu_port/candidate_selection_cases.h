// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_CANDIDATE_SELECTION_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_CANDIDATE_SELECTION_CASES_H_
#include "../../src/common/npu/candidate_selection_abi.h"
#include "split_finalization_cases.h"
namespace selection_test {
namespace p=xgboost::common::npu::precise;
namespace t=precise_test;
namespace qt=quantizer_test;
struct Case { std::vector<p::Score> left,right; };
inline p::Score Score(uint32_t gain) { return {0x80000000U,0x3f000000U,gain,0}; }
inline std::vector<Case> Cases() {
  std::vector<Case> out;
  for (unsigned n:{1,31,32,33,65,257}) {
    for (auto gain:{0x3f800000U,0U,0x80000000U,0xff800000U}) out.push_back({std::vector<p::Score>(n,Score(gain)),std::vector<p::Score>(n,Score(gain))});
    for (unsigned i=0;i<n;++i) {
      Case c{std::vector<p::Score>(n,Score(0xff800000U)),std::vector<p::Score>(n,Score(0))};
      c.right[i]=Score(0x40000000U); c.right[n-1]=Score(0x40000000U); out.push_back(c);
    }
    std::mt19937 rng{n};
    for (unsigned j=0;j<20;++j) {
      Case c;
      for (unsigned i=0;i<n;++i) {c.left.push_back(Score(t::Bits<uint32_t>(static_cast<float>(rng()%9)-4)));c.right.push_back(Score(t::Bits<uint32_t>(static_cast<float>(rng()%9)-4)));}
      out.push_back(c);
    }
  }
  for (auto invalid:{0x7f800000U,0x7fc00000U,0xffc00000U}) {
    Case c{{Score(0),Score(invalid)},{Score(0),Score(0)}};out.push_back(c);
  }
  Case c{{Score(0),Score(0)},{Score(0),Score(0)}};c.left[1].left_weight=0x7f800000U;out.push_back(c);
  c.left[1]=Score(0);c.right[1].error=8;out.push_back(c);
  out.push_back({{Score(0xff7fffffU)},{Score(0xff7fffffU)}});
  out.push_back({{Score(0xff7ffffeU)},{Score(0xff7ffffeU)}});
  out.push_back({{}, {}});
  return out;
}
inline p::Candidate Expected(Case const& c,uint32_t feature=7) {
  auto best=p::EmptyCandidate();uint32_t error=0;
  for (uint32_t i=0;i<c.left.size();++i) {
    p::DirectionInput input{c.left[i],c.right[i]};auto d=finalization_test::Expected(input);error|=d.error;
    float threshold=best.valid?t::Bits<float>(best.score.gain):-std::numeric_limits<float>::max();
    if (t::Bits<float>(d.score.gain)>threshold) best={d.score,feature,i,d.missing_left,1,0,0};
  }
  best.error=error;return best;
}
inline p::Candidate Actual(Case const& c,uint32_t feature=7) {
  auto best=p::EmptyCandidate();uint32_t errors=0;
  for (unsigned b=0;b<c.left.size();b+=32) {
    p::Candidate lanes[32];
    for (unsigned lane=0;lane<32;++lane) {
      auto i=b+lane;lanes[lane]=p::EmptyCandidate();lanes[lane].candidate=i;
      if (i<c.left.size()) {lanes[lane]=p::MakeCandidate(c.left[i],c.right[i],feature,i);errors|=lanes[lane].error;}
    }
    best=p::UpdateCandidate(best,p::WarpArgMax(lanes));
  }
  best.error|=errors;return best;
}
inline void Compare(p::Candidate a,p::Candidate b) {
  qt::Require(a.error==b.error,"candidate error mismatch");if (b.error) return;
  qt::Require(a.feature==b.feature && a.candidate==b.candidate && a.valid==b.valid && a.missing_left==b.missing_left &&
    a.score.gain==b.score.gain && a.score.left_weight==b.score.left_weight && a.score.right_weight==b.score.right_weight,"candidate identity/weight bits mismatch");
}
inline void HostTests() {
  for (auto const& c:Cases()) Compare(Actual(c),Expected(c));
  qt::Passed("candidate_lengths_ties_order_empty_nonfinite");
  std::mt19937 rng{20261017};
  for (unsigned trial=0;trial<500;++trial) {
    std::vector<uint32_t> keys(32);std::iota(keys.begin(),keys.end(),0);std::shuffle(keys.begin(),keys.end(),rng);
    p::Candidate lanes[32];
    for (unsigned i=0;i<32;++i) lanes[i]={Score((i&1)?0:0x80000000U),7,keys[i],0,1,0,0};
    auto winner=p::WarpArgMax(lanes);qt::Require(winner.candidate==0,"ArgMax is minimum key, not left operand/lane");
  }
  qt::Passed("candidate_pinned_argmax_permuted_keys_signed_zero");
  finalization_test::OrderingWitness();
  auto c=Case{{Score(0),Score(0)},{Score(0x3f800000U),Score(0x3f800001U)}};
  auto best=Actual(c);auto gain=p::SubtractParent(best.score.gain,0xce800000U);
  qt::Require(best.candidate==1 && !gain.error && gain.loss_change==p::SubtractParent(0x3f800000U,0xce800000U).loss_change,"parent subtracted before selection");
  qt::Passed("candidate_raw_gain_before_parent_subtraction");
  for (auto order:{std::vector<uint32_t>{9,1,7},std::vector<uint32_t>{7,9,1},std::vector<uint32_t>{1,7,9}}) {
    auto best=p::EmptyCandidate();for (auto f:order) best=p::UpdateCandidate(best,{Score(0),f,0,0,1,0,0});
    qt::Require(best.feature==order.front(),"feature tie chose minimum id rather than left operand");
  }
  qt::Passed("candidate_feature_left_preserving_permutations");
}
}  // namespace selection_test
#endif