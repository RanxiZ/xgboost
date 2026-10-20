// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_NUMERICAL_SPLIT_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_NUMERICAL_SPLIT_CASES_H_
#include "../../src/common/npu/numerical_split_evaluator_abi.h"
#include "candidate_selection_cases.h"
#include "split_statistics_cases.h"
#include <numeric>
namespace numerical_test {
namespace p=xgboost::common::npu::precise;
namespace t=precise_test;
namespace qt=quantizer_test;
namespace st=split_statistics_test;
inline std::vector<histogram_test::Case> Cases() {
  auto cases=st::Cases();
  for (int n:{31,32,33,65,257}) cases.push_back({"ties_"+std::to_string(n),{-3,1,7,5},{1,2,3,4},{n,n,0},{{0,n-1,n/2,-1},{0,n-1,n/2,-1},{-1,-1,-1,-1}},{0,1,2,3}});
  return cases;
}
inline p::SplitSummary Expected(histogram_test::Case const& c,std::vector<int64_t> const& rows,p::Parameters parameters,std::vector<uint32_t> const& order) {
  p::SplitSummary out{};out.best=p::EmptyCandidate();
  for (auto i:rows) {out.gradient+=c.gradient[i];out.hessian+=c.hessian[i];}
  out.node=finalization_test::Expected(p::NodeInput{out.gradient,out.hessian,parameters});
  // Independent host CUDA tree evaluator for scores. CUB's 256 striped threads
  // are modeled separately from the implementation's host metadata traversal.
  struct Entry { p::Candidate best;int64_t lg,lh,rg,rh; };
  std::vector<Entry> features;
  for (auto f:order) {
    Entry entry{p::EmptyCandidate(),0,0,0,0};
    auto g=st::Oracle(c,rows,f,c.gradient),h=st::Oracle(c,rows,f,c.hessian);
    for (uint32_t i=0;i<static_cast<uint32_t>(c.bin_counts[f]);++i) {
      auto l=t::Expected(p::Input{g.left_missing_left[i],h.left_missing_left[i],g.right_missing_left[i],h.right_missing_left[i],parameters});
      auto r=t::Expected(p::Input{g.left_missing_right[i],h.left_missing_right[i],g.right_missing_right[i],h.right_missing_right[i],parameters});
      auto d=finalization_test::Expected(p::DirectionInput{l,r});
      if (t::Bits<float>(d.score.gain)>t::Bits<float>(entry.best.score.gain)) {
        entry.best={d.score,f,i,d.missing_left,1,0,0};
        entry.lg=d.missing_left?g.left_missing_left[i]:g.left_missing_right[i];
        entry.lh=d.missing_left?h.left_missing_left[i]:h.left_missing_right[i];
        entry.rg=d.missing_left?g.right_missing_left[i]:g.right_missing_right[i];
        entry.rh=d.missing_left?h.right_missing_left[i]:h.right_missing_right[i];
      }
    }
    features.push_back(entry);
  }
  Entry threads[256];for (auto& v:threads) v={p::EmptyCandidate(),0,0,0,0};
  for (std::size_t i=0;i<features.size();++i) if (t::Bits<float>(features[i].best.score.gain)>t::Bits<float>(threads[i%256].best.score.gain)) threads[i%256]=features[i];
  Entry best{p::EmptyCandidate(),0,0,0,0};
  for (auto const& v:threads) if (t::Bits<float>(v.best.score.gain)>t::Bits<float>(best.best.score.gain)) best=v;
  out.best=best.best;out.left_gradient=best.lg;out.left_hessian=best.lh;out.right_gradient=best.rg;out.right_hessian=best.rh;
  out.loss_change=finalization_test::Expected(p::GainInput{out.best.score.gain,out.node.gain}).loss_change;
  return out;
}
inline void Compare(p::SplitSummary a,p::SplitSummary b) {
  selection_test::Compare(a.best,b.best);
  qt::Require(!a.error && !a.node.error && a.node.weight==b.node.weight && a.node.gain==b.node.gain && a.loss_change==b.loss_change &&
    a.gradient==b.gradient && a.hessian==b.hessian && a.left_gradient==b.left_gradient && a.left_hessian==b.left_hessian &&
    a.right_gradient==b.right_gradient && a.right_hessian==b.right_hessian,"numerical evaluator summary mismatch");
}
inline void HostTests() {
  auto good=t::Params();qt::Require(p::ValidParameters(good),"valid parameters rejected");
  for (uint32_t invalid:{0x80000000U,0xbf800000U,0x7f800000U,0x7fc00000U,0xff800000U}) {
    for (unsigned field=0;field<4;++field) {
      auto v=good;
      if (field==0) v.min_child_weight=invalid;
      if (field==1) v.reg_lambda=invalid;
      if (field==2) v.reg_alpha=invalid;
      if (field==3) v.max_delta_step=invalid;
      qt::Require(!p::ValidParameters(v),"invalid parameter bits accepted");
    }
  }
  for (int invalid:{-1023,1023,INT32_MIN,INT32_MAX}) {
    auto v=good;v.gradient_scale=invalid;qt::Require(!p::ValidParameters(v),"gradient scale accepted");
    v=good;v.hessian_scale=invalid;qt::Require(!p::ValidParameters(v),"hessian scale accepted");
  }
  qt::Passed("numerical_strict_parameter_and_scale_guards");
  auto v=p::NodeQuantities(1,1,t::Params(0,0,0,0,-1022,1022));qt::Require(v.error,"unsupported arithmetic domain accepted");
  qt::Passed("numerical_strict_arithmetic_domain_rejection");
  for (auto const& c:Cases()) {
    std::vector<uint32_t> order(c.bin_counts.size());std::iota(order.begin(),order.end(),0);
    auto s=Expected(c,c.selected,t::Params(0,1,0.2f,0.7f,58,58),order);
    qt::Require(!s.best.valid || s.best.feature<c.bin_counts.size(),"reference feature");
  }
  qt::Passed("numerical_host_histogram_statistics_score_selection_oracle");
  histogram_test::Case c{"feature_stripes",{-3,3},{1,1},std::vector<int64_t>(257,1),std::vector<std::vector<int64_t>>(257,std::vector<int64_t>{0,0}),{0,1}};
  c.bin_counts[1]=2;c.columns[1]={0,1};c.bin_counts[256]=2;c.columns[256]={0,1};
  std::vector<uint32_t> order(257);std::iota(order.begin(),order.end(),0);
  auto s=Expected(c,c.selected,t::Params(0,1),order);qt::Require(s.best.feature==256,"CUB block stripes lost tied position 256 over 1");
  qt::Passed("numerical_pinned_feature_stripes_257");
}
}
#endif