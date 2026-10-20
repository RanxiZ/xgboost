// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_SPLIT_FINALIZATION_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_SPLIT_FINALIZATION_CASES_H_
#include "../../src/common/npu/split_finalization.h"
#include "precise_scoring_cases.h"
namespace finalization_test {
namespace p=xgboost::common::npu::precise;
namespace t=precise_test;
namespace qt=quantizer_test;
struct NodeCase { p::NodeInput input; bool reject; };
inline std::vector<NodeCase> NodeCases() {
  std::vector<NodeCase> out;
  for (auto const& c:t::Cases()) for (auto const& v:c.inputs) {
    out.push_back({{v.left_gradient,v.left_hessian,v.parameters},c.reject});
    out.push_back({{v.right_gradient,v.right_hessian,v.parameters},false});
    // For a rejected split fixture the other child may share bad parameters.
    if (c.reject) out.pop_back();
  }
  out.push_back({{3,1,t::Params(100,0)},false});
  out.push_back({{7,0,t::Params(100,0)},false});
  out.push_back({{7,-1,t::Params(100,0)},false});
  out.push_back({{0,1,t::Params(100,0)},false});
  return out;
}
inline p::NodeOutput Expected(p::NodeInput const& v) {
  auto const& p=v.parameters;
  xgboost::tree::EvalParam q{};
  q.min_child_weight=t::Bits<float>(p.min_child_weight);q.reg_lambda=t::Bits<float>(p.reg_lambda);
  q.reg_alpha=t::Bits<float>(p.reg_alpha);q.max_delta_step=t::Bits<float>(p.max_delta_step);
  using E=xgboost::tree::TreeEvaluator::SplitEvaluator<xgboost::tree::EvalParam>;
  E e{nullptr,nullptr,nullptr,false,1};
  xgboost::GradientPairPrecise s{static_cast<double>(v.gradient)*std::ldexp(1.0,-p.gradient_scale),
                               static_cast<double>(v.hessian)*std::ldexp(1.0,-p.hessian_scale)};
  return {t::Bits<uint32_t>(e.CalcWeight(0,q,s)),t::Bits<uint32_t>(e.CalcGain(0,q,s)),0,0};
}
inline void Compare(p::NodeOutput a,NodeCase const& c) {
  if (c.reject) {qt::Require(a.error!=0,"node domain accepted");return;}
  auto b=Expected(c.input);
  qt::Require(!a.error && a.weight==b.weight && a.gain==b.gain && !a.padding,"node weight/gain bit mismatch");
}
inline p::Score Score(uint32_t gain,uint32_t error=0) {
  return {t::Bits<uint32_t>(0.3f),t::Bits<uint32_t>(-0.7f),gain,error};
}
inline std::vector<p::DirectionInput> DirectionCases() {
  std::vector<p::DirectionInput> out;
  std::vector<uint32_t> values{0,0x80000000U,0xff800000U,1,0x80000001U,0x3f800000U,
    0xbf800000U,0x7f7fffffU,0xff7fffffU,0x3f800001U};
  for (auto l:values) for (auto r:values) out.push_back({Score(l),Score(r)});
  std::mt19937_64 rng{20261021};
  for (unsigned i=0;i<8192;++i) {
    uint32_t l=static_cast<uint32_t>(rng()),r=static_cast<uint32_t>(rng());
    if ((l&0x7fffffffU)>=0x7f800000U || (r&0x7fffffffU)>=0x7f800000U) continue;
    out.push_back({Score(l),Score(r)});
  }
  for (auto v:{0x7f800000U,0x7fc00000U,0xffc00001U}) out.push_back({Score(v),Score(0)});
  out.push_back({Score(0,1),Score(0)});
  out.push_back({{0x7f800000U,0,0,0},Score(0)});
  return out;
}
inline p::DirectionOutput Expected(p::DirectionInput const& v) {
  auto const& l=v.missing_left;auto const& r=v.missing_right;
  bool left=t::Bits<float>(l.gain)>t::Bits<float>(r.gain);
  auto valid_gain=[](uint32_t b) {return std::isfinite(t::Bits<float>(b)) || b==0xff800000U;};
  uint32_t error=l.error|r.error;
  if (!valid_gain(l.gain) || !valid_gain(r.gain) || !std::isfinite(t::Bits<float>(l.left_weight)) ||
      !std::isfinite(t::Bits<float>(l.right_weight)) || !std::isfinite(t::Bits<float>(r.left_weight)) ||
      !std::isfinite(t::Bits<float>(r.right_weight))) error|=2;
  return {left?l:r,static_cast<uint32_t>(left),error};
}
inline void Compare(p::DirectionOutput a,p::DirectionInput const& v) {
  auto b=Expected(v);
  qt::Require(a.error==b.error,"missing-direction rejection mismatch");
  if (b.error) return;
  qt::Require(a.missing_left==b.missing_left && a.score.left_weight==b.score.left_weight &&
    a.score.right_weight==b.score.right_weight && a.score.gain==b.score.gain && a.score.error==b.score.error,
    "missing-direction bit mismatch");
}
inline std::vector<p::GainInput> GainCases() {
  std::vector<p::GainInput> out;
  std::vector<uint32_t> v{0,0x80000000U,1,0x80000001U,0x7f7fffffU,0xff7fffffU,
                        t::Bits<uint32_t>(1.0f),t::Bits<uint32_t>(std::nextafter(1.0f,2.0f))};
  for (auto s:v) for (auto p:v) out.push_back({s,p});
  out.push_back({t::Bits<uint32_t>(1.0f),t::Bits<uint32_t>(-1073741824.0f)});
  out.push_back({t::Bits<uint32_t>(std::nextafter(1.0f,2.0f)),t::Bits<uint32_t>(-1073741824.0f)});
  std::mt19937_64 rng{20261022};
  for (unsigned i=0;i<8192;++i) out.push_back({static_cast<uint32_t>(rng()),static_cast<uint32_t>(rng())});
  out.push_back({0xff800000U,0});out.push_back({0,0xff800000U});out.push_back({0x7fc00000U,0});
  return out;
}
inline p::GainOutput Expected(p::GainInput v) {
  float s=t::Bits<float>(v.selected_child_gain),parent=t::Bits<float>(v.parent_gain);
  if ((!std::isfinite(s) && v.selected_child_gain!=0xff800000U) || !std::isfinite(parent)) return {0,2};
  if (v.selected_child_gain==0xff800000U) return {v.selected_child_gain,0};
  float gain=s-parent;
  if (!std::isfinite(gain)) return {0,1};
  return {t::Bits<uint32_t>(gain),0};
}
inline void Compare(p::GainOutput a,p::GainInput v) {
  auto b=Expected(v);qt::Require(a.error==b.error,"parent subtraction rejection mismatch");
  if (!b.error) qt::Require(a.loss_change==b.loss_change,"parent subtraction bit mismatch");
}
inline void OrderingWitness() {
  auto lo=t::Bits<uint32_t>(1.0f),hi=t::Bits<uint32_t>(std::nextafter(1.0f,2.0f));
  auto parent=t::Bits<uint32_t>(-1073741824.0f);
  auto a=Expected(p::GainInput{lo,parent}),b=Expected(p::GainInput{hi,parent});
  qt::Require(hi!=lo && a.loss_change==b.loss_change,"missing parent-subtraction ordering witness");
}
}  // namespace finalization_test
#endif  // XGBOOST_TESTS_NPU_PORT_SPLIT_FINALIZATION_CASES_H_
