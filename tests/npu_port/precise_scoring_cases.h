// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_PRECISE_SCORING_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_PRECISE_SCORING_CASES_H_
#include "../../src/common/npu/precise_scoring_abi.h"
#include "../../src/tree/split_evaluator.h"
#include "quantizer_cases.h"
#include "precise_arithmetic_abi.h"
#include <cfenv>
#include <utility>
namespace precise_test {
namespace p=xgboost::common::npu::precise;
namespace qt=quantizer_test;
template<typename To,typename From> To Bits(From v) {
  static_assert(sizeof(To)==sizeof(From)); To out; std::memcpy(&out,&v,sizeof(out)); return out;
}
inline ArithmeticOutput Expected(ArithmeticInput c) {
  double a=Bits<double>(c.a),b=Bits<double>(c.b),v=0; uint64_t value=0;
  if (c.operation<=2 && (!std::isfinite(a) || !std::isfinite(b) ||
      std::fpclassify(a)==FP_SUBNORMAL || std::fpclassify(b)==FP_SUBNORMAL)) return {0,1,0};
  if (c.operation==0) v=a+b;
  if (c.operation==1) v=a*b;
  if (c.operation==2) { if (b==0) return {0,1,0}; v=a/b; }
  if (c.operation==3) {
    volatile double integer=static_cast<double>(Bits<int64_t>(c.a));
    v=integer*std::ldexp(1.0,-static_cast<int>(c.b));
  }
  if (c.operation==4) v=static_cast<double>(Bits<float>(static_cast<uint32_t>(c.a)));
  if (c.operation==5) {
    if (!std::isfinite(a) || std::fpclassify(a)==FP_SUBNORMAL) return {0,1,0};
    float f=static_cast<float>(a); if (!std::isfinite(f)) return {0,1,0};
    return {Bits<uint32_t>(f),0,0};
  }
  if (!std::isfinite(v) || std::fpclassify(v)==FP_SUBNORMAL) return {0,1,0};
  value=Bits<uint64_t>(v); return {value,0,0};
}
inline ArithmeticOutput Actual(ArithmeticInput c) {
  p::Arithmetic a; uint64_t v=0;
  if (c.operation==0) v=a.Add(c.a,c.b);
  if (c.operation==1) v=a.Mul(c.a,c.b);
  if (c.operation==2) v=a.Div(c.a,c.b);
  if (c.operation==3) v=a.FromInt(Bits<int64_t>(c.a),static_cast<int>(c.b));
  if (c.operation==4) v=a.FromFloat(static_cast<uint32_t>(c.a));
  if (c.operation==5) v=a.ToFloat(c.a);
  return {v,a.error,0};
}
inline std::vector<ArithmeticInput> ArithmeticCases() {
  std::vector<ArithmeticInput> out;
  std::mt19937_64 random{20261019};
  for (uint32_t op=0;op<6;++op) {
    for (unsigned i=0;i<8192;++i) {
      uint64_t a=(random()&0x800fffffffffffffULL)|((700+random()%600)<<52);
      uint64_t b=(random()&0x800fffffffffffffULL)|((700+random()%600)<<52);
      if (op==3) { a=random(); b=random()%180; }
      if (op==4) a=static_cast<uint32_t>(random());
      out.push_back({a,b,op});
    }
  }
  std::vector<double> values{0.0,-0.0,1,-1,2,-2,0.5,-0.5,
    std::nextafter(1.0,0.0),std::nextafter(1.0,2.0),std::ldexp(1.0,-52),std::ldexp(1.0,-53),
    std::ldexp(1.0,-54),std::ldexp(1.0,-1022),std::numeric_limits<double>::max()};
  for (double a:values) for (double b:values) for (uint32_t op=0;op<3;++op) out.push_back({Bits<uint64_t>(a),Bits<uint64_t>(b),op});
  for (int64_t v:{INT64_MIN,INT64_MAX,INT64_C(0),INT64_C(1),-INT64_C(1),
                   (INT64_C(1)<<53)+1,(INT64_C(1)<<53)+3}) {
    for (int k:{0,58,1022,-1022}) out.push_back({Bits<uint64_t>(v),static_cast<uint64_t>(k),3});
  }
  for (double v:{std::ldexp(1.0,-149),std::ldexp(1.0,-150),std::ldexp(1.0,-151),
                 std::nextafter(std::ldexp(1.0,-150),1.0),std::ldexp(1.0,-126),
                 static_cast<double>(std::numeric_limits<float>::max()),
                 std::ldexp(1.0,128),std::numeric_limits<double>::denorm_min()}) {
    out.push_back({Bits<uint64_t>(v),0,5}); out.push_back({Bits<uint64_t>(-v),0,5});
  }
  for (uint32_t op=0;op<3;++op) for (unsigned i=0;i<32768;++i) {
    uint64_t a=(random()&0x800fffffffffffffULL)|((1+random()%2046)<<52);
    uint64_t b=(random()&0x800fffffffffffffULL)|((1+random()%2046)<<52);
    out.push_back({a,b,op});
  }
  for (double v:{1.0+std::ldexp(1.0,-24),1.0+3*std::ldexp(1.0,-24),
                 std::nextafter(1.0+std::ldexp(1.0,-24),0.0),
                 std::nextafter(1.0+std::ldexp(1.0,-24),2.0),3*std::ldexp(1.0,-150)}) {
    out.push_back({Bits<uint64_t>(v),0,5}); out.push_back({Bits<uint64_t>(-v),0,5});
  }
  return out;
}
struct Case { std::string name; std::vector<p::Input> inputs; bool reject{false}; };
inline p::Parameters Params(float minimum=1,float lambda=1,float alpha=0,float delta=0,int gs=0,int hs=0) {
  return {Bits<uint32_t>(minimum),Bits<uint32_t>(lambda),Bits<uint32_t>(alpha),Bits<uint32_t>(delta),gs,hs};
}
inline p::Score Expected(p::Input const& v) {
  auto const& p=v.parameters;
  xgboost::tree::EvalParam parameters{};
  parameters.min_child_weight=Bits<float>(p.min_child_weight); parameters.reg_lambda=Bits<float>(p.reg_lambda);
  parameters.reg_alpha=Bits<float>(p.reg_alpha); parameters.max_delta_step=Bits<float>(p.max_delta_step);
  using Evaluator=xgboost::tree::TreeEvaluator::SplitEvaluator<xgboost::tree::EvalParam>;
  Evaluator eval{nullptr,nullptr,nullptr,false,1};
  xgboost::GradientPairPrecise left{static_cast<double>(v.left_gradient)*std::ldexp(1.0,-p.gradient_scale),
                                  static_cast<double>(v.left_hessian)*std::ldexp(1.0,-p.hessian_scale)};
  xgboost::GradientPairPrecise right{static_cast<double>(v.right_gradient)*std::ldexp(1.0,-p.gradient_scale),
                                   static_cast<double>(v.right_hessian)*std::ldexp(1.0,-p.hessian_scale)};
  auto gain=eval.CalcSplitGain(parameters,0,0,left,right);
  if (std::isinf(gain) && gain<0) return {0,0,Bits<uint32_t>(gain),0};
  return {Bits<uint32_t>(eval.CalcWeight(0,parameters,left)),
          Bits<uint32_t>(eval.CalcWeight(0,parameters,right)),Bits<uint32_t>(gain),0};
}
inline std::vector<Case> Cases() {
  constexpr int64_t s=INT64_C(1)<<58;
  std::vector<Case> out{
    {"basic",{{-2,1,2,1,Params()},{-4,2,4,2,Params()},{0,1,0,1,Params()}}},
    {"min_child_weight_boundary",{{-2,1,2,1,Params(1)},{-2,1,2,1,Params(1.0000001f)},
                                  {-2,0,2,2,Params(0)},{-2,-1,2,2,Params(0)}}},
    {"l1_l2_delta",{{-6,3,6,3,Params(1,1,2,0.25f)},
                    {-2,1,2,1,Params(0,0,2)},{-1,1,1,1,Params(0,0,2)},
                    {-7,3,7,4,Params(0,0.2f,0.3f,0.7f)}}},
    {"quantized_large_int64",{{-2*s,s,2*s,s,Params(1,1,0,0,58,58)},
                              {INT64_MAX,INT64_MAX,-INT64_MAX,INT64_MAX,Params(0,1,0,0,62,62)},
                              {(INT64_C(1)<<53)+3,INT64_C(1)<<54,-17,3,Params(0,0.2f,0.1f,0,50,51)}}},
    {"float_subnormal_weight",{{1,1,1,1,Params(0,1,0,0,149,0)},
                              {1,1,-1,1,Params(0,0,0,0,149,0)}}},
    {"float_weight_square_and_delta_boundary",{{3,7,-5,11,Params(0,0)},
      {16777217,16777216,-16777217,16777216,Params(0,0,0,1)},
      {16777219,16777216,-16777219,16777216,Params(0,0,0,std::nextafter(1.0f,2.0f))},
      {INT64_C(1)<<54,INT64_C(1)<<54,-(INT64_C(1)<<54),INT64_C(1)<<54,Params(0,0,1,0,54,54)}}},
    {"reject_negative_parameter",{{1,1,1,1,Params(0,-1)},{1,1,1,1,Params(0,-0.0f)}},true},
    {"reject_nonfinite_parameter",{{1,1,1,1,Params(0,std::numeric_limits<float>::infinity())}},true},
    {"reject_scale_range",{{1,1,1,1,Params(0,1,0,0,1023,0)}},true},
    {"reject_binary64_overflow",{{INT64_MAX,1,1,1,Params(0,1,0,0,-1022,0)}},true},
    {"reject_float32_overflow",{{INT64_MAX,1,1,1,Params(0,0,0,0,-100,100)}},true}
  };
  Case random{"random_official_formula",{},false}; std::mt19937_64 rng{20261020};
  float params[]{0,0.1f,0.2f,0.7f,1,2};
  for (unsigned i=0;i<16384;++i) {
    int64_t lg=static_cast<int64_t>(rng()%(INT64_C(1)<<60)),rg=static_cast<int64_t>(rng()%(INT64_C(1)<<60));
    if (rng()&1) lg=-lg;
    if (rng()&1) rg=-rg;
    random.inputs.push_back({lg,static_cast<int64_t>(rng()%(INT64_C(1)<<60)),
                             rg,static_cast<int64_t>(rng()%(INT64_C(1)<<60)),
      Params(params[rng()%6],params[rng()%6],params[rng()%6],params[rng()%6],40+rng()%23,40+rng()%23)});
  }
  out.push_back(std::move(random)); return out;
}
inline void Compare(p::Score actual,p::Score expected) {
  qt::Require(!actual.error && actual.left_weight==expected.left_weight &&
              actual.right_weight==expected.right_weight && actual.gain==expected.gain,
              "weight/gain differs bitwise from official shared scalar evaluator");
}
}  // namespace precise_test
#endif  // XGBOOST_TESTS_NPU_PORT_PRECISE_SCORING_CASES_H_
