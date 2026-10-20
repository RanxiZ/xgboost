// Copyright 2026, XGBoost Contributors
#include "precise_scoring_cases.h"
int main() {
  namespace t=precise_test; namespace p=xgboost::common::npu::precise;
  try {
    quantizer_test::Require(std::fegetround()==FE_TONEAREST,"rounding mode must be nearest-even");
    auto arithmetic=t::ArithmeticCases();
    for (std::size_t i=0;i<arithmetic.size();++i) {
      auto actual=t::Actual(arithmetic[i]),expected=t::Expected(arithmetic[i]);
      if ((actual.error!=0)!=(expected.error!=0) || (!expected.error && actual.value!=expected.value)) {
        std::cerr<<"arithmetic mismatch index="<<i<<" op="<<arithmetic[i].operation
          <<" a="<<std::hex<<arithmetic[i].a<<" b="<<arithmetic[i].b<<" actual="<<actual.value
          <<" expected="<<expected.value<<std::dec<<" error="<<actual.error<<" expected_error="<<expected.error<<'\n';
        throw std::runtime_error("binary64 arithmetic mismatch");
      }
    }
    quantizer_test::Passed("host_binary64_arithmetic_bitwise");
    std::size_t scores=0,groups=1;
    for (auto const& c:t::Cases()) {
      for (auto const& v:c.inputs) {
        auto actual=p::Evaluate(v.left_gradient,v.left_hessian,v.right_gradient,v.right_hessian,v.parameters);
        if (c.reject) quantizer_test::Require(actual.error!=0,"unsupported scoring domain accepted");
        else {
          auto expected=t::Expected(v);
          try { t::Compare(actual,expected); } catch (...) {
            std::cerr<<"score mismatch case="<<c.name<<" index="<<scores<<" actual="<<std::hex
              <<actual.left_weight<<','<<actual.right_weight<<','<<actual.gain<<" expected="
              <<expected.left_weight<<','<<expected.right_weight<<','<<expected.gain<<std::dec
              <<" error="<<actual.error<<'\n'; throw;
          }
        }
        ++scores;
      }
      quantizer_test::Passed("host_"+c.name); ++groups;
    }
    std::cout<<"PRECISE_SCORING_HOST_PASS groups="<<groups<<" arithmetic="<<arithmetic.size()
             <<" scores="<<scores<<" cuda_execution=not_run\n"; return 0;
  } catch (std::exception const& e) { std::cerr<<"PRECISE_SCORING_HOST_FAIL: "<<e.what()<<'\n'; return 1; }
}
