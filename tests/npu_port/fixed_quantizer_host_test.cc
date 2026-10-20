// Copyright 2026, XGBoost Contributors
// Host-only scale and binary64 reference checks; no CANN/CUDA/NPU execution.
#include "quantizer_cases.h"
#include <algorithm>

namespace qt = quantizer_test;
namespace npu = xgboost::common::npu;
int main() {
  try {
    qt::Require(npu::QuantizerScale::FromBounds(10,3,8).exponent==58, "gradient scale");
    qt::Require(npu::QuantizerScale::FromBounds(10,0,8).exponent==58, "Hessian scale");
    qt::Passed("scale_cuda_fixed_fixture");
    qt::Require(npu::QuantizerScale::FromBounds(0,0,8).exponent==62, "zero bound scale");
    qt::Passed("scale_all_zero");
    qt::Require(npu::QuantizerScale::FromBounds(1,0,1).exponent==61, "power-of-two bound");
    qt::Passed("scale_power_of_two_boundary");
    int invalid=0;
    for (auto args : {std::pair<double,double>{-1,0}, {0,-1},
                     {std::numeric_limits<double>::infinity(),0},
                     {std::numeric_limits<double>::quiet_NaN(),0}}) {
      try { npu::QuantizerScale::FromBounds(args.first,args.second,1); }
      catch (std::invalid_argument const&) { ++invalid; }
    }
    try { npu::QuantizerScale::FromBounds(1,0,0); }
    catch (std::invalid_argument const&) { ++invalid; }
    try { npu::QuantizerScale::FromBounds(1,0,std::size_t{1}<<31); }
    catch (std::invalid_argument const&) { ++invalid; }
    qt::Require(invalid==6, "invalid bounds accepted"); qt::Passed("scale_invalid_bounds_rejected");
    auto inconsistent=npu::QuantizerScale::FromExponent(58); inconsistent.to_fixed*=2;
    bool rejected=false;
    try { inconsistent.Validate(); } catch (std::invalid_argument const&) { rejected=true; }
    qt::Require(rejected, "inconsistent factors accepted"); qt::Passed("scale_inconsistent_factors_rejected");
    rejected=false;
    try { npu::QuantizerScale::FromBounds(std::numeric_limits<double>::max(),0,1); }
    catch (std::overflow_error const&) { rejected=true; }
    qt::Require(rejected,"overflowed bound correction accepted"); qt::Passed("scale_corrected_bound_overflow_rejected");
    rejected=false;
    try { npu::QuantizerScale::FromBounds(std::ldexp(1.0,-962),0,1); }
    catch (std::invalid_argument const&) { rejected=true; }
    qt::Require(rejected,"factory returned factors outside validator domain"); qt::Passed("scale_factory_validator_consistency");
    npu::QuantizerScale::FromExponent(-1022).Validate();
    npu::QuantizerScale::FromExponent(1022).Validate();
    qt::Passed("scale_exponent_endpoints");
    auto cases=qt::Cases();
    for (auto const& c : cases) {
      bool failed=false;
      try { auto expected=qt::Expected(c); qt::Require(!expected.values.empty(), "empty fixture"); }
      catch (std::exception const&) { failed=true; }
      qt::Require(failed==c.reject, "fixture oracle rejection differs from declared expectation");
      qt::Passed("oracle_"+c.name);
    }
    std::cout << "FIXED_QUANTIZER_HOST_PASS checks=32 device_execution=not_run\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "FIXED_QUANTIZER_HOST_FAIL: " << e.what() << '\n'; return 1;
  }
}
