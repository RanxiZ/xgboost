// Copyright 2026, XGBoost Contributors
// Host-only certificate checks. No device library or numerical CPU backend.
#include "dynamic_quantizer_cases.h"
#include <iomanip>

namespace npu = xgboost::common::npu;
namespace dt = dynamic_test;
namespace qt = quantizer_test;
namespace {
void ExportInterval(npu::SignificandBuckets const& bins, npu::ScaleCertificate const& c) {
  std::cout << "INTERVAL {\"positive\":[\"" << std::hexfloat << c.positive_exact.lower
    << "\",\"" << c.positive_exact.upper << "\"],\"negative\":[\""
    << c.negative_exact.lower << "\",\"" << c.negative_exact.upper << "\"],\"bins\":[" << std::defaultfloat;
  for (std::size_t i=0;i<bins.size();++i) std::cout << (i ? "," : "") << bins[i];
  std::cout << "]}\n";
}
}
int main() {
  try {
    for (auto const& c : dt::Cases()) {
      bool rejected=false;
      try {
        auto bins=dt::Buckets(c.input);
        auto cert=npu::CertifyScale(bins,c.input.size());
        for (bool balanced : {false,true}) {
          auto [p,n]=dt::Sums(c.input,balanced);
          qt::Require(p>=cert.positive_tree.lower && p<=cert.positive_tree.upper &&
                      n>=cert.negative_tree.lower && n<=cert.negative_tree.upper,"tree escaped certificate");
          auto reference=npu::QuantizerScale::FromBounds(p,n,c.input.size());
          qt::Require(reference.exponent==cert.scale.exponent,"certified scale differs from tree");
        }
        qt::Expected({c.name,c.input,cert.scale,c.hessian});
      } catch (npu::AmbiguousQuantizerScale const&) { if (!c.ambiguous) throw; rejected=true; }
        catch (std::invalid_argument const&) { if (!c.nonfinite) throw; rejected=true; }
      qt::Require(rejected==(c.nonfinite||c.ambiguous),"host fixture rejection mismatch");
      std::cout << "{\"test\":\"host_"<<c.name<<"\",\"status\":\"passed\",\"device_fixture\":\""
                << c.name << "\"}\n";
    }
    auto zero=npu::SignificandBuckets{};
    std::fesetround(FE_UPWARD); bool rejected=false;
    try { npu::CertifyScale(zero,1); } catch (std::invalid_argument const&) { rejected=true; }
    std::fesetround(FE_TONEAREST);
    qt::Require(rejected,"non-nearest rounding accepted"); qt::Passed("host_rounding_mode_guard");
    int rejected_rows=0;
    for (auto rows : {std::size_t{0},static_cast<std::size_t>(INT32_MAX)+1}) {
      try { npu::CertifyScale(zero,rows); } catch (std::invalid_argument const&) { ++rejected_rows; }
    }
    qt::Require(rejected_rows==2,"row-count guard failed"); qt::Passed("host_row_count_guards");
    int rejected_bins=0;
    for (auto index : {0,255,511}) {
      auto bad=zero; bad[index]=index==0 ? -1 : 1;
      try { npu::CertifyScale(bad,1); } catch (std::invalid_argument const&) { ++rejected_bins; }
    }
    auto bad=zero; bad[1]=INT64_C(1)<<24;
    try { npu::CertifyScale(bad,1); } catch (std::invalid_argument const&) { ++rejected_bins; }
    qt::Require(rejected_bins==4,"bucket guards failed"); qt::Passed("host_invalid_bucket_guards");
    double a=0x1.fffffep-1, b=0x1.fffffep-25, c=0x1.6p-50, t=0x1p-55;
    volatile double q=(a+b)+c; volatile double first=q+t; volatile double second=t+t;
    double left=first+t, right=q+second;
    qt::Require(left!=right && npu::QuantizerScale::FromBounds(left,0,5).exponent==62 &&
                npu::QuantizerScale::FromBounds(right,0,5).exponent==61,
                "expected two legitimate reduction trees with different scales");
    qt::Passed("host_reduction_order_counterexample");
    auto large=zero; large[254]=static_cast<int64_t>(INT32_MAX)*((INT64_C(1)<<24)-1);
    auto cert=npu::CertifyScale(large,INT32_MAX); ExportInterval(large,cert);
    qt::Passed("host_max_row_bucket_limit");
    for (int64_t v : {(INT64_C(1)<<53)-1, (INT64_C(1)<<53), (INT64_C(1)<<53)+1,
                       (INT64_C(1)<<53)+2, (INT64_C(1)<<53)+3, (INT64_C(1)<<54)-1,
                       (INT64_C(1)<<54)+1}) {
      auto bins=zero; bins[100]=v;
      auto result=npu::CertifyScale(bins,INT32_MAX); ExportInterval(bins,result);
    }
    qt::Passed("host_integer_cast_rounding_boundaries");
    std::mt19937_64 random{20261017};
    for (int i=0;i<128;++i) {
      auto bins=zero;
      for (int j=0;j<8;++j) bins[(random()%2)*256+(random()%255)]=static_cast<int64_t>(random()%((UINT64_C(1)<<51)-1))+1;
      auto result=npu::CertifyScale(bins,INT32_MAX); ExportInterval(bins,result);
    }
    qt::Passed("host_128_exact_interval_fixtures");
    std::cout<<"DYNAMIC_QUANTIZER_HOST_PASS checks=28 interval_fixtures=136 device_execution=not_run\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr<<"DYNAMIC_QUANTIZER_HOST_FAIL: "<<e.what()<<'\n'; return 1;
  }
}
