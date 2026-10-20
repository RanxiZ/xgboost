// Copyright 2026, XGBoost Contributors
// Host-only contract and fixture checks; no CANN or tree builder.
#include "histogram_cases.h"
namespace ht=histogram_test;
namespace qt=quantizer_test;
namespace npu=xgboost::common::npu;
int main() {
  try {
    for (auto const& c:ht::Cases()) {
      auto parent=ht::Oracle(c,c.selected); auto [left,right]=ht::Split(c.selected);
      auto child=ht::Oracle(c,left), sibling=ht::Oracle(c,right);
      qt::Require(ht::Add(child.root_g,sibling.root_g)==parent.root_g &&
                  ht::Add(child.root_h,sibling.root_h)==parent.root_h,"root conservation fixture");
      for (std::size_t f=0;f<c.columns.size();++f) {
        for (std::size_t b=0;b<parent.gradient[f].size();++b) {
          qt::Require(ht::Add(child.gradient[f][b],sibling.gradient[f][b])==parent.gradient[f][b] &&
                      ht::Add(child.hessian[f][b],sibling.hessian[f][b])==parent.hessian[f][b],"histogram conservation fixture");
        }
      }
      qt::Passed("host_"+c.name);
    }
    auto fixture=ht::Cases().front(); auto expected=ht::Oracle(fixture,fixture.selected);
    constexpr int64_t s=INT64_C(1)<<58;
    qt::Require(expected.root_g==7*s && expected.root_h==10*s &&
                expected.gradient[0]==std::vector<int64_t>({0,4*s,4*s,0}) &&
                expected.hessian[0]==std::vector<int64_t>({4*s,3*s,2*s,0}),"CUDA-derived fixture mismatch");
    std::cout<<"CUDA_FIXTURE {\"root\":[\""<<expected.root_g<<"\",\""<<expected.root_h<<"\"],\"histogram\":[";
    for (std::size_t i=0;i<4;++i) std::cout<<(i?",":"")<<"[\""<<expected.gradient[0][i]<<"\",\""<<expected.hessian[0][i]<<"\"]";
    std::cout<<"]}\n"; qt::Passed("host_cuda_fixed_contract");
    int guards=0;
    for (auto rows:{INT64_C(0),INT64_C(-1),static_cast<int64_t>(INT32_MAX)+1}) {
      try { npu::ValidateHistogramLayout(rows,{1}); } catch (std::invalid_argument const&) { ++guards; }
    }
    for (auto bins:std::vector<std::vector<int64_t>>{{},{-1},{static_cast<int64_t>(INT32_MAX)+1}}) {
      try { npu::ValidateHistogramLayout(1,bins); } catch (std::invalid_argument const&) { ++guards; }
    }
    npu::ValidateHistogramLayout(INT32_MAX,{0,1,INT32_MAX});
    qt::Require(guards==6,"host layout guards"); qt::Passed("host_layout_guards");
    qt::Require(npu::CheckedMagnitudeTotal(0,0)==0 &&
                npu::CheckedMagnitudeTotal(UINT32_MAX,UINT32_C(0x7fffffff))==INT64_MAX &&
                npu::CheckedMagnitudeTotal(INT64_C(1)<<32,0)==(INT64_C(1)<<32),"limb assembly");
    qt::Passed("host_magnitude_limb_limits");
    int overflows=0;
    for (auto args:std::vector<std::pair<int64_t,int64_t>>{{-1,0},{0,-1},{0,INT64_C(1)<<31},
                                                        {INT64_C(1)<<32,UINT32_C(0x7fffffff)}}) {
      try { npu::CheckedMagnitudeTotal(args.first,args.second); } catch (std::overflow_error const&) { ++overflows; }
    }
    qt::Require(overflows==4,"magnitude overflow guards"); qt::Passed("host_magnitude_overflow_guards");
    std::cout<<"INTEGER_HISTOGRAM_HOST_PASS checks=14 device_execution=not_run\n"; return 0;
  } catch (std::exception const& e) {
    std::cerr<<"INTEGER_HISTOGRAM_HOST_FAIL: "<<e.what()<<'\n'; return 1;
  }
}
