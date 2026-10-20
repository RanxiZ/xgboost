// Copyright 2026, XGBoost Contributors
// Host fixture validation only, not a split-scoring implementation.
#include "split_statistics_cases.h"
namespace st=split_statistics_test;
namespace qt=quantizer_test;
int main() {
  try {
    std::size_t checks=0;
    for (auto const& c:st::Cases()) {
      st::CheckOracle(c,c.selected);
      auto [left,right]=histogram_test::Split(c.selected);
      st::CheckOracle(c,left); st::CheckOracle(c,right);
      qt::Passed("host_"+c.name); ++checks;
    }
    auto c=st::Cases().front();
    auto g=st::Oracle(c,c.selected,0,c.gradient),h=st::Oracle(c,c.selected,0,c.hessian);
    constexpr int64_t s=INT64_C(1)<<58;
    qt::Require(g.left_missing_right==std::vector<int64_t>({0,0,4*s,8*s}) &&
                h.left_missing_right==std::vector<int64_t>({0,4*s,7*s,9*s}) &&
                g.missing==-s && h.missing==s,"pinned CUDA exclusive-prefix fixture mismatch");
    std::cout<<"CUDA_SPLIT_FIXTURE {\"prefix\":[";
    for (std::size_t i=0;i<4;++i) std::cout<<(i?",":"")<<"[\""<<g.left_missing_right[i]<<"\",\""<<h.left_missing_right[i]<<"\"]";
    std::cout<<"],\"missing\":[\""<<g.missing<<"\",\""<<h.missing<<"\"]}\n";
    qt::Passed("host_cuda_exclusive_prefix_contract"); ++checks;
    std::cout<<"SPLIT_STATISTICS_HOST_PASS checks="<<checks<<" split_scoring=not_implemented\n"; return 0;
  } catch (std::exception const& e) {
    std::cerr<<"SPLIT_STATISTICS_HOST_FAIL: "<<e.what()<<'\n'; return 1;
  }
}
