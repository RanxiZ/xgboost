// Copyright 2026, XGBoost Contributors
#include "row_partition_cases.h"
namespace rt = row_partition_test;
namespace qt = quantizer_test;
int main() {
  try {
    std::size_t checks=0;
    for (auto const& c : rt::Cases()) {
      auto out=rt::Oracle(c);
      rt::Require(out.left.size()+out.right.size()==c.selected.size(), "lost rows");
      auto joined=out.left; joined.insert(joined.end(),out.right.begin(),out.right.end());
      auto sorted=c.selected; std::sort(joined.begin(),joined.end()); std::sort(sorted.begin(),sorted.end());
      rt::Require(joined==sorted, "changed membership");
      if (c.name=="fixed_i2_missing_left") {
        rt::Require(out.left==std::vector<int64_t>({0,1,2,4,5,7}) &&
                    out.right==std::vector<int64_t>({3,6}), "fixed source fixture mismatch");
      }
      qt::Passed("host_"+c.name); ++checks;
    }
    rt::RoutingThresholdFixture(); qt::Passed("host_raw_vs_cut_threshold"); ++checks;
    auto base=rt::Cases().front();
    for (int bad=0;bad<9;++bad) {
      auto c=base;
      if (bad==0) c.selected={-1};
      if (bad==1) c.selected={8};
      if (bad==2) c.selected={INT64_MAX};
      if (bad==3) c.selected={0,0};
      if (bad==4) c.bins[0]=-2;
      if (bad==5) c.bins[0]=4;
      if (bad==6) c.split.candidate=-1;
      if (bad==7) c.split.candidate=4;
      if (bad==8) c.split.feature=1;
      bool rejected=false; try { rt::Oracle(c); } catch (std::invalid_argument const&) { rejected=true; }
      rt::Require(rejected,"accepted invalid fixture"); qt::Passed("host_reject_"+std::to_string(bad)); ++checks;
    }
    bool rejected=false;
    try { rt::npu::ValidateRowSplit({0},{0,0,false}); } catch (std::invalid_argument const&) { rejected=true; }
    rt::Require(rejected,"zero-bin candidate accepted"); qt::Passed("host_zero_bin_candidate_rejected"); ++checks;
    std::cout<<"ROW_PARTITION_HOST_PASS checks="<<checks<<" cuda_execution=not_run npu_execution=not_run\n";
    return 0;
  } catch (std::exception const& e) { std::cerr<<"ROW_PARTITION_HOST_FAIL: "<<e.what()<<'\n'; return 1; }
}