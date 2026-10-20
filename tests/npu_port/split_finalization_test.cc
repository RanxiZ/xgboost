// Copyright 2026, XGBoost Contributors
#include "split_finalization_cases.h"
#include "../../src/common/npu/precise_split_scorer.h"
#include "split_statistics_cases.h"
#include <fstream>
namespace npu=xgboost::common::npu;
namespace p=npu::precise;
namespace t=finalization_test;
namespace qt=quantizer_test;
namespace xgboost::common::npu {
class PreciseScorerTestAccess {
 public:
  static Buffer const& Left(PreciseSplitScorer::Result const& v) {return v.missing_left_;}
  static Buffer const& Right(PreciseSplitScorer::Result const& v) {return v.missing_right_;}
};
}
namespace {
unsigned checks=0;
void Passed(std::string const& name) {++checks;qt::Passed(name);}
template<typename T> npu::Buffer Upload(npu::Runtime const& r,std::vector<T> const& input) {
  auto result=r.AllocateElements<T>(input.size());result.Upload(input.data(),input.size()*sizeof(T));return result;
}
template<typename T,typename Launch,typename Compare>
void Run(npu::Runtime const& r,uint32_t count,T canary,Launch launch,Compare compare) {
  std::vector<T> output(count+2,canary);auto buffer=Upload(r,output);
  launch(static_cast<char*>(buffer.Data())+sizeof(T));r.Synchronize();
  buffer.Download(output.data(),output.size()*sizeof(T));
  qt::Require(std::memcmp(&output.front(),&canary,sizeof(T))==0 &&
              std::memcmp(&output.back(),&canary,sizeof(T))==0,"finalization output tail canary changed");
  for (uint32_t i=0;i<count;++i) compare(output[i+1],i);
}
void Fixtures(npu::Runtime const& r) {
  auto cases=t::NodeCases();std::vector<p::NodeInput> input;for (auto c:cases) input.push_back(c.input);
  auto nodes=Upload(r,input);
  Run(r,input.size(),p::NodeOutput{0x12345678U,0x23456789U,0x34567890U,0x45678901U},
    [&](void* out){XgbNpuNodeQuantities(r.Stream(),nodes.Data(),out,input.size());},
    [&](p::NodeOutput actual,uint32_t i){t::Compare(actual,cases[i]);});
  Passed("npu_node_quantities_bitwise");
  auto directions=t::DirectionCases();std::vector<p::Score> left,right;
  for (auto d:directions) {left.push_back(d.missing_left);right.push_back(d.missing_right);}
  auto l=Upload(r,left),rr=Upload(r,right);
  constexpr p::DirectionOutput dc{{0x12345678U,0x23456789U,0x34567890U,0x45678901U},7,8};
  Run(r,directions.size(),dc,
    [&](void* out){XgbNpuChooseDirection(r.Stream(),l.Data(),rr.Data(),out,directions.size());},
    [&](p::DirectionOutput actual,uint32_t i){t::Compare(actual,directions[i]);});
  Passed("npu_missing_direction_bitwise");
  auto gaincases=t::GainCases();auto gains=Upload(r,gaincases);
  Run(r,gaincases.size(),p::GainOutput{0x12345678U,0x23456789U},
    [&](void* out){XgbNpuSubtractParent(r.Stream(),gains.Data(),out,gaincases.size());},
    [&](p::GainOutput actual,uint32_t i){t::Compare(actual,gaincases[i]);});
  Passed("npu_parent_subtraction_bitwise");
}
void Tail(npu::Runtime const& r) {
  for (uint32_t n:{1,127,128,129,2049}) {
    p::NodeInput node{-3,1,precise_test::Params(100,1)};
    auto input=Upload(r,std::vector<p::NodeInput>(n,node));
    Run(r,n,p::NodeOutput{1,2,3,4},
      [&](void* out){XgbNpuNodeQuantities(r.Stream(),input.Data(),out,n);},
      [&](p::NodeOutput actual,uint32_t){t::Compare(actual,t::NodeCase{node,false});});
    p::DirectionInput direction{t::Score(0),t::Score(0x80000000U)};
    auto l=Upload(r,std::vector<p::Score>(n,direction.missing_left));
    auto right=Upload(r,std::vector<p::Score>(n,direction.missing_right));
    Run(r,n,p::DirectionOutput{{1,2,3,4},5,6},
      [&](void* out){XgbNpuChooseDirection(r.Stream(),l.Data(),right.Data(),out,n);},
      [&](p::DirectionOutput actual,uint32_t){t::Compare(actual,direction);});
    p::GainInput gain{0x3f800001U,precise_test::Bits<uint32_t>(-1073741824.0f)};
    auto gi=Upload(r,std::vector<p::GainInput>(n,gain));
    Run(r,n,p::GainOutput{1,2},
      [&](void* out){XgbNpuSubtractParent(r.Stream(),gi.Data(),out,n);},
      [&](p::GainOutput actual,uint32_t){t::Compare(actual,gain);});
  }
  Passed("npu_finalization_grid_tail_canaries");
}
npu::Tensor Tensor(npu::Runtime const& r,std::vector<int64_t> const& v) {
  npu::Tensor out{r,static_cast<int64_t>(v.size()),ACL_INT64};out.Storage().Upload(v.data(),v.size()*8);return out;
}
void Integration(npu::Runtime const& r) {
  npu::Ops ops{r};
  for (auto const& c:split_statistics_test::Cases()) {
    std::vector<npu::Tensor> columns;for (auto const& v:c.columns) columns.push_back(Tensor(r,v));
    npu::IntegerHistogramBuilder builder{ops,std::move(columns),Tensor(r,c.gradient),Tensor(r,c.hessian),c.bin_counts};
    auto ids=c.selected.empty()?nullptr:std::make_unique<npu::Tensor>(Tensor(r,c.selected));
    auto hist=builder.Build(ids.get());npu::NumericalSplitStatistics statistics{ops};npu::PreciseSplitScorer scorer{ops};
    auto params=precise_test::Params(0,1,0.2f,0.7f,58,58);
    for (std::size_t f=0;f<c.bin_counts.size();++f) {
      auto candidate=statistics.Build(hist,f);if (!candidate) continue;
      auto scores=scorer.Evaluate(*candidate,params);
      auto const& l=npu::PreciseScorerTestAccess::Left(scores);auto const& rr=npu::PreciseScorerTestAccess::Right(scores);
      auto g=split_statistics_test::Oracle(c,c.selected,f,c.gradient);
      auto h=split_statistics_test::Oracle(c,c.selected,f,c.hessian);
      Run(r,c.bin_counts[f],p::DirectionOutput{{1,2,3,4},5,6},
        [&](void* out){XgbNpuChooseDirection(r.Stream(),l.Data(),rr.Data(),out,c.bin_counts[f]);},
        [&](p::DirectionOutput actual,uint32_t i) {
          auto left=precise_test::Expected(p::Input{g.left_missing_left[i],h.left_missing_left[i],
            g.right_missing_left[i],h.right_missing_left[i],params});
          auto right=precise_test::Expected(p::Input{g.left_missing_right[i],h.left_missing_right[i],
            g.right_missing_right[i],h.right_missing_right[i],params});
          t::Compare(actual,p::DirectionInput{left,right});
        });
    }
  }
  Passed("npu_histogram_statistics_scores_direction");
}
void Tests() {
  int errors=0;auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error=[&](std::string const& message){++errors;std::cerr<<"cleanup failure: "<<message<<'\n';};
  {
    npu::Runtime r{api,0};Fixtures(r);Tail(r);Integration(r);
    qt::Require(!r.LiveBytes(),"finalization device buffers leaked");
  }
  qt::Require(!errors,"finalization cleanup failed");Passed("npu_finalization_resources_released");
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" && std::string(argv[2])=="0" &&
              std::string(argv[3])=="--confirm-device-idle";
  if (!host && !device) {std::cerr<<"Usage: split_finalization_test --host-api-only | --device-test 0 --confirm-device-idle\n";return 2;}
  try {
    npu::MakeAclApi()->Validate();
    if (host) {
      XgbNpuNodeQuantities(nullptr,nullptr,nullptr,0);XgbNpuChooseDirection(nullptr,nullptr,nullptr,nullptr,0);
      XgbNpuSubtractParent(nullptr,nullptr,nullptr,0);
      std::cout<<"SPLIT_FINALIZATION_API_HOST_PASS\n";return 0;
    }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try {Tests();} catch (...) {aclFinalize();throw;}
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    std::cout<<"SPLIT_FINALIZATION_NPU_PASS checks="<<checks<<" candidate_reduction=not_implemented training_backend=false\n";
    return 0;
  } catch (std::exception const& e) {std::cerr<<"SPLIT_FINALIZATION_FAIL: "<<e.what()<<'\n';return 1;}
}
