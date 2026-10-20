// Copyright 2026, XGBoost Contributors
#include "../../src/common/npu/precise_split_scorer.h"
#include "../../src/common/npu/dynamic_quantizer.h"
#include "precise_scoring_cases.h"
#include "split_statistics_cases.h"
#include <dlfcn.h>
#include <fstream>
namespace npu=xgboost::common::npu;
namespace p=npu::precise;
namespace t=precise_test;
namespace st=split_statistics_test;
namespace qt=quantizer_test;
using Builder=npu::IntegerHistogramBuilder;
namespace xgboost::common::npu {
class PreciseScorerTestAccess {
 public:
  static Buffer const& Left(PreciseSplitScorer::Result const& r) { return r.missing_left_; }
  static Buffer const& Right(PreciseSplitScorer::Result const& r) { return r.missing_right_; }
};
}
namespace {
std::size_t checks=0;
void Passed(std::string const& name) { ++checks; qt::Passed(name); }
template<typename T> npu::Buffer Upload(npu::Runtime const& r,std::vector<T> const& values) {
  auto out=r.AllocateElements<T>(values.size()); out.Upload(values.data(),values.size()*sizeof(T)); return out;
}
npu::Tensor UploadTensor(npu::Runtime const& r,std::vector<int64_t> const& values) {
  npu::Tensor out{r,static_cast<int64_t>(values.size()),ACL_INT64};
  out.Storage().Upload(values.data(),values.size()*sizeof(int64_t)); return out;
}
void Arithmetic(npu::Runtime const& r) {
  auto cases=t::ArithmeticCases(); auto input=Upload(r,cases);
  auto output=r.AllocateElements<t::ArithmeticOutput>(cases.size()); output.Zero();
  XgbNpuTestArithmetic(r.Stream(),input.Data(),output.Data(),cases.size()); r.Synchronize();
  std::vector<t::ArithmeticOutput> actual(cases.size()); output.Download(actual.data(),actual.size()*sizeof(actual[0]));
  for (std::size_t i=0;i<cases.size();++i) {
    auto expected=t::Expected(cases[i]);
    if ((actual[i].error!=0)!=(expected.error!=0) || (!expected.error && actual[i].value!=expected.value)) {
      std::cerr<<"NPU arithmetic mismatch index="<<i<<" op="<<cases[i].operation<<" a="<<std::hex
        <<cases[i].a<<" b="<<cases[i].b<<" actual="<<actual[i].value<<" expected="<<expected.value
        <<std::dec<<" error="<<actual[i].error<<" expected_error="<<expected.error<<'\n';
      throw std::runtime_error("NPU binary64 arithmetic mismatch");
    }
  }
  Passed("npu_binary64_arithmetic_bitwise");
}
void Scores(npu::Runtime const& r) {
  for (auto const& c:t::Cases()) {
    auto input=Upload(r,c.inputs); auto output=r.AllocateElements<p::Score>(c.inputs.size()); output.Zero();
    XgbNpuPreciseScore(r.Stream(),input.Data(),output.Data(),c.inputs.size()); r.Synchronize();
    std::vector<p::Score> actual(c.inputs.size()); output.Download(actual.data(),actual.size()*sizeof(actual[0]));
    for (std::size_t i=0;i<c.inputs.size();++i) {
      if (c.reject) qt::Require(actual[i].error!=0,"NPU accepted unsupported numerical domain");
      else t::Compare(actual[i],t::Expected(c.inputs[i]));
    }
    Passed("npu_"+c.name);
  }
}
void Compare(npu::Ops& ops,Builder::Histogram const& histogram,histogram_test::Case const& c,
             std::vector<int64_t> const& rows,p::Parameters parameters) {
  npu::NumericalSplitStatistics statistics{ops}; npu::PreciseSplitScorer scorer{ops};
  for (std::size_t f=0;f<c.bin_counts.size();++f) {
    auto candidates=statistics.Build(histogram,f); if (!candidates) continue;
    auto before=ops.ControlDownloadBytes(); auto launches=scorer.Launches();
    auto actual=scorer.Evaluate(*candidates,parameters);
    qt::Require(ops.ControlDownloadBytes()-before==16 && scorer.Launches()-launches==2,
                "scoring should download only two int64 status summaries");
    qt::Require(actual.Candidates()==c.bin_counts[f],"scored candidate count");
    auto g=st::Oracle(c,rows,f,c.gradient),h=st::Oracle(c,rows,f,c.hessian);
    for (bool missing_left:{false,true}) {
      auto const& buffer=missing_left?npu::PreciseScorerTestAccess::Left(actual):npu::PreciseScorerTestAccess::Right(actual);
      std::vector<p::Score> values(c.bin_counts[f]); buffer.Download(values.data(),values.size()*sizeof(values[0]));
      for (std::size_t i=0;i<values.size();++i) {
        p::Input input{missing_left?g.left_missing_left[i]:g.left_missing_right[i],
                       missing_left?h.left_missing_left[i]:h.left_missing_right[i],
                       missing_left?g.right_missing_left[i]:g.right_missing_right[i],
                       missing_left?h.right_missing_left[i]:h.right_missing_right[i],parameters};
        t::Compare(values[i],t::Expected(input));
      }
    }
  }
}
void Integration(npu::Runtime const& r,npu::Ops& ops) {
  auto cases=st::Cases();
  cases.push_back({"columns_grid_stride",{1,-3,7,5},{1,2,3,4},{2049},{{0,2048,1024,-1}},{0,1,2,3}});
  for (auto const& c:cases) {
    std::vector<npu::Tensor> cols; for (auto const& col:c.columns) cols.push_back(UploadTensor(r,col));
    Builder b{ops,std::move(cols),UploadTensor(r,c.gradient),UploadTensor(r,c.hessian),c.bin_counts};
    auto rows=c.selected.empty()?nullptr:std::make_unique<npu::Tensor>(UploadTensor(r,c.selected));
    auto hist=b.Build(rows.get()); Compare(ops,hist,c,c.selected,t::Params(0,1,0.2f,0.7f,58,58));
    auto [left,right]=histogram_test::Split(c.selected);
    auto ids=left.empty()?nullptr:std::make_unique<npu::Tensor>(UploadTensor(r,left));
    auto child=b.Build(ids.get()); auto sibling=b.SubtractSibling(hist,child);
    Compare(ops,sibling,c,right,t::Params(0,1,0.2f,0.7f,58,58));
    Passed("npu_integration_"+c.name);
  }
  auto c=st::Cases().front(); std::vector<float> gf{-2,1,3,0,-1,2,4,0},hf{1,1,2,0,1,3,2,0};
  npu::Tensor g{r,8,ACL_FLOAT},h{r,8,ACL_FLOAT}; g.Storage().Upload(gf.data(),32); h.Storage().Upload(hf.data(),32);
  npu::DynamicQuantizer q{ops}; auto qg=q.Convert(g,false),qh=q.Convert(h,true);
  qt::Require(qg.certificate.scale.exponent==58 && qh.certificate.scale.exponent==58,"dynamic integration scale");
  std::vector<npu::Tensor> columns; columns.push_back(UploadTensor(r,c.columns[0]));
  Builder b{ops,std::move(columns),std::move(qg.channel.values),std::move(qh.channel.values),c.bin_counts};
  auto ids=UploadTensor(r,c.selected); auto histogram=b.Build(&ids);
  Compare(ops,histogram,c,c.selected,t::Params(1,1,0.2f,0.7f,58,58));
  Passed("npu_dynamic_quantizer_histogram_statistics_scoring");
  npu::NumericalSplitStatistics statistics{ops}; auto candidate=statistics.Build(histogram,0);
  npu::PreciseSplitScorer scorer{ops}; auto bytes=r.LiveBytes(),launches=scorer.Launches();
  for (auto param:{t::Params(0,-1),t::Params(0,1,0,0,1023,0),t::Params(0,std::numeric_limits<float>::infinity())}) {
    bool rejected=false;
    try { scorer.Evaluate(*candidate,param); } catch (std::invalid_argument const&) { rejected=true; }
    qt::Require(rejected && r.LiveBytes()==bytes && scorer.Launches()==launches,"parameter guard executed or leaked");
  }
  Passed("npu_scorer_parameter_guards");
  bool rejected=false;
  try { scorer.Evaluate(*candidate,t::Params(0,0,0,0,-100,100)); } catch (std::overflow_error const&) { rejected=true; }
  qt::Require(rejected && r.LiveBytes()==bytes,"scorer domain failure did not clean resources");
  Passed("npu_scorer_device_domain_rejection");
  npu::Runtime foreign{npu::MakeAclApi(),0}; npu::Ops other_ops{foreign}; npu::PreciseSplitScorer other{other_ops};
  rejected=false;
  try { other.Evaluate(*candidate,t::Params()); } catch (std::invalid_argument const&) { rejected=true; }
  qt::Require(rejected && !other.Launches() && foreign.LiveBytes()==0,"foreign runtime guard");
  Passed("npu_scorer_owner_guard");
}
void TailGuards(npu::Runtime const& r) {
  constexpr p::Score canary{0x12345678U,0x23456789U,0x34567890U,0x45678901U};
  constexpr int64_t error_canary=INT64_C(0x1122334455667788);
  auto metadata=Upload(r,std::vector<p::Parameters>{t::Params(0,1,0.2f,0.7f)});
  for (uint32_t count:{1,127,128,129,2049}) {
    auto lg=Upload(r,std::vector<int64_t>(count,-3)),lh=Upload(r,std::vector<int64_t>(count,7));
    auto rg=Upload(r,std::vector<int64_t>(count,5)),rh=Upload(r,std::vector<int64_t>(count,11));
    std::vector<p::Score> actual(count+2,canary); auto out=Upload(r,actual);
    std::vector<int64_t> errors(count+2,error_canary); auto status=Upload(r,errors);
    XgbNpuScoreColumns(r.Stream(),lg.Data(),lh.Data(),rg.Data(),rh.Data(),metadata.Data(),
      static_cast<char*>(out.Data())+sizeof(p::Score),static_cast<char*>(status.Data())+sizeof(int64_t),count);
    r.Synchronize(); out.Download(actual.data(),actual.size()*sizeof(actual[0])); status.Download(errors.data(),errors.size()*8);
    qt::Require(std::memcmp(&actual.front(),&canary,sizeof(canary))==0 &&
                std::memcmp(&actual.back(),&canary,sizeof(canary))==0 &&
                errors.front()==error_canary && errors.back()==error_canary,"kernel tail overwrote canary");
    auto expected=t::Expected(p::Input{-3,7,5,11,t::Params(0,1,0.2f,0.7f)});
    for (uint32_t i=0;i<count;++i) {t::Compare(actual[i+1],expected);qt::Require(errors[i+1]==0,"unwritten status");}
  }
  Passed("npu_columns_grid_tail_canaries");
}
void Tests() {
  int errors=0; auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error=[&](std::string const& message){++errors;std::cerr<<"cleanup failure: "<<message<<'\n';};
  {
    npu::Runtime r{api,0}; npu::Ops ops{r}; Arithmetic(r); Scores(r); Integration(r,ops); TailGuards(r);
    qt::Require(!r.LiveBytes(),"precise scoring device memory leaked");
  }
  qt::Require(!errors,"precise scoring resource cleanup failed"); Passed("npu_precise_scoring_resources_released");
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" && std::string(argv[2])=="0" &&
              std::string(argv[3])=="--confirm-device-idle";
  if (!host && !device) { std::cerr<<"Usage: precise_scoring_test --host-api-only | --device-test 0 --confirm-device-idle\n";return 2; }
  try {
    npu::MakeAclApi()->Validate();
    if (host) {
      XgbNpuPreciseScore(nullptr,nullptr,nullptr,0);
      XgbNpuTestArithmetic(nullptr,nullptr,nullptr,0);
      XgbNpuScoreColumns(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0);
      std::cout<<"PRECISE_SCORING_API_HOST_PASS zero_count_wrappers=passed\n";return 0;
    }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try { Tests(); } catch (...) { aclFinalize(); throw; }
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    std::cout<<"PRECISE_SCORING_NPU_PASS checks="<<checks<<" selection=not_implemented training_backend=false\n";return 0;
  } catch (std::exception const& e) {std::cerr<<"PRECISE_SCORING_FAIL: "<<e.what()<<'\n';return 1;}
}
