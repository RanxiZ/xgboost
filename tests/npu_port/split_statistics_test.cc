// Copyright 2026, XGBoost Contributors
// Device integer candidate statistics; gains and candidate selection are absent.
#include "../../src/common/npu/split_statistics.h"
#include "../../src/common/npu/dynamic_quantizer.h"
#include "split_statistics_cases.h"
#include <dlfcn.h>
#include <fstream>

namespace npu=xgboost::common::npu;
namespace st=split_statistics_test;
namespace ht=histogram_test;
namespace qt=quantizer_test;
using Builder=npu::IntegerHistogramBuilder;
using Statistics=npu::NumericalSplitStatistics;
namespace xgboost::common::npu {
class SplitStatisticsTestAccess {
 public:
  static auto const& Gradient(Statistics::Result const& r) { return r.gradient_; }
  static auto const& Hessian(Statistics::Result const& r) { return r.hessian_; }
};
}
namespace {
std::size_t checks=0;
void Passed(std::string const& name) { ++checks; qt::Passed(name); }
npu::Tensor Upload(npu::Runtime const& r,std::vector<int64_t> const& values) {
  npu::Tensor out{r,static_cast<int64_t>(values.size()),ACL_INT64};
  out.Storage().Upload(values.data(),values.size()*sizeof(int64_t)); return out;
}
std::unique_ptr<Builder> Make(npu::Ops& ops,ht::Case const& c) {
  auto const& r=ops.GetRuntime(); std::vector<npu::Tensor> columns;
  for (auto const& col:c.columns) columns.push_back(Upload(r,col));
  return std::make_unique<Builder>(ops,std::move(columns),Upload(r,c.gradient),Upload(r,c.hessian),c.bin_counts);
}
Builder::Histogram Build(Builder& builder,npu::Runtime const& r,std::vector<int64_t> const& rows) {
  if (rows.empty()) return builder.Build(nullptr);
  auto ids=Upload(r,rows); return builder.Build(&ids);
}
std::vector<int64_t> Download(npu::Tensor const& t) {
  std::vector<int64_t> out(t.Count()); t.Storage().Download(out.data(),out.size()*sizeof(int64_t)); return out;
}
template<typename Channel> void CompareChannel(Channel const& actual,st::Channel const& expected) {
  qt::Require(Download(actual.missing)==std::vector<int64_t>{expected.missing} &&
              Download(actual.left_missing_right)==expected.left_missing_right &&
              Download(actual.right_missing_right)==expected.right_missing_right &&
              Download(actual.left_missing_left)==expected.left_missing_left &&
              Download(actual.right_missing_left)==expected.right_missing_left,
              "device candidate integers differ from independent row oracle");
}
void Compare(npu::Ops& ops,Builder::Histogram const& histogram,ht::Case const& c,
             std::vector<int64_t> const& rows) {
  Statistics statistics{ops};
  for (std::size_t f=0;f<c.bin_counts.size();++f) {
    auto bytes=ops.ControlDownloadBytes(),launches=statistics.Launches(),operators=ops.Launches();
    auto result=statistics.Build(histogram,f);
    qt::Require(ops.ControlDownloadBytes()==bytes,"candidate statistics downloaded device data");
    qt::Require(ops.Launches()==operators,"candidate statistics launched an ACLNN operator");
    if (!c.bin_counts[f]) {
      qt::Require(!result && statistics.Launches()==launches,"zero-bin feature produced candidates or launches"); continue;
    }
    qt::Require(result && result->Candidates()==c.bin_counts[f],"wrong candidate count");
    // One custom SIMT statistics kernel runs per channel; downloads below
    // are test inspection only, not the production statistics path.
    qt::Require(statistics.Launches()-launches==2,"unexpected candidate kernel launch count");
    using Access=npu::SplitStatisticsTestAccess;
    CompareChannel(Access::Gradient(*result),st::Oracle(c,rows,f,c.gradient));
    CompareChannel(Access::Hessian(*result),st::Oracle(c,rows,f,c.hessian));
  }
}
void KernelTails(npu::Runtime const& r,npu::Ops& ops) {
  constexpr int64_t big=(INT64_C(1)<<62)-1,canary=INT64_C(0x123456789abcdef);
  for (int64_t bins:{1,2,31,32,33,127,128,129,255,256,257,1025,2049}) {
    ht::Case c{"kernel_tail",{big,big,1,-big,-big,-1,0,0},{big,big,1,0,0,0,0,0},
               {bins},{{0,-1,bins-1,0,bins-1,-1,0,-1}},{0,1,2,3,4,5,6,7}};
    auto builder=Make(ops,c); auto histogram=Build(*builder,r,c.selected);
    Compare(ops,histogram,c,c.selected);
    // Surround every output with canaries, including the scalar. Compare all
    // elements to classification of original rows, never a second prefix scan.
    auto hist=ht::Oracle(c,c.selected);
    for (bool hessian:{false,true}) {
      auto input=Upload(r,hessian?hist.hessian[0]:hist.gradient[0]);
      auto parent=Upload(r,{hessian?hist.root_h:hist.root_g});
      auto missing=Upload(r,std::vector<int64_t>(3,canary));
      auto lr=Upload(r,std::vector<int64_t>(bins+2,canary));
      auto rr=Upload(r,std::vector<int64_t>(bins+2,canary));
      auto ll=Upload(r,std::vector<int64_t>(bins+2,canary));
      auto rl=Upload(r,std::vector<int64_t>(bins+2,canary));
      auto offset=[](npu::Tensor const& t) { return static_cast<int64_t*>(t.Storage().Data())+1; };
      XgbNpuSplitStatistics(r.Stream(),input.Storage().Data(),parent.Storage().Data(),
          offset(missing),offset(lr),offset(rr),offset(ll),offset(rl),static_cast<uint32_t>(bins));
      r.Synchronize();
      auto expected=st::Oracle(c,c.selected,0,hessian?c.hessian:c.gradient);
      auto check=[&](npu::Tensor const& actual,std::vector<int64_t> const& values) {
        auto got=Download(actual);
        qt::Require(got.front()==canary && got.back()==canary,"statistics kernel overwrote canary");
        qt::Require(std::vector<int64_t>(got.begin()+1,got.end()-1)==values,
                    "statistics kernel differs from exact independent integer oracle");
      };
      check(missing,{expected.missing}); check(lr,expected.left_missing_right);
      check(rr,expected.right_missing_right); check(ll,expected.left_missing_left);
      check(rl,expected.right_missing_left);
    }
  }
  Passed("npu_statistics_kernel_tails_cancellation_canaries");
}
void Cases(npu::Runtime const& r,npu::Ops& ops) {
  KernelTails(r,ops);
  for (auto const& c:st::Cases()) {
    auto baseline=r.LiveBytes();
    {
      auto builder=Make(ops,c); auto parent=Build(*builder,r,c.selected);
      Compare(ops,parent,c,c.selected); Passed("npu_direct_"+c.name);
      auto [left,right]=ht::Split(c.selected); auto child=Build(*builder,r,left);
      auto sibling=builder->SubtractSibling(parent,child);
      Compare(ops,sibling,c,right); Passed("npu_subtracted_"+c.name);
    }
    qt::Require(r.LiveBytes()==baseline,"candidate case leaked device storage");
  }
}
template<typename F> void Reject(std::string const& name,npu::Ops& ops,F fn) {
  auto launches=ops.Launches(); bool rejected=false;
  try { fn(); } catch (std::invalid_argument const&) { rejected=true; }
  qt::Require(rejected && ops.Launches()==launches,"split statistics guard launched or accepted input"); Passed(name);
}
void Guards(npu::Runtime const& r,npu::Ops& ops) {
  auto c=st::Cases()[1]; auto builder=Make(ops,c); auto histogram=Build(*builder,r,c.selected);
  Statistics statistics{ops};
  Reject("npu_reject_feature_range",ops,[&]{ statistics.Build(histogram,c.bin_counts.size()); });
  Reject("npu_reject_extreme_feature",ops,[&]{ statistics.Build(histogram,SIZE_MAX); });
  npu::Tensor wrong{r,1,ACL_FLOAT};
  Reject("npu_reject_scan_dtype",ops,[&]{ ops.ExclusiveInt64Scan(wrong); });
  npu::Runtime foreign{npu::MakeAclApi(),0}; npu::Ops foreign_ops{foreign}; Statistics foreign_statistics{foreign_ops};
  Reject("npu_reject_histogram_runtime",foreign_ops,[&]{ foreign_statistics.Build(histogram,0); });
  auto foreign_tensor=Upload(foreign,{1});
  Reject("npu_reject_scan_runtime",ops,[&]{ ops.ExclusiveInt64Scan(foreign_tensor); });
  qt::Require(statistics.Launches()==0 && foreign_statistics.Launches()==0,
              "split statistics guard launched a custom kernel");
}
void Integration(npu::Runtime const& r,npu::Ops& ops) {
  auto c=st::Cases().front(); std::vector<float> gf{-2,1,3,0,-1,2,4,0},hf{1,1,2,0,1,3,2,0};
  npu::Tensor g{r,8,ACL_FLOAT},h{r,8,ACL_FLOAT}; g.Storage().Upload(gf.data(),32); h.Storage().Upload(hf.data(),32);
  npu::DynamicQuantizer quantizer{ops}; auto qg=quantizer.Convert(g,false),qh=quantizer.Convert(h,true);
  qt::Require(qg.certificate.scale.exponent==58 && qh.certificate.scale.exponent==58,"integration scale mismatch");
  std::vector<npu::Tensor> columns; columns.push_back(Upload(r,c.columns[0]));
  Builder builder{ops,std::move(columns),std::move(qg.channel.values),std::move(qh.channel.values),c.bin_counts};
  auto histogram=Build(builder,r,c.selected); Compare(ops,histogram,c,c.selected);
  Passed("npu_dynamic_quantizer_histogram_split_statistics");
}
void Tests() {
  int errors=0; auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error=[&](std::string const& message){ ++errors; std::cerr<<"cleanup failure: "<<message<<'\n'; };
  {
    npu::Runtime r{api,0}; npu::Ops ops{r}; Cases(r,ops); Guards(r,ops); Integration(r,ops);
    qt::Require(r.LiveBytes()==0,"device memory leaked");
  }
  qt::Require(errors==0,"resource cleanup failed"); Passed("npu_split_statistics_resources_released");
}
void Provenance() {
  for (auto symbol:{"aclInit","aclnnCumsum","aclnnAdd","aclnnReduceSum","XgbNpuSplitStatistics"}) {
    Dl_info info{}; qt::Require(dladdr(dlsym(RTLD_DEFAULT,symbol),&info)&&info.dli_fname,"missing library provenance");
    std::cout<<"library "<<symbol<<' '<<info.dli_fname<<'\n';
  }
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" && std::string(argv[2])=="0" &&
              std::string(argv[3])=="--confirm-device-idle";
  if (!host && !device) { std::cerr<<"Usage: split_statistics_test --host-api-only | --device-test 0 --confirm-device-idle\n"; return 2; }
  try {
    Provenance(); npu::MakeAclApi()->Validate();
    if (host) {
      XgbNpuSplitStatistics(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0);
      std::cout<<"SPLIT_STATISTICS_API_HOST_PASS device_execution=not_run zero_count_wrapper=passed\n"; return 0;
    }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try { Tests(); }
    catch (...) { auto status=aclFinalize(); if (status) std::cerr<<"aclFinalize status="<<status<<'\n'; throw; }
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    std::cout<<"SPLIT_STATISTICS_NPU_PASS checks="<<checks<<" split_scoring=not_implemented training_backend=false\n"; return 0;
  } catch (std::exception const& e) { std::cerr<<"SPLIT_STATISTICS_FAIL: "<<e.what()<<'\n'; return 1; }
}
