// Copyright 2026, XGBoost Contributors
// Independent histogram build/subtraction tests, not XGBoost tree training.
#include "../../src/common/npu/integer_histogram.h"
#include "../../src/common/npu/dynamic_quantizer.h"
#include "histogram_cases.h"
#include <dlfcn.h>
#include <fstream>

namespace npu=xgboost::common::npu;
namespace ht=histogram_test;
namespace qt=quantizer_test;
using Builder=npu::IntegerHistogramBuilder;
// Test-only inspection; production Histogram publishes no raw device handles.
namespace xgboost::common::npu {
class IntegerHistogramTestAccess {
 public:
  static auto const& Features(IntegerHistogramBuilder::Histogram const& h) { return h.features_; }
  static auto const& RootGradient(IntegerHistogramBuilder::Histogram const& h) { return h.root_gradient_; }
  static auto const& RootHessian(IntegerHistogramBuilder::Histogram const& h) { return h.root_hessian_; }
  static auto const& Packed(IntegerHistogramBuilder::Histogram const& h) { return h.packed_; }
};
}
namespace {
std::size_t checks=0;
void Passed(std::string const& name) { ++checks; qt::Passed(name); }
npu::Tensor Upload(npu::Runtime const& runtime,std::vector<int64_t> const& values) {
  npu::Tensor out{runtime,static_cast<int64_t>(values.size()),ACL_INT64};
  out.Storage().Upload(values.data(),values.size()*sizeof(int64_t)); return out;
}
std::unique_ptr<Builder> Make(npu::Ops& ops,ht::Case const& c) {
  auto const& r=ops.GetRuntime(); std::vector<npu::Tensor> columns;
  for (auto const& col:c.columns) columns.push_back(Upload(r,col));
  return std::make_unique<Builder>(ops,std::move(columns),Upload(r,c.gradient),Upload(r,c.hessian),c.bin_counts);
}
Builder::Histogram Build(Builder& b,npu::Runtime const& r,std::vector<int64_t> const& rows) {
  if (rows.empty()) return b.Build(nullptr);
  auto ids=Upload(r,rows); return b.Build(&ids);
}
std::vector<int64_t> Download(npu::Tensor const& t) {
  std::vector<int64_t> out(t.Count()); t.Storage().Download(out.data(),out.size()*sizeof(int64_t)); return out;
}
void Compare(Builder::Histogram const& actual,ht::Expected const& expected,std::size_t rows) {
  using Access=npu::IntegerHistogramTestAccess;
  qt::Require(actual.Rows()==static_cast<int64_t>(rows) && Download(Access::RootGradient(actual))[0]==expected.root_g &&
              Download(Access::RootHessian(actual))[0]==expected.root_h,"device root stats mismatch");
  qt::Require(actual.FeatureCount()==expected.gradient.size(),"feature count mismatch");
  int64_t offset = 0;
  auto base = static_cast<int64_t*>(Access::Packed(actual).Storage().Data());
  for (std::size_t f=0;f<actual.FeatureCount();++f) {
    auto const& feature=Access::Features(actual)[f];
    auto count = std::max<int64_t>(1, feature.bin_count);
    qt::Require(feature.gradient.Storage().Data() == base + offset &&
                feature.hessian.Storage().Data() == base + offset + count,
                "histogram features are not contiguous bounded views");
    offset += 2 * count;
    qt::Require(static_cast<int64_t>(expected.gradient[f].size())==std::max<int64_t>(1,actual.FeatureBinCount(f)),"bin count mismatch");
    qt::Require(Download(feature.gradient)==expected.gradient[f] &&
                Download(feature.hessian)==expected.hessian[f],"integer histogram mismatch");
  }
}
template<typename F> void Reject(std::string const& name,F fn) {
  bool rejected=false;
  try { fn(); } catch (std::invalid_argument const&) { rejected=true; }
    catch (std::overflow_error const&) { rejected=true; }
  qt::Require(rejected,"unsupported histogram input was accepted"); Passed(name);
}
void Cases(npu::Runtime const& r,npu::Ops& ops) {
  for (auto const& c:ht::Cases()) {
    auto baseline=r.LiveBytes();
    {
      auto b=Make(ops,c); auto before=ops.ControlDownloadBytes(); auto launch=ops.Launches();
      auto fused_before=b->BuildLaunches();
      auto parent=Build(*b,r,c.selected);
      qt::Require(b->BuildLaunches()==fused_before+(c.selected.empty()?0:1),
                  "fused histogram launch count mismatch");
      qt::Require(ops.ControlDownloadBytes()-before==(c.selected.empty()?0:16),"build downloaded more than row guards");
      Compare(parent,ht::Oracle(c,c.selected),c.selected.size());
      std::cout<<"{\"test\":\"npu_direct_"<<c.name<<"\",\"status\":\"passed\",\"rows\":"<<c.selected.size()
        <<",\"control_download_bytes\":"<<ops.ControlDownloadBytes()-before
        <<",\"operator_launches\":"<<ops.Launches()-launch<<"}\n"; ++checks;
      auto [left,right]=ht::Split(c.selected);
      auto child=Build(*b,r,left); auto direct=Build(*b,r,right);
      before=ops.ControlDownloadBytes();
      auto subtraction_launches = ops.Launches();
      auto sibling=b->SubtractSibling(parent,child);
      qt::Require(ops.Launches() - subtraction_launches == 7,
                  "packed subtraction launch count depends on feature count");
      qt::Require(ops.ControlDownloadBytes()-before==8,"subtraction downloaded more than subset guard");
      Compare(sibling,ht::Oracle(c,right),right.size()); Compare(direct,ht::Oracle(c,right),right.size());
      auto empty=b->SubtractSibling(parent,parent); Compare(empty,ht::Oracle(c,{}),0);
      auto again=b->SubtractSibling(parent,empty); Compare(again,ht::Oracle(c,c.selected),c.selected.size());
      // Subtracted memberships remain usable as a parent for another subtraction.
      auto sibling_empty=b->SubtractSibling(sibling,direct); Compare(sibling_empty,ht::Oracle(c,{}),0);
      Passed("npu_subtract_"+c.name);
    }
    qt::Require(r.LiveBytes()==baseline,"histogram case leaked device storage");
  }
}
void SharedBins(npu::Runtime const& r, npu::Ops& ops) {
  auto baseline = r.LiveBytes();
  auto c = ht::Cases()[1];
  std::unique_ptr<Builder::Histogram> survivor;
  {
    std::vector<npu::Tensor> columns;
    for (auto const& col : c.columns) columns.push_back(Upload(r, col));
    auto bins = Builder::BindBins(ops, std::move(columns), c.bin_counts);
    Builder first{ops, bins, Upload(r, c.gradient), Upload(r, c.hessian)};
    auto channel_launches = ops.Launches();
    auto bytes = r.LiveBytes();
    Builder second{ops, bins, Upload(r, c.gradient), Upload(r, c.hessian)};
    // Only two per-tree channels allocate; no copied columns or revalidation.
    qt::Require(r.LiveBytes() - bytes == 2 * c.gradient.size() * sizeof(int64_t),
                "shared bins copied device columns");
    auto second_launches = ops.Launches() - channel_launches;
    Builder third{ops, bins, Upload(r, c.gradient), Upload(r, c.hessian)};
    qt::Require(ops.Launches() - channel_launches == 2 * second_launches,
                "shared bins construction changed channel validation");
    survivor = std::make_unique<Builder::Histogram>(Build(first, r, c.selected));
    auto other = Build(second, r, c.selected);
    bool rejected = false;
    try { first.SubtractSibling(*survivor, other); }
    catch (std::invalid_argument const&) { rejected = true; }
    qt::Require(rejected, "shared bins allowed cross-tree gradient/scale subtraction");
    rejected = false;
    try { Builder invalid{ops, bins, Upload(r, {1}), Upload(r, {1})}; }
    catch (std::invalid_argument const&) { rejected = true; }
    qt::Require(rejected, "shared bins accepted wrong row count");
    rejected = false;
    try { Builder invalid{ops, std::shared_ptr<Builder::ImmutableBins const>{},
                          Upload(r, c.gradient), Upload(r, c.hessian)}; }
    catch (std::invalid_argument const&) { rejected = true; }
    qt::Require(rejected, "null immutable bins accepted");
    {
      npu::Runtime foreign{npu::MakeAclApi(), 0}; npu::Ops foreign_ops{foreign};
      rejected = false;
      try { Builder invalid{foreign_ops, bins, Upload(foreign, c.gradient),
                            Upload(foreign, c.hessian)}; }
      catch (std::invalid_argument const&) { rejected = true; }
      qt::Require(rejected, "shared bins accepted foreign runtime");
    }
    bins.reset();
  }
  Compare(*survivor, ht::Oracle(c, c.selected), c.selected.size());
  survivor.reset();
  qt::Require(r.LiveBytes() == baseline, "shared bins or tensor views leaked storage");
  Passed("npu_shared_immutable_bins_and_packed_lifetime");
}
void Guards(npu::Runtime const& r,npu::Ops& ops) {
  auto c=ht::Cases()[1]; auto b=Make(ops,c);
  auto empty_launches=b->BuildLaunches();
  auto empty_node=b->Build(nullptr);
  qt::Require(b->BuildLaunches()==empty_launches,"empty histogram launched fused kernel");
  (void)empty_node;
  for (auto const& item:std::vector<std::pair<std::string,std::vector<int64_t>>>{
       {"negative_row",{-1}},{"large_row",{6}},{"extreme_row",{INT64_MAX}},
       {"duplicate_row",{0,0}},{"too_many_rows",{0,1,2,3,4,5,0}}}) {
    Reject("npu_reject_"+item.first,[&]{ Build(*b,r,item.second); });
  }
  {
    npu::Tensor wrong{r,1,ACL_FLOAT}; auto launches=ops.Launches();
    Reject("npu_reject_row_dtype",[&]{ b->Build(&wrong); });
    qt::Require(ops.Launches()==launches,"dtype guard launched kernels");
  }
  auto p=Build(*b,r,{0}),ch=Build(*b,r,{1}),empty=b->Build(nullptr);
  Reject("npu_reject_nonsubset",[&]{ b->SubtractSibling(p,ch); });
  Reject("npu_reject_larger_child",[&]{ b->SubtractSibling(empty,ch); });
  auto other=Make(ops,c); auto same_rows=Build(*other,r,{0});
  Reject("npu_reject_source_mismatch",[&]{ b->SubtractSibling(p,same_rows); });
  {
    npu::Runtime foreign{npu::MakeAclApi(),0}; auto row=Upload(foreign,{0}); auto launches=ops.Launches();
    Reject("npu_reject_foreign_runtime",[&]{ b->Build(&row); });
    qt::Require(ops.Launches()==launches,"owner guard launched kernels");
  }
  for (auto const& name:std::vector<std::string>{"negative_bin","large_bin","zero_feature_bin","shape",
       "channel_min","channel_max","positive_overflow","negative_overflow","hidden_sign_overflow"}) {
    auto bad=c;
    if (name=="negative_bin") bad.columns[0][0]=-2;
    if (name=="large_bin") bad.columns[0][0]=bad.bin_counts[0];
    if (name=="zero_feature_bin") bad.columns[2][0]=0;
    if (name=="shape") bad.hessian.pop_back();
    if (name=="channel_min") bad.gradient[0]=INT64_MIN;
    if (name=="channel_max") bad.hessian[0]=INT64_MAX;
    auto big=(INT64_C(1)<<62)-1;
    if (name=="positive_overflow") bad.gradient={big,big,big,0,0,0};
    if (name=="negative_overflow") bad.hessian={-big,-big,-big,0,0,0};
    if (name=="hidden_sign_overflow") bad.gradient={big,big,big,-big,-big,-big};
    Reject("npu_reject_"+name,[&]{ Make(ops,bad); });
  }
}
void DirectKernelShapes(npu::Runtime const& r) {
  constexpr int64_t kCanary=INT64_C(0x13579bdf2468ace);
  std::mt19937 random{20261019};
  // Cover the 16-block ownership boundary, multiple feature strides, row
  // tails, and bins larger than the 128-thread block. Keep canaries around
  // each channel and repeat every launch after re-poisoning its outputs.
  for (auto shape : std::vector<std::pair<std::size_t,std::size_t>>{
           {1,129},{16,129},{17,129},{31,257},{32,513},{33,1025},{158,2049},{17,4}}) {
    auto features=shape.first, rows=shape.second;
    std::vector<int64_t> row_ids(rows), gradient(rows), hessian(rows);
    std::iota(row_ids.begin(),row_ids.end(),0); std::shuffle(row_ids.begin(),row_ids.end(),random);
    for (std::size_t i=0;i<rows;++i) {
      gradient[i]=static_cast<int64_t>((i*37)%1000003)-500001;
      hessian[i]=static_cast<int64_t>((i*19)%1000001)-500000;
    }
    npu::Tensor d_rows{r,static_cast<int64_t>(rows),ACL_INT64};
    npu::Tensor d_gradient{r,static_cast<int64_t>(rows),ACL_INT64};
    npu::Tensor d_hessian{r,static_cast<int64_t>(rows),ACL_INT64};
    d_rows.Storage().Upload(row_ids.data(),rows*sizeof(int64_t));
    d_gradient.Storage().Upload(gradient.data(),rows*sizeof(int64_t));
    d_hessian.Storage().Upload(hessian.data(),rows*sizeof(int64_t));
    std::vector<npu::Tensor> columns, output_gradient, output_hessian;
    std::vector<npu::histogram::FeatureDescriptor> metadata;
    columns.reserve(features); output_gradient.reserve(features); output_hessian.reserve(features);
    metadata.reserve(features);
    std::vector<std::vector<int64_t>> expected_gradient, expected_hessian;
    expected_gradient.reserve(features); expected_hessian.reserve(features);
    for (std::size_t f=0;f<features;++f) {
      auto bins=(f%11==0) ? 0 : ((f%7==0) ? 256 : static_cast<int64_t>((f%13)+1));
      std::vector<int64_t> values(rows), expected_g(std::max<int64_t>(1,bins),0),
          expected_h(std::max<int64_t>(1,bins),0);
      for (std::size_t row=0;row<rows;++row) {
        values[row]=bins==0 ? -1 : static_cast<int64_t>((row*7+f*3)%(bins+1))-1;
      }
      for (auto row:row_ids) {
        auto bin=values[row]; if (bin!=-1) { expected_g[bin]+=gradient[row]; expected_h[bin]+=hessian[row]; }
      }
      columns.emplace_back(r,static_cast<int64_t>(rows),ACL_INT64);
      columns.back().Storage().Upload(values.data(),rows*sizeof(int64_t));
      output_gradient.emplace_back(r,std::max<int64_t>(1,bins)+2,ACL_INT64);
      output_hessian.emplace_back(r,std::max<int64_t>(1,bins)+2,ACL_INT64);
      std::vector<int64_t> canary(std::max<int64_t>(1,bins)+2,kCanary);
      output_gradient.back().Storage().Upload(canary.data(),canary.size()*sizeof(int64_t));
      output_hessian.back().Storage().Upload(canary.data(),canary.size()*sizeof(int64_t));
      metadata.push_back({static_cast<uint64_t>(reinterpret_cast<uintptr_t>(columns.back().Storage().Data())),
                          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_gradient.back().Storage().Data()))+sizeof(int64_t),
                          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output_hessian.back().Storage().Data()))+sizeof(int64_t),
                          static_cast<uint32_t>(bins),0});
      expected_gradient.push_back(std::move(expected_g)); expected_hessian.push_back(std::move(expected_h));
    }
    auto d_metadata=r.Allocate(metadata.size()*sizeof(metadata[0]));
    d_metadata.Upload(metadata.data(),metadata.size()*sizeof(metadata[0]));
    auto Run=[&] {
      XgbNpuIntegerHistogramBuild(r.Stream(),d_rows.Storage().Data(),d_gradient.Storage().Data(),
                                  d_hessian.Storage().Data(),d_metadata.Data(),rows,features);
      r.Synchronize();
    };
    Run();
    for (std::size_t f=0;f<features;++f) {
      auto bins=expected_gradient[f].size(); std::vector<int64_t> got(bins+2), got_h(bins+2);
      output_gradient[f].Storage().Download(got.data(),got.size()*sizeof(int64_t));
      output_hessian[f].Storage().Download(got_h.data(),got_h.size()*sizeof(int64_t));
      qt::Require(got.front()==kCanary && got.back()==kCanary && got_h.front()==kCanary && got_h.back()==kCanary,
                  "fused histogram canary overwritten");
      for (std::size_t bin=0;bin<bins;++bin) {
        if (got[bin+1]!=expected_gradient[f][bin] || got_h[bin+1]!=expected_hessian[f][bin]) {
          std::cerr << "HIST_MISMATCH features=" << features << " rows=" << rows
                    << " feature=" << f << " bin=" << bin << " actual_g=" << got[bin+1]
                    << " expected_g=" << expected_gradient[f][bin] << " actual_h=" << got_h[bin+1]
                    << " expected_h=" << expected_hessian[f][bin] << '\n';
        }
      }
      qt::Require(std::equal(expected_gradient[f].begin(),expected_gradient[f].end(),got.begin()+1) &&
                  std::equal(expected_hessian[f].begin(),expected_hessian[f].end(),got_h.begin()+1),
                  "direct fused histogram mismatch");
    }
    for (std::size_t f=0;f<features;++f) {
      std::vector<int64_t> canary(expected_gradient[f].size()+2,kCanary);
      output_gradient[f].Storage().Upload(canary.data(),canary.size()*sizeof(int64_t));
      output_hessian[f].Storage().Upload(canary.data(),canary.size()*sizeof(int64_t));
    }
    Run();
    for (std::size_t f=0;f<features;++f) {
      auto bins=expected_gradient[f].size(); std::vector<int64_t> got(bins+2),got_h(bins+2);
      output_gradient[f].Storage().Download(got.data(),got.size()*sizeof(int64_t));
      output_hessian[f].Storage().Download(got_h.data(),got_h.size()*sizeof(int64_t));
      qt::Require(got.front()==kCanary && got.back()==kCanary && got_h.front()==kCanary && got_h.back()==kCanary,
                  "repeated fused histogram canary overwritten");
      qt::Require(std::equal(expected_gradient[f].begin(),expected_gradient[f].end(),got.begin()+1) &&
                  std::equal(expected_hessian[f].begin(),expected_hessian[f].end(),got_h.begin()+1),
                  "repeated fused histogram mismatch");
    }
    Passed("npu_direct_fused_"+std::to_string(features)+"x"+std::to_string(rows));
  }
}

void QuantizerIntegration(npu::Runtime const& r,npu::Ops& ops) {
  auto c=ht::Cases().front(); std::vector<float> gf{-2,1,3,0,-1,2,4,0},hf{1,1,2,0,1,3,2,0};
  npu::Tensor g{r,8,ACL_FLOAT},h{r,8,ACL_FLOAT};
  g.Storage().Upload(gf.data(),32); h.Storage().Upload(hf.data(),32);
  npu::DynamicQuantizer quantizer{ops}; auto qg=quantizer.Convert(g,false),qh=quantizer.Convert(h,true);
  qt::Require(qg.certificate.scale.exponent==58 && qh.certificate.scale.exponent==58,"unexpected integration scales");
  std::vector<npu::Tensor> cols; cols.push_back(Upload(r,c.columns[0]));
  Builder builder{ops,std::move(cols),std::move(qg.channel.values),std::move(qh.channel.values),c.bin_counts};
  auto actual=Build(builder,r,c.selected); Compare(actual,ht::Oracle(c,c.selected),8);
  Passed("npu_dynamic_quantizer_histogram_integration");
}
void Tests() {
  int errors=0; auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error=[&](std::string const& message){ ++errors; std::cerr<<"cleanup failure: "<<message<<'\n'; };
  {
    npu::Runtime r{api,0}; npu::Ops ops{r}; Cases(r,ops); SharedBins(r,ops); Guards(r,ops); DirectKernelShapes(r); QuantizerIntegration(r,ops);
    qt::Require(r.LiveBytes()==0,"device storage leaked");
  }
  qt::Require(errors==0,"device resource cleanup failed"); Passed("npu_histogram_resources_released");
}
void Provenance() {
  for (auto symbol:{"aclInit","aclnnIndexSelect","aclnnScatterAdd","aclnnAdd"}) {
    Dl_info info{}; qt::Require(dladdr(dlsym(RTLD_DEFAULT,symbol),&info)&&info.dli_fname,"missing library provenance");
    std::cout<<"library "<<symbol<<' '<<info.dli_fname<<'\n';
  }
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" && std::string(argv[2])=="0" &&
              std::string(argv[3])=="--confirm-device-idle";
  if (!host && !device) { std::cerr<<"Usage: integer_histogram_test --host-api-only | --device-test 0 --confirm-device-idle\n"; return 2; }
  try {
    Provenance(); npu::MakeAclApi()->Validate();
    if (host) { std::cout<<"INTEGER_HISTOGRAM_API_HOST_PASS device_execution=not_run\n"; return 0; }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try { Tests(); }
    catch (...) { auto status=aclFinalize(); if (status) std::cerr<<"aclFinalize status="<<status<<'\n'; throw; }
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    std::cout<<"INTEGER_HISTOGRAM_NPU_PASS checks="<<checks<<" cuda_execution=not_run training_backend=false\n"; return 0;
  } catch (std::exception const& e) { std::cerr<<"INTEGER_HISTOGRAM_FAIL: "<<e.what()<<'\n'; return 1; }
}
