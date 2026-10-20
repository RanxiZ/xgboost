// Copyright 2026, XGBoost Contributors
#include "../../src/common/npu/numerical_split_evaluator.h"
#include "numerical_split_cases.h"
#include <fstream>
namespace npu=xgboost::common::npu;
namespace p=npu::precise;
namespace nt=numerical_test;
namespace t=precise_test;
namespace qt=quantizer_test;
namespace {
std::size_t checks=0;bool measure=false;std::vector<std::size_t> downloads;
std::size_t allocation_calls=0;
void Passed(std::string const& name) {++checks;qt::Passed(name);}
template<typename T> npu::Buffer Upload(npu::Runtime const& r,std::vector<T> const& values) {
  auto out=r.AllocateElements<T>(values.size());out.Upload(values.data(),values.size()*sizeof(T));return out;
}
npu::Tensor Tensor(npu::Runtime const& r,std::vector<int64_t> const& values) {
  npu::Tensor out{r,static_cast<int64_t>(values.size()),ACL_INT64};out.Storage().Upload(values.data(),values.size()*8);return out;
}
void BatchStatus(npu::Runtime const& r) {
  // Omit each producer in turn. Consumers must reject zero completion flags
  // without touching uninitialized payload; all stages present must succeed.
  for (unsigned omitted=0;omitted<4;++omitted) {
    auto g=Upload(r,std::vector<int64_t>{-3,3}),h=Upload(r,std::vector<int64_t>{1,1});
    auto root_g=Upload(r,std::vector<int64_t>{0}),root_h=Upload(r,std::vector<int64_t>{2});
    p::NodeBatchHeader header{t::Params(0,1),1,0};
    p::NodeBatchFeature feature{reinterpret_cast<uint64_t>(g.Data()),reinterpret_cast<uint64_t>(h.Data()),0,0,2,7};
    std::vector<uint64_t> words((sizeof(header)+sizeof(feature))/8);
    std::memcpy(words.data(),&header,sizeof(header));
    std::memcpy(reinterpret_cast<char*>(words.data())+sizeof(header),&feature,sizeof(feature));
    auto metadata=Upload(r,words);
    auto pg=r.AllocateElements<int64_t>(3),ph=r.AllocateElements<int64_t>(3);
    auto scores=r.AllocateElements<p::DirectionOutput>(2),selected=r.AllocateElements<p::Candidate>(1);
    auto status=r.AllocateElements<uint32_t>(4);status.Zero();
    p::SplitSummary seed{};seed.error=p::kBatchIncomplete;
    auto summary=Upload(r,std::vector<p::SplitSummary>{seed});
    auto* prefix_status=static_cast<char*>(status.Data());
    auto* score_status=prefix_status+sizeof(uint32_t);auto* selected_status=score_status+2*sizeof(uint32_t);
    auto stream=r.Stream();
    if (omitted!=0) XgbNpuBatchedSplitPrefix(stream,metadata.Data(),pg.Data(),ph.Data(),prefix_status,1);
    if (omitted!=1) XgbNpuBatchedSplitScore(stream,metadata.Data(),root_g.Data(),root_h.Data(),pg.Data(),ph.Data(),prefix_status,scores.Data(),score_status,1);
    if (omitted!=2) XgbNpuBatchedSplitSelect(stream,metadata.Data(),prefix_status,scores.Data(),score_status,selected.Data(),selected_status,1);
    XgbNpuBatchedSplitFinish(stream,metadata.Data(),root_g.Data(),root_h.Data(),pg.Data(),ph.Data(),prefix_status,selected.Data(),selected_status,summary.Data());
    p::SplitSummary out{};summary.Download(&out,sizeof(out));
    qt::Require(omitted==3?out.error==0:(out.error&p::kBatchIncomplete)!=0,"batch incomplete stage status lost");
  }
}
void Candidates(npu::Runtime const& r) {
  for (auto const& c:selection_test::Cases()) {
    if (c.left.empty()) continue;
    auto l=Upload(r,c.left),right=Upload(r,c.right);
    p::Candidate canary{{0x12345678U,0x23456789U,0x34567890U,0x45678901U},11,12,13,14,15,16};
    auto output=Upload(r,std::vector<p::Candidate>(3,canary));
    XgbNpuSelectCandidates(r.Stream(),l.Data(),right.Data(),static_cast<char*>(output.Data())+sizeof(p::Candidate),c.left.size(),7);
    r.Synchronize();std::vector<p::Candidate> actual(3);output.Download(actual.data(),actual.size()*sizeof(actual[0]));
    qt::Require(std::memcmp(&actual.front(),&canary,sizeof(canary))==0 && std::memcmp(&actual.back(),&canary,sizeof(canary))==0,"candidate output canary overwrite");
    selection_test::Compare(actual[1],selection_test::Expected(c));
  }
  // Controlled device witness: subtracting a large parent coalesces distinct
  // raw gains, so selecting after subtraction would pick candidate zero.
  selection_test::Case witness{{selection_test::Score(0),selection_test::Score(0)},
                               {selection_test::Score(0x3f800000U),selection_test::Score(0x3f800001U)}};
  auto left=Upload(r,witness.left),right=Upload(r,witness.right);
  auto selected=r.AllocateElements<p::Candidate>(1);
  XgbNpuSelectCandidates(r.Stream(),left.Data(),right.Data(),selected.Data(),2,7);
  p::SplitSummary initial{};initial.best=p::EmptyCandidate();initial.node.gain=0x4e800000U;
  auto summary=Upload(r,std::vector<p::SplitSummary>{initial});auto channel=Upload(r,std::vector<int64_t>{11,22});
  std::vector<uint64_t> pointers(8,reinterpret_cast<uint64_t>(channel.Data()));auto columns=Upload(r,pointers);
  XgbNpuSplitMerge(r.Stream(),selected.Data(),columns.Data(),summary.Data());XgbNpuSplitFinish(r.Stream(),summary.Data());
  p::SplitSummary actual{};summary.Download(&actual,sizeof(actual));
  qt::Require(actual.best.candidate==1 && actual.best.score.gain==0x3f800001U && !actual.error &&
    actual.loss_change==p::SubtractParent(0x3f800000U,initial.node.gain).loss_change && actual.left_gradient==22,
    "device parent subtraction happened before raw-gain selection");
  BatchStatus(r);
  Passed("npu_candidate_ties_31_32_33_65_257_nonfinite_tail_canaries");
}
void Compare(npu::Ops& ops,npu::IntegerHistogramBuilder::Histogram const& histogram,histogram_test::Case const& c,std::vector<int64_t> const& rows,p::Parameters parameters,std::vector<uint32_t> order,npu::NumericalSplitEvaluator* reused=nullptr) {
  npu::NumericalSplitEvaluator fresh{ops};auto& evaluator=reused?*reused:fresh;
  auto before_bytes=evaluator.ControlDownloadBytes(),before_launches=evaluator.Launches();
  auto live_bytes=ops.GetRuntime().LiveBytes();
  downloads.clear();measure=true;p::SplitSummary result{};
  try {result=evaluator.Evaluate(histogram,parameters,order);} catch (...) {measure=false;throw;}
  measure=false;
  nt::Compare(result,nt::Expected(c,rows,parameters,order));
  qt::Require(evaluator.ControlDownloadBytes()-before_bytes==sizeof(p::SplitSummary) &&
    downloads==std::vector<std::size_t>{sizeof(p::SplitSummary)},"batch evaluator must download exactly one 112-byte summary");
  qt::Require(evaluator.Launches()-before_launches==4,"batch evaluator launch count depends on features");
  qt::Require(ops.GetRuntime().LiveBytes()==live_bytes,"batch evaluator retained fresh workspace");
}
void Integration(npu::Runtime const& r,npu::Ops& ops) {
  for (auto const& c:nt::Cases()) {
    std::vector<npu::Tensor> columns;for (auto const& col:c.columns) columns.push_back(Tensor(r,col));
    npu::IntegerHistogramBuilder builder{ops,std::move(columns),Tensor(r,c.gradient),Tensor(r,c.hessian),c.bin_counts};
    auto rows=c.selected.empty()?nullptr:std::make_unique<npu::Tensor>(Tensor(r,c.selected));auto histogram=builder.Build(rows.get());
    std::vector<uint32_t> order(c.bin_counts.size());std::iota(order.begin(),order.end(),0);
    auto parameters=t::Params(0,1,0.2f,0.7f,58,58);
    Compare(ops,histogram,c,c.selected,parameters,order);
    std::reverse(order.begin(),order.end());Compare(ops,histogram,c,c.selected,parameters,order);
    auto [left,right]=histogram_test::Split(c.selected);auto ids=left.empty()?nullptr:std::make_unique<npu::Tensor>(Tensor(r,left));
    auto child=builder.Build(ids.get());auto sibling=builder.SubtractSibling(histogram,child);Compare(ops,sibling,c,right,parameters,order);
    Compare(ops,histogram,c,c.selected,parameters,{});
    Compare(ops,histogram,c,c.selected,t::Params(1000000,1,0,0,58,58),order);
    Passed("npu_full_numerical_"+c.name);
  }
  // Additional coverage stays inside the existing integration group; preserve
  // every independent oracle case and its externally checked Passed records.
  npu::NumericalSplitEvaluator reused{ops};
  for (bool stripes:{false,true}) {
    histogram_test::Case c{"mixed_batch",{-3,1,7,5},{1,2,3,4},
      {0,1,31,32,33,65,127,128,129,256,257},{},{0,1,2,3}};
    if (stripes) {
      c.bin_counts.assign(513,1);c.bin_counts[1]=2;c.bin_counts[256]=2;
      c.bin_counts[512]=2;
    }
    for (auto bins:c.bin_counts) {
      c.columns.push_back(bins?std::vector<int64_t>{0,bins-1,bins/2,-1}:std::vector<int64_t>(4,-1));
    }
    std::vector<npu::Tensor> columns;for (auto const& col:c.columns) columns.push_back(Tensor(r,col));
    npu::IntegerHistogramBuilder builder{ops,std::move(columns),Tensor(r,c.gradient),Tensor(r,c.hessian),c.bin_counts};
    auto rows=Tensor(r,c.selected);auto histogram=builder.Build(&rows);
    std::vector<uint32_t> order(c.bin_counts.size());std::iota(order.begin(),order.end(),0);
    auto parameters=t::Params(0,1,0.2f,0.7f,58,58);
    Compare(ops,histogram,c,c.selected,parameters,order,&reused);
    Compare(ops,histogram,c,c.selected,parameters,{},&reused);
    Compare(ops,histogram,c,c.selected,parameters,{0},&reused);
    std::reverse(order.begin(),order.end());
    Compare(ops,histogram,c,c.selected,parameters,order,&reused);
    Compare(ops,histogram,c,c.selected,t::Params(1000000,1,0,0,58,58),order,&reused);
    // Alternate large/empty/striped orders and batch sizes. Each node keeps its
    // own parameters and tie order even when adjacent workspace shapes change.
    for (std::size_t count : {3U, 1U, 7U, 2U}) {
      std::vector<npu::NumericalSplitEvaluator::Request> batch;
      std::vector<p::SplitSummary> expected;
      for (std::size_t i = 0; i < count; ++i) {
        auto selected = i % 3 == 0 ? order : std::vector<uint32_t>{};
        if (i % 3 == 2) selected = {static_cast<uint32_t>(c.bin_counts.size() - 1), 0};
        if (!selected.empty()) std::rotate(selected.begin(), selected.begin() + i % selected.size(), selected.end());
        auto params = t::Params(i % 2 ? 1000000 : 0, static_cast<float>(i + 1),
                                0.2f, 0.7f, 58, 58);
        batch.push_back({&histogram, params, selected});
        expected.push_back(nt::Expected(c, c.selected, params, selected));
      }
      auto bytes = r.LiveBytes();
      downloads.clear(); allocation_calls = 0; measure = true;
      std::vector<p::SplitSummary> actual;
      try { actual = reused.EvaluateBatch(batch); } catch (...) { measure = false; throw; }
      measure = false;
      qt::Require(actual.size() == count && allocation_calls <= 1 && r.LiveBytes() == bytes &&
                  downloads == std::vector<std::size_t>{count * sizeof(p::SplitSummary)},
                  "changing striped node batches lost workspace/download isolation");
      for (std::size_t i = 0; i < count; ++i) nt::Compare(actual[i], expected[i]);
    }
  }
}
void MultiNode(npu::Runtime const& r, npu::Ops& ops) {
  auto cases = nt::Cases();
  std::vector<std::unique_ptr<npu::IntegerHistogramBuilder>> builders;
  std::vector<npu::IntegerHistogramBuilder::Histogram> histograms;
  for (std::size_t i = 0; i < 2; ++i) {
    auto const& c = cases[i]; std::vector<npu::Tensor> columns;
    for (auto const& col : c.columns) columns.push_back(Tensor(r, col));
    builders.push_back(std::make_unique<npu::IntegerHistogramBuilder>(
        ops, std::move(columns), Tensor(r, c.gradient), Tensor(r, c.hessian), c.bin_counts));
    auto rows = c.selected.empty() ? nullptr : std::make_unique<npu::Tensor>(Tensor(r, c.selected));
    histograms.push_back(builders.back()->Build(rows.get()));
  }
  auto empty = builders[0]->Build(nullptr);
  npu::NumericalSplitEvaluator evaluator{ops};
  using Request = npu::NumericalSplitEvaluator::Request;
  std::vector<Request> requests;
  std::vector<p::SplitSummary> expected;
  for (std::size_t i = 0; i < 2; ++i) {
    std::vector<uint32_t> order(cases[i].bin_counts.size()); std::iota(order.begin(), order.end(), 0);
    if (i) std::reverse(order.begin(), order.end());
    auto params = t::Params(0, static_cast<float>(i + 1), 0.2f, 0.7f, 58, 58);
    requests.push_back({&histograms[i], params, order});
    expected.push_back(nt::Expected(cases[i], cases[i].selected, params, order));
  }
  requests.push_back({&empty, t::Params(), {}});
  expected.push_back(nt::Expected(cases[0], {}, t::Params(), {}));
  requests.push_back({&histograms[0], t::Params(0, 1, 0, 0, 57, 58), {}});
  expected.push_back(nt::Expected(cases[0], cases[0].selected, requests.back().parameters, {}));
  auto bytes = r.LiveBytes(), launches = evaluator.Launches();
  auto controls = evaluator.ControlDownloadBytes();
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    downloads.clear(); allocation_calls = 0; measure = true;
    std::vector<p::SplitSummary> actual;
    try { actual = evaluator.EvaluateBatch(requests); } catch (...) { measure = false; throw; }
    measure = false;
    qt::Require(actual.size() == expected.size(), "multi-node summary count");
    for (std::size_t i = 0; i < actual.size(); ++i) nt::Compare(actual[i], expected[i]);
    qt::Require(downloads == std::vector<std::size_t>{actual.size() * sizeof(p::SplitSummary)},
                "multi-node evaluation downloaded per-node payloads");
    qt::Require(allocation_calls <= 1, "multi-node evaluation did not share a workspace allocation");
    qt::Require(r.LiveBytes() == bytes, "multi-node workspace leaked");
  }
  qt::Require(evaluator.Launches() - launches == 8 * requests.size() &&
              evaluator.ControlDownloadBytes() - controls == 2 * requests.size() * sizeof(p::SplitSummary),
              "multi-node launch or scalar-summary contract changed");
  auto valid_launches = evaluator.Launches(); controls = evaluator.ControlDownloadBytes();
  qt::Require(evaluator.EvaluateBatch({}).empty() && evaluator.Launches() == valid_launches,
              "empty multi-node batch launched kernels");
  for (auto bad : {Request{nullptr, t::Params(), {}},
                   Request{&histograms[0], t::Params(), {UINT32_MAX}},
                   Request{&histograms[0], t::Params(-0.0f), {}}}) {
    auto invalid = requests; invalid.push_back(bad); bool rejected = false;
    allocation_calls = 0; measure = true;
    try { evaluator.EvaluateBatch(invalid); } catch (std::invalid_argument const&) { rejected = true; }
    measure = false;
    qt::Require(rejected && !allocation_calls && evaluator.Launches() == valid_launches &&
                evaluator.ControlDownloadBytes() == controls && r.LiveBytes() == bytes,
                "late multi-node metadata guard allocated or launched");
  }
  auto invalid = requests;
  invalid.push_back({&histograms[0], t::Params(0, 0, 0, 0, -1022, 1022), {0}});
  bool rejected = false;
  try { evaluator.EvaluateBatch(invalid); } catch (std::overflow_error const&) { rejected = true; }
  qt::Require(rejected && r.LiveBytes() == bytes, "multi-node arithmetic-domain rejection leaked");
  Passed("npu_multi_node_isolation_single_workspace_and_download");
}
void Guards(npu::Runtime const& r,npu::Ops& ops) {
  std::vector<npu::Tensor> cols;cols.push_back(Tensor(r,{0,1}));
  npu::IntegerHistogramBuilder b{ops,std::move(cols),Tensor(r,{-3,3}),Tensor(r,{1,1}),{2}};
  auto ids=Tensor(r,{0,1});auto hist=b.Build(&ids);npu::NumericalSplitEvaluator e{ops};auto bytes=r.LiveBytes();
  for (auto bad:{t::Params(0,-1),t::Params(0,1,0,0,1023,0),t::Params(0,std::numeric_limits<float>::infinity()),t::Params(-0.0f)}) {
    bool rejected=false;try {e.Evaluate(hist,bad,{0});} catch (std::invalid_argument const&) {rejected=true;}
    qt::Require(rejected && r.LiveBytes()==bytes && e.ControlDownloadBytes()==0,"parameter guard allocated/launched");
  }
  for (auto order:{std::vector<uint32_t>{0,0},std::vector<uint32_t>{1},std::vector<uint32_t>{UINT32_MAX}}) {
    bool rejected=false;try {e.Evaluate(hist,t::Params(),order);} catch (std::invalid_argument const&) {rejected=true;}
    qt::Require(rejected && r.LiveBytes()==bytes,"feature guard leaked");
  }
  Passed("npu_evaluator_strict_parameter_feature_guards");
  bool rejected=false;try {e.Evaluate(hist,t::Params(0,0,0,0,-1022,1022),{0});} catch (std::overflow_error const&) {rejected=true;}
  qt::Require(rejected && r.LiveBytes()==bytes,"domain rejection leaked");Passed("npu_evaluator_device_domain_rejection");
  npu::Runtime foreign{npu::MakeAclApi(),0};npu::Ops other_ops{foreign};npu::NumericalSplitEvaluator other{other_ops};
  rejected=false;try {other.Evaluate(hist,t::Params(),{0});} catch (std::invalid_argument const&) {rejected=true;}
  qt::Require(rejected && foreign.LiveBytes()==0 && !other.ControlDownloadBytes(),"foreign runtime guard");Passed("npu_evaluator_runtime_provenance_guard");
}
void Tests() {
  int errors=0;auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());auto copy=api->copy;
  api->copy=[copy](void* a,void const* b,std::size_t bytes,bool h2d){if (measure&&!h2d) downloads.push_back(bytes);copy(a,b,bytes,h2d);};
  auto allocate = api->allocate;
  api->allocate = [allocate](std::size_t bytes) { if (measure) ++allocation_calls; return allocate(bytes); };
  api->cleanup_error=[&](std::string const& message){++errors;std::cerr<<"cleanup failure: "<<message<<'\n';};
  {npu::Runtime r{api,0};npu::Ops ops{r};Candidates(r);Integration(r,ops);MultiNode(r,ops);Guards(r,ops);qt::Require(!r.LiveBytes(),"evaluator leaked device memory");}
  qt::Require(!errors,"evaluator cleanup failed");Passed("npu_evaluator_resources_released");
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" && std::string(argv[2])=="0" && std::string(argv[3])=="--confirm-device-idle";
  if (!host&&!device) {std::cerr<<"Usage: numerical_split_test --host-api-only | --device-test 0 --confirm-device-idle\n";return 2;}
  try {
    npu::MakeAclApi()->Validate();
    if (host) {XgbNpuSelectCandidates(nullptr,nullptr,nullptr,nullptr,0,0);std::cout<<"NUMERICAL_SPLIT_API_HOST_PASS\n";return 0;}
    npu::CheckAcl(aclInit(nullptr),"aclInit");try {Tests();} catch (...) {aclFinalize();throw;}npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");for (std::string line;std::getline(maps,line);) if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    std::cout<<"NUMERICAL_SPLIT_NPU_PASS checks="<<checks<<" training_backend=false\n";return 0;
  } catch (std::exception const& ex) {std::cerr<<"NUMERICAL_SPLIT_NPU_FAIL: "<<ex.what()<<'\n';return 1;}
}