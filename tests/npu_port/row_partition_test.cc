// Copyright 2026, XGBoost Contributors
// Independent component tests; full downloads below are test-only inspection.
#include "../../src/common/npu/device_row_partitioner.h"
#include "row_partition_cases.h"
#include <dlfcn.h>
#include <fstream>
#include <type_traits>
namespace npu = xgboost::common::npu;
namespace rt = row_partition_test;
namespace qt = quantizer_test;
using Builder = npu::IntegerHistogramBuilder;
using Part = npu::DeviceRowPartitioner;
namespace xgboost::common::npu {
class RowPartitionTestAccess {
 public:
  static Tensor const* Ids(DeviceRowPartitioner::Rows const& rows) { return rows.ids_.get(); }
  static bool Buffered(DeviceRowPartitioner::Rows const& rows) { return static_cast<bool>(rows.arena_); }
  static void* Base(DeviceRowPartitioner::Rows const& rows) { return rows.arena_->At(rows.slot_).Storage().Data(); }
  static int64_t Offset(DeviceRowPartitioner::Rows const& rows) { return rows.offset_; }
};
}
namespace {
template<typename T,typename=void> struct HasStorage : std::false_type {};
template<typename T> struct HasStorage<T,std::void_t<decltype(std::declval<T const&>().Storage())>> : std::true_type {};
static_assert(!HasStorage<Part::Rows>::value && !std::is_copy_constructible_v<Part::Rows>);
static_assert(!std::is_default_constructible_v<Part::Rows>);
std::size_t checks=0;
void Passed(std::string const& name) { ++checks; qt::Passed(name); }
template<typename F> void Reject(std::string const& name,F fn) {
  bool rejected=false; try { fn(); } catch (std::invalid_argument const&) { rejected=true; }
  rt::Require(rejected,"unsupported row-partition input accepted"); Passed(name);
}
npu::Tensor Upload(npu::Runtime const& r,std::vector<int64_t> const& values) {
  npu::Tensor t{r,static_cast<int64_t>(values.size()),ACL_INT64};
  t.Storage().Upload(values.data(),values.size()*sizeof(int64_t)); return t;
}
std::unique_ptr<Builder> Make(npu::Ops& ops,rt::Case const& c) {
  std::vector<npu::Tensor> columns; columns.push_back(Upload(ops.GetRuntime(),c.bins));
  return std::make_unique<Builder>(ops,std::move(columns),ops.Constant(c.bins.size(),0),
                                  ops.Constant(c.bins.size(),1),std::vector<int64_t>{c.bin_count});
}
Builder::Histogram Hist(Builder& b,npu::Runtime const& r,std::vector<int64_t> const& rows) {
  if (rows.empty()) return b.Build(nullptr);
  auto ids=Upload(r,rows); return b.Build(&ids);
}
Part::Rows Capture(Part& p,Builder::Histogram const& h,npu::Runtime const& r,std::vector<int64_t> const& rows) {
  if (rows.empty()) return p.Capture(h,nullptr);
  auto ids=Upload(r,rows); return p.Capture(h,&ids);
}
void Compare(Part::Rows const& rows,std::vector<int64_t> const& expected) {
  auto t=npu::RowPartitionTestAccess::Ids(rows);
  rt::Require(rows.Count()==static_cast<int64_t>(expected.size()),"child count mismatch");
  if (expected.empty()) { rt::Require(!t,"empty child has tensor"); return; }
  rt::Require(t && t->Type()==ACL_INT64,"child row tensor dtype");
  if (npu::RowPartitionTestAccess::Buffered(rows)) {
    auto base = reinterpret_cast<uintptr_t>(npu::RowPartitionTestAccess::Base(rows));
    auto offset = npu::RowPartitionTestAccess::Offset(rows);
    rt::Require(reinterpret_cast<uintptr_t>(t->Storage().Data()) == base + offset * sizeof(int64_t),
                "buffered row view does not match its node interval");
  }
  std::vector<int64_t> actual(t->Count()); t->Storage().Download(actual.data(),actual.size()*sizeof(int64_t));
  // Physical ordering is deliberately not a CUDA parity requirement.
  auto sorted=expected; std::sort(actual.begin(),actual.end()); std::sort(sorted.begin(),sorted.end());
  rt::Require(actual==sorted,"device child membership mismatch");
}
void Cases(npu::Runtime const& r,npu::Ops& ops) {
  for (auto const& c:rt::Cases()) {
    auto live=r.LiveBytes();
    {
      auto builder=Make(ops,c); auto h=Hist(*builder,r,c.selected); Part p{ops,h};
      auto before=ops.ControlDownloadBytes();
      auto rows=Capture(p,h,r,c.selected);
      rt::Require(ops.ControlDownloadBytes()-before==(c.selected.empty()?0:24),"capture downloaded rows");
      before=ops.ControlDownloadBytes(); auto launch=ops.Launches();
      auto out=p.Partition(rows,c.split); auto expected=rt::Oracle(c);
      rt::Require(ops.ControlDownloadBytes()-before==(c.selected.empty()?0:16),"partition exceeded scalar controls");
      Compare(out.left,expected.left); Compare(out.right,expected.right); Compare(rows,c.selected);
      std::cout<<"{\"test\":\"npu_"<<c.name<<"\",\"status\":\"passed\",\"control_download_bytes\":"
        <<ops.ControlDownloadBytes()-before<<",\"operator_launches\":"<<ops.Launches()-launch<<"}\n"; ++checks;
      auto left_h=p.BuildHistogram(*builder,out.left),right_h=p.BuildHistogram(*builder,out.right);
      rt::Require(left_h.Rows()==out.left.Count() && right_h.Rows()==out.right.Count(),"child histogram rows");
      auto again=p.Partition(out.left,c.split); Compare(again.left,expected.left); Compare(again.right,{});
      // Preserve the immutable dataset even if the original builder is gone.
      builder.reset(); auto after=p.Partition(rows,c.split);
      Compare(after.left,expected.left); Compare(after.right,expected.right);
    }
    rt::Require(r.LiveBytes()==live,"row-partition fixture leaked device storage");
  }
}
void BufferedCases(npu::Runtime const& r, npu::Ops& ops) {
  for (auto c : rt::Cases()) {
    auto baseline = r.LiveBytes();
    {
      c.selected.resize(c.bins.size()); std::iota(c.selected.begin(), c.selected.end(), 0);
      auto b = Make(ops, c); auto empty = b->Build(nullptr); Part p{ops, empty};
      auto before = r.LiveBytes(); auto root = p.RootBuffered();
      auto arena_bytes = 2 * c.bins.size() * sizeof(int64_t);
      rt::Require(r.LiveBytes() - before == arena_bytes, "buffered root did not allocate exactly two row buffers");
      Compare(root, c.selected);
      auto controls = ops.ControlDownloadBytes();
      auto first = p.PartitionConsume(std::move(root), c.split); auto expected = rt::Oracle(c);
      rt::Require(ops.ControlDownloadBytes() - controls == 16, "buffered partition downloaded row membership");
      rt::Require(r.LiveBytes() - before == arena_bytes, "buffered children allocated independent row storage");
      Compare(first.left, expected.left); Compare(first.right, expected.right);
      bool rejected = false;
      try { p.PartitionConsume(std::move(root), c.split); }
      catch (std::invalid_argument const&) { rejected = true; }
      rt::Require(rejected, "consumed parent remained writable");
      // Advance the right branch first, then its sibling, then a grandchild.
      // Mixed depths and retained terminal siblings must never share a range.
      auto right_case = c; right_case.selected = expected.right;
      right_case.split.missing_left = !c.split.missing_left;
      auto right_expected = rt::Oracle(right_case);
      auto right = p.PartitionConsume(std::move(first.right), right_case.split);
      Compare(first.left, expected.left);
      Compare(right.left, right_expected.left); Compare(right.right, right_expected.right);
      auto left = p.PartitionConsume(std::move(first.left), c.split);
      Compare(left.left, expected.left); Compare(left.right, {});
      Compare(right.left, right_expected.left); Compare(right.right, right_expected.right);
      auto again_case = right_case; again_case.selected = right_expected.left;
      auto again_expected = rt::Oracle(again_case);
      auto again = p.PartitionConsume(std::move(right.left), again_case.split);
      Compare(again.left, again_expected.left); Compare(again.right, again_expected.right);
      Compare(right.right, right_expected.right); Compare(left.left, expected.left);
      auto hist = p.BuildHistogram(*b, left.left);
      rt::Require(hist.Rows() == static_cast<int64_t>(expected.left.size()), "buffered row histogram adapter");
      b.reset();
      Compare(again.left, again_expected.left); Compare(left.left, expected.left);
    }
    rt::Require(r.LiveBytes() == baseline, "buffered arena leaked device storage");
  }
  Passed("npu_buffered_ranges_mixed_depth_and_lifetime");
}
void BufferedGuards(npu::Runtime const&, npu::Ops& ops) {
  auto c = rt::Cases().front(); auto b = Make(ops, c); auto h = b->Build(nullptr); Part p{ops, h};
  auto ordinary = p.Root(); auto launches = ops.Launches(); bool rejected = false;
  try { p.PartitionConsume(std::move(ordinary), c.split); }
  catch (std::invalid_argument const&) { rejected = true; }
  rt::Require(rejected && launches == ops.Launches(), "non-buffered consume guard launched operations");
  std::vector<int64_t> all(c.bins.size()); std::iota(all.begin(), all.end(), 0); Compare(ordinary, all);
  auto buffered = p.RootBuffered(); launches = ops.Launches(); rejected = false;
  try { p.PartitionConsume(std::move(buffered), {0, c.bin_count, false}); }
  catch (std::invalid_argument const&) { rejected = true; }
  rt::Require(rejected && launches == ops.Launches(), "invalid consuming split launched operations");
  Compare(buffered, all);
  auto other = Make(ops, c); auto other_h = other->Build(nullptr); Part foreign{ops, other_h};
  rejected = false; launches = ops.Launches();
  try { foreign.PartitionConsume(std::move(buffered), c.split); }
  catch (std::invalid_argument const&) { rejected = true; }
  rt::Require(rejected && launches == ops.Launches(), "consuming partition accepted foreign source");
  Compare(buffered, all);
  Passed("npu_buffered_consuming_guards_preserve_parent");
}
void Guards(npu::Runtime const& r,npu::Ops& ops) {
  auto c=rt::Cases().front(); auto b=Make(ops,c); auto h=Hist(*b,r,{0,1}); Part p{ops,h};
  auto root=p.Root(); std::vector<int64_t> all(c.bins.size()); std::iota(all.begin(),all.end(),0); Compare(root,all);
  Passed("npu_root_generated_on_device");
  for (auto const& item:std::vector<std::pair<std::string,std::vector<int64_t>>>{
       {"negative_row",{-1,0}},{"large_row",{8,0}},{"extreme_row",{INT64_MAX,0}},
       {"duplicate_row",{0,0}},{"wrong_membership",{0,2}},{"wrong_count",{0}}}) {
    Reject("npu_reject_"+item.first,[&]{ Capture(p,h,r,item.second); });
  }
  auto launches=ops.Launches();
  npu::Tensor dtype{r,2,ACL_FLOAT};
  Reject("npu_reject_dtype",[&]{ p.Capture(h,&dtype); });
  auto moved_ids=Upload(r,{0,1}); auto held_ids=std::move(moved_ids);
  Reject("npu_reject_moved_input_tensor",[&]{ p.Capture(h,&moved_ids); });
  rt::Require(held_ids.Count()==2,"moved input holder invalid");
  Reject("npu_reject_null_nonempty",[&]{ p.Capture(h,nullptr); });
  Reject("npu_reject_feature",[&]{ p.Partition(root,{1,0,false}); });
  Reject("npu_reject_negative_candidate",[&]{ p.Partition(root,{0,-1,false}); });
  Reject("npu_reject_endpoint_candidate",[&]{ p.Partition(root,{0,4,false}); });
  rt::Require(ops.Launches()==launches,"metadata guard launched operators");
  auto other=Make(ops,c); auto other_h=Hist(*other,r,{0,1}); Part foreign_source{ops,other_h};
  auto other_root=foreign_source.Root(); launches=ops.Launches();
  Reject("npu_reject_histogram_source",[&]{ Capture(p,other_h,r,{0,1}); });
  Reject("npu_reject_row_source",[&]{ p.Partition(other_root,{0,0,false}); });
  Reject("npu_reject_builder_source",[&]{ p.BuildHistogram(*other,root); });
  rt::Require(ops.Launches()==launches,"source guards launched operators");
  auto moved=std::move(other_root);
  Reject("npu_reject_moved_rows",[&]{ foreign_source.Partition(other_root,{0,0,false}); });
  // Capture owns a device clone; mutating the caller row buffer cannot change it.
  auto ids=Upload(r,{0,1}); auto captured=p.Capture(h,&ids);
  std::vector<int64_t> changed{7,6}; ids.Storage().Upload(changed.data(),16); Compare(captured,{0,1});
  Passed("npu_capture_owns_independent_rows");
  {
    npu::Runtime foreign{npu::MakeAclApi(),0}; npu::Ops foreign_ops{foreign};
    auto foreign_ids=Upload(foreign,{0,1}); launches=ops.Launches();
    Reject("npu_reject_foreign_row_runtime",[&]{ p.Capture(h,&foreign_ids); });
    Reject("npu_reject_foreign_context_runtime",[&]{ Part bad{foreign_ops,h}; });
    rt::Require(ops.Launches()==launches,"runtime guards launched operators");
  }
  for (auto const& name:{"negative_bin","large_bin","zero_bin_value"}) {
    auto bad=c;
    if (std::string(name)=="negative_bin") bad.bins[0]=-2;
    else if (std::string(name)=="large_bin") bad.bins[0]=4;
    else { bad.bin_count=0; bad.bins.assign(8,-1); bad.bins[0]=0; }
    Reject("npu_reject_"+std::string(name),[&]{ Make(ops,bad); });
  }
  c.bins.assign(8,-1); c.bin_count=0; auto zero=Make(ops,c); auto zero_h=zero->Build(nullptr);
  Part zero_p{ops,zero_h}; auto zero_root=zero_p.Root(); launches=ops.Launches();
  Reject("npu_reject_zero_bin_candidate",[&]{ zero_p.Partition(zero_root,{0,0,false}); });
  rt::Require(ops.Launches()==launches,"zero-bin guard launched operators");
}
void FeatureSelection(npu::Runtime const& r,npu::Ops& ops) {
  std::vector<npu::Tensor> columns;
  columns.push_back(Upload(r,{0,0,0,0})); columns.push_back(Upload(r,{2,-1,0,1}));
  Builder b{ops,std::move(columns),ops.Constant(4,0),ops.Constant(4,1),{1,3}};
  auto empty=b.Build(nullptr); Part p{ops,empty}; auto root=p.Root();
  auto out=p.Partition(root,{1,1,true}); Compare(out.left,{1,2}); Compare(out.right,{0,3});
  auto left=p.BuildHistogram(b,out.left); rt::Require(left.FeatureCount()==2 && left.Rows()==2,"selected feature child");
  Passed("npu_selected_nonzero_feature");
}
void Provenance() {
  for (auto symbol:{"aclInit","aclnnIndexSelect","aclnnCumsum","aclnnScatterAdd"}) {
    Dl_info info{}; rt::Require(dladdr(dlsym(RTLD_DEFAULT,symbol),&info) && info.dli_fname,"missing ACL provenance");
    std::cout<<"library "<<symbol<<' '<<info.dli_fname<<'\n';
  }
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" && std::string(argv[2])=="0" &&
              std::string(argv[3])=="--confirm-device-idle";
  if (!host && !device) { std::cerr<<"Usage: row_partition_test --host-api-only | --device-test 0 --confirm-device-idle\n"; return 2; }
  try {
    Provenance(); npu::MakeAclApi()->Validate();
    if (host) { std::cout<<"ROW_PARTITION_API_HOST_PASS device_execution=not_run\n"; return 0; }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try {
      int errors=0; auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
      api->cleanup_error=[&](std::string const& message){ ++errors; std::cerr<<"cleanup failure: "<<message<<'\n'; };
      { npu::Runtime r{api,0}; npu::Ops ops{r}; Cases(r,ops); Guards(r,ops); FeatureSelection(r,ops); BufferedCases(r,ops); BufferedGuards(r,ops);
        rt::Require(r.LiveBytes()==0,"row partition leaked storage"); }
      rt::Require(errors==0,"row partition resource cleanup failed"); Passed("npu_resources_released");
    } catch (...) { auto status=aclFinalize(); if (status) std::cerr<<"aclFinalize status="<<status<<'\n'; throw; }
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    std::cout<<"ROW_PARTITION_NPU_PASS checks="<<checks<<" cuda_execution=not_run training_backend=false\n"; return 0;
  } catch (std::exception const& e) { std::cerr<<"ROW_PARTITION_FAIL: "<<e.what()<<'\n'; return 1; }
}