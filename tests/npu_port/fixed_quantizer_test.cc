// Copyright 2026, XGBoost Contributors
// Independent supplied-scale conversion + root test, not fitted quantization
// or XGBoost tree training. The binary64 reference is test-only.
#include "../../src/common/npu/fixed_quantizer.h"
#include "quantizer_cases.h"
#include <dlfcn.h>
#include <fstream>

namespace npu = xgboost::common::npu;
namespace qt = quantizer_test;
namespace {
void AllocationFailureTests() {
  int resources=0, cancelled=0, executed=0, cleanup_errors=0;
  auto api=std::make_shared<npu::Api>();
  api->create_context=[&](int)->void* { ++resources; return api.get(); };
  api->set_context=[](void*) {};
  api->destroy_context=[&](void*) { --resources; };
  api->create_stream=[&]()->void* { ++resources; return api.get(); };
  api->destroy_stream=[&](void*) { --resources; };
  api->synchronize=[](void*) {};
  api->free=[](void*) { throw std::runtime_error("unexpected free"); };
  api->copy=[](void*,void const*,std::size_t,bool) { throw std::runtime_error("unexpected copy"); };
  api->zero=[](void*,std::size_t) { throw std::runtime_error("unexpected zero"); };
  api->cleanup_error=[&](std::string const&) { ++cleanup_errors; };
  int token=0;
  {
    api->allocate=[](std::size_t)->void* { throw std::bad_alloc{}; };
    npu::Runtime runtime{api,0}; npu::Ops ops{runtime};
    auto prepare=[&](uint64_t* bytes,aclOpExecutor** executor) {
      *bytes=64; *executor=reinterpret_cast<aclOpExecutor*>(&token); return aclnnStatus{0};
    };
    auto execute=[&](void*,uint64_t,aclOpExecutor*,aclrtStream) { ++executed; return aclnnStatus{0}; };
    auto cancel=[&](aclOpExecutor* p) {
      qt::Require(p==reinterpret_cast<aclOpExecutor*>(&token),"cancelled wrong executor");
      ++cancelled; return aclnnStatus{0};
    };
    bool failed=false;
    try { ops.Run("injected-allocation-failure",prepare,execute,cancel); }
    catch (std::bad_alloc const&) { failed=true; }
    qt::Require(failed && cancelled==1 && executed==0,"allocation failure lost prepared executor");
    qt::Passed("host_workspace_exception_cancels_executor");
    api->allocate=[](std::size_t)->void* { return nullptr; }; failed=false;
    try { ops.Run("injected-null-allocation",prepare,execute,cancel); }
    catch (std::runtime_error const&) { failed=true; }
    qt::Require(failed && cancelled==2 && executed==0 && runtime.LiveBytes()==0,
                "null allocation leaked prepared executor");
    qt::Passed("host_null_workspace_cancels_executor");
  }
  qt::Require(resources==0 && cleanup_errors==0,"injected runtime cleanup failure");
}
void DeviceTests(int ordinal) {
  int cleanup_errors=0;
  auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error=[&](std::string const& s) { ++cleanup_errors; std::cerr<<"cleanup failure: "<<s<<'\n'; };
  {
    npu::Runtime runtime{api,ordinal}; npu::Ops ops{runtime}; npu::FixedQuantizer quantizer{ops};
    for (auto const& c : qt::Cases()) {
      auto baseline=runtime.LiveBytes();
      {
        npu::Tensor input{runtime,static_cast<int64_t>(c.input.size()),ACL_FLOAT};
        input.Storage().Upload(c.input.data(),input.Storage().Size());
        auto downloaded=ops.ControlDownloadBytes(); auto launches=ops.Launches();
        bool rejected=false, accepted=false;
        try {
          auto actual=quantizer.Convert(input,c.scale,c.hessian); accepted=true;
          if (!c.reject) {
            auto expected=qt::Expected(c);
            std::vector<int64_t> values(c.input.size());
            actual.values.Storage().Download(values.data(),values.size()*sizeof(int64_t));
            if (values!=expected.values) {
              auto mismatch=std::mismatch(values.begin(),values.end(),expected.values.begin());
              auto i=std::distance(values.begin(),mismatch.first);
              throw std::runtime_error(c.name+" integer mismatch row="+std::to_string(i)+
                " actual="+std::to_string(values[i])+" expected="+std::to_string(expected.values[i]));
            }
            qt::Require(actual.root==expected.root && actual.positive_magnitude==expected.positive &&
                        actual.negative_magnitude==expected.negative,"root/magnitude summary mismatch");
            qt::Require(ops.ControlDownloadBytes()-downloaded==48,"component downloaded non-summary data");
          }
        } catch (std::invalid_argument const&) { if (!c.reject) throw; rejected=true; }
          catch (std::overflow_error const&) { if (!c.reject) throw; rejected=true; }
        qt::Require(c.reject ? rejected && !accepted : accepted,"component rejection expectation mismatch");
        std::cout<<"{\"test\":\"npu_"<<c.name<<"\",\"status\":\"passed\",\"rows\":"<<c.input.size()
          <<",\"scale_exponent\":"<<c.scale.exponent<<",\"operator_launches\":"<<ops.Launches()-launches
          <<",\"component_control_download_bytes\":"<<ops.ControlDownloadBytes()-downloaded
          <<",\"bounds_source\":\"supplied_test_fixture\"}\n";
      }
      qt::Require(runtime.LiveBytes()==baseline,"component leaked device storage");
    }
    {
      npu::Tensor bad{runtime,1,ACL_INT64}; bool rejected=false;
      try { quantizer.Convert(bad,npu::QuantizerScale::FromExponent(0),false); }
      catch (std::invalid_argument const&) { rejected=true; }
      qt::Require(rejected,"non-float32 input accepted"); qt::Passed("npu_reject_wrong_dtype");
    }
    {
      npu::Tensor input{runtime,1,ACL_FLOAT}; auto scale=npu::QuantizerScale::FromExponent(0);
      scale.to_fixed=2; auto launches=ops.Launches(); bool rejected=false;
      try { quantizer.Convert(input,scale,false); }
      catch (std::invalid_argument const&) { rejected=true; }
      qt::Require(rejected && launches==ops.Launches(),"inconsistent scale reached kernels");
      qt::Passed("npu_reject_inconsistent_scale_before_launch");
    }
    qt::Require(runtime.LiveBytes()==0,"device memory not released");
  }
  qt::Require(cleanup_errors==0,"runtime cleanup errors"); qt::Passed("npu_quantizer_resources_released");
}
void Provenance() {
  for (auto symbol : {"aclInit","aclnnLeftShift","aclnnRightShift","aclnnReduceSum"}) {
    Dl_info info{};
    qt::Require(dladdr(dlsym(RTLD_DEFAULT,symbol),&info) && info.dli_fname,"missing library provenance");
    std::cout<<"library "<<symbol<<' '<<info.dli_fname<<'\n';
  }
}
}  // namespace
int main(int argc,char** argv) {
  try {
    bool host=argc==2 && std::string(argv[1])=="--host-api-only";
    bool device=argc==4 && std::string(argv[1])=="--device-test" &&
                std::string(argv[3])=="--confirm-device-idle";
    if (!host && !device) {
      std::cerr<<"Usage: fixed_quantizer_test --host-api-only\n"
        <<"       fixed_quantizer_test --device-test DEVICE --confirm-device-idle\n"; return 2;
    }
    int ordinal=0;
    if (device) {
      std::size_t consumed=0; ordinal=std::stoi(argv[2],&consumed);
      qt::Require(consumed==std::string(argv[2]).size() && ordinal>=0,"invalid device ordinal");
    }
    Provenance(); npu::MakeAclApi()->Validate();
    if (host) {
      AllocationFailureTests();
      std::cout<<"FIXED_QUANTIZER_API_HOST_PASS checks=2 device_execution=not_run\n"; return 0;
    }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try { DeviceTests(ordinal); }
    catch (...) { auto status=aclFinalize(); if (status) std::cerr<<"aclFinalize status="<<status<<'\n'; throw; }
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) {
      if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    }
    std::cout<<"FIXED_QUANTIZER_NPU_PASS checks=27 dynamic_device_bounds=not_implemented training_backend=false\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr<<"FIXED_QUANTIZER_FAIL: "<<e.what()<<'\n'; return 1;
  }
}
