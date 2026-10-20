// Copyright 2026, XGBoost Contributors
// Device bounds + certified dynamic scale + quantized root; not tree training.
#include "../../src/common/npu/dynamic_quantizer.h"
#include "dynamic_quantizer_cases.h"
#include <dlfcn.h>
#include <fstream>

namespace npu = xgboost::common::npu;
namespace dt = dynamic_test;
namespace qt = quantizer_test;
namespace {
void Tests() {
  int cleanup_errors=0;
  auto api=std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error=[&](std::string const& message) { ++cleanup_errors; std::cerr<<"cleanup failure: "<<message<<'\n'; };
  {
    npu::Runtime runtime{api,0}; npu::Ops ops{runtime}; npu::DynamicQuantizer quantizer{ops};
    for (auto const& c : dt::Cases()) {
      auto baseline=runtime.LiveBytes();
      {
        npu::Tensor input{runtime,static_cast<int64_t>(c.input.size()),ACL_FLOAT};
        input.Storage().Upload(c.input.data(),input.Storage().Size());
        auto launch=ops.Launches(), before=ops.ControlDownloadBytes();
        bool rejected=false; std::size_t fit_bytes=0; int exponent=0;
        try {
          auto bins=quantizer.ReduceBounds(input);
          qt::Require(!c.nonfinite && bins==dt::Buckets(c.input),"device bounds differ from exact integer oracle");
          qt::Require(ops.ControlDownloadBytes()-before==4104,"bounds downloaded per-row data");
          before=ops.ControlDownloadBytes();
          auto actual=quantizer.Convert(input,c.hessian);
          fit_bytes=ops.ControlDownloadBytes()-before; exponent=actual.certificate.scale.exponent;
          qt::Require(!c.ambiguous && fit_bytes==4152,"fit download contract or rejection failed");
          auto expected=qt::Expected({c.name,c.input,actual.certificate.scale,c.hessian});
          std::vector<int64_t> values(c.input.size());
          actual.channel.values.Storage().Download(values.data(),values.size()*sizeof(int64_t));
          qt::Require(values==expected.values && actual.channel.root==expected.root &&
                      actual.channel.positive_magnitude==expected.positive &&
                      actual.channel.negative_magnitude==expected.negative,"dynamic quantization differs from oracle");
          for (bool balanced : {false,true}) {
            auto [p,n]=dt::Sums(c.input,balanced);
            auto reference=npu::QuantizerScale::FromBounds(p,n,c.input.size());
            qt::Require(reference.exponent==exponent,"dynamic scale differs from tested CUDA-style tree");
          }
        } catch (npu::AmbiguousQuantizerScale const&) {
          if (!c.ambiguous) throw;
          rejected=true; fit_bytes=ops.ControlDownloadBytes()-before;
          qt::Require(fit_bytes==4104,"ambiguous certificate ran quantization or downloaded rows");
        } catch (std::invalid_argument const&) {
          if (!c.nonfinite) throw;
          rejected=true; fit_bytes=ops.ControlDownloadBytes()-before;
          qt::Require(fit_bytes==8,"invalid exponent reached bucket scatter");
        }
        qt::Require(rejected==(c.nonfinite||c.ambiguous),"device fixture rejection mismatch");
        std::cout<<"{\"test\":\"npu_"<<c.name<<"\",\"status\":\"passed\",\"rows\":"<<c.input.size()
          <<",\"scale_exponent\":"<<exponent<<",\"fit_control_download_bytes\":"<<fit_bytes
          <<",\"operator_launches\":"<<ops.Launches()-launch
          <<",\"bounds_source\":\"npu_integer_significand_buckets\"}\n";
      }
      qt::Require(runtime.LiveBytes()==baseline,"device bounds/quantization leaked storage");
    }
    {
      npu::Tensor wrong{runtime,1,ACL_INT64}; auto launches=ops.Launches(); bool rejected=false;
      try { quantizer.Convert(wrong,false); } catch (std::invalid_argument const&) { rejected=true; }
      qt::Require(rejected && launches==ops.Launches(),"dtype guard launched kernels");
      qt::Passed("npu_wrong_dtype_before_launch");
    }
    {
      bool rejected=false; try { npu::Tensor empty{runtime,0,ACL_FLOAT}; }
      catch (std::invalid_argument const&) { rejected=true; }
      qt::Require(rejected,"empty tensor accepted"); qt::Passed("npu_empty_input_guard");
    }
    qt::Require(runtime.LiveBytes()==0,"device buffers not released");
  }
  qt::Require(cleanup_errors==0,"context/stream cleanup errors"); qt::Passed("npu_dynamic_resources_released");
}
void Provenance() {
  for (auto symbol : {"aclInit","aclnnScatterAdd","aclnnRightShift","aclnnReduceSum"}) {
    Dl_info info{};
    qt::Require(dladdr(dlsym(RTLD_DEFAULT,symbol),&info) && info.dli_fname,"missing library provenance");
    std::cout<<"library "<<symbol<<' '<<info.dli_fname<<'\n';
  }
}
}
int main(int argc,char** argv) {
  bool host=argc==2 && std::string(argv[1])=="--host-api-only";
  bool device=argc==4 && std::string(argv[1])=="--device-test" &&
              std::string(argv[2])=="0" && std::string(argv[3])=="--confirm-device-idle";
  if (!host && !device) {
    std::cerr<<"Usage: dynamic_quantizer_test --host-api-only | --device-test 0 --confirm-device-idle\n"; return 2;
  }
  try {
    Provenance(); npu::MakeAclApi()->Validate();
    if (host) { std::cout<<"DYNAMIC_QUANTIZER_API_HOST_PASS device_execution=not_run\n"; return 0; }
    npu::CheckAcl(aclInit(nullptr),"aclInit");
    try { Tests(); }
    catch (...) { auto status=aclFinalize(); if (status) std::cerr<<"aclFinalize status="<<status<<'\n'; throw; }
    npu::CheckAcl(aclFinalize(),"aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line;std::getline(maps,line);) {
      if (line.find("/Ascend/")!=std::string::npos) std::cout<<"mapping "<<line<<'\n';
    }
    std::cout<<"DYNAMIC_QUANTIZER_NPU_PASS checks=24 cuda_execution=not_run training_backend=false\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr<<"DYNAMIC_QUANTIZER_FAIL: "<<e.what()<<'\n'; return 1;
  }
}
