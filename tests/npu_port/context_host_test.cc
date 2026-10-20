// Copyright 2026, XGBoost Contributors
// Host-only context/parser contracts. This does not test NPU execution.
#include <xgboost/context.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using xgboost::Args;
using xgboost::Context;
using xgboost::DeviceOrd;

static_assert(DeviceOrd::kCPU == 0);
static_assert(DeviceOrd::kCUDA == 1);
static_assert(DeviceOrd::kSyclDefault == 2);
static_assert(DeviceOrd::kSyclCPU == 3);
static_assert(DeviceOrd::kSyclGPU == 4);
static_assert(DeviceOrd::kNPU == 5);
static_assert(sizeof(DeviceOrd) == sizeof(std::int32_t));

void Require(bool valid, std::string const& message) {
  if (!valid) {
    throw std::runtime_error(message);
  }
}

std::string Reject(std::string const& input) {
  Context ctx;
  try {
    ctx.Init(Args{{"device", input}});
  } catch (dmlc::Error const& error) {
    Require(ctx.IsCPU(), "rejected device changed the effective context: " + input);
    return error.what();
  }
  throw std::runtime_error("unexpectedly accepted device: " + input);
}
}  // namespace

int main() {
  try {
    auto npu = DeviceOrd::NPU(7);
    Require(npu.IsNPU() && !npu.IsCPU() && !npu.IsCUDA() && !npu.IsSycl(), "NPU predicates");
    Require(npu.Name() == "npu:7", "NPU device name");
    Require(DeviceOrd::CPU().Name() == "cpu", "CPU device name");
    Require(DeviceOrd::CUDA(7).Name() == "cuda:7", "CUDA device name");

    Context cpu;
    cpu.Init({{"device", "cpu"}, {"nthread", "2"}});
    Require(cpu.IsCPU() && !cpu.IsNPU() && cpu.Ordinal() == -1, "CPU context");
    Require(cpu.DispatchDevice([] { return 1; }, [] { return 2; }) == 1, "CPU dispatch");
    Require(cpu.DispatchDevice([] { return 1; }, [] { return 2; }, [] { return 3; }) == 1,
            "CPU three-way dispatch");

    std::vector<std::pair<std::string, DeviceOrd>> sycl{
        {"sycl", DeviceOrd::SyclDefault()}, {"sycl:-1", DeviceOrd::SyclDefault()},
        {"sycl:42", DeviceOrd::SyclDefault(42)}, {"sycl:cpu", DeviceOrd::SyclCPU()},
        {"sycl:cpu:-1", DeviceOrd::SyclCPU()}, {"sycl:cpu:42", DeviceOrd::SyclCPU(42)},
        {"sycl:gpu", DeviceOrd::SyclGPU()}, {"sycl:gpu:-1", DeviceOrd::SyclGPU()},
        {"sycl:gpu:42", DeviceOrd::SyclGPU(42)}};
    for (auto const& item : sycl) {
      Context ctx;
      ctx.Init({{"device", item.first}});
      Require(ctx.Device() == item.second && ctx.DeviceName() == item.second.Name(), item.first);
      Require(ctx.DispatchDevice([] { return 1; }, [] { return 2; }, [] { return 3; }) == 3,
              "SYCL dispatch: " + item.first);
      ctx.UpdateAllowUnknown(Args{{"nthread", "1"}});
      Require(ctx.Device() == item.second, "SYCL ordinal changed on unrelated update");
    }

    std::vector<std::pair<std::string, std::string>> valid_npu{
        {"npu", "npu:0"}, {"npu:0", "npu:0"}, {"npu:7", "npu:7"},
        {"npu:0007", "npu:7"},
        {"npu:" + std::to_string(std::numeric_limits<xgboost::bst_d_ordinal_t>::max()),
         "npu:" + std::to_string(std::numeric_limits<xgboost::bst_d_ordinal_t>::max())}};
    for (auto const& item : valid_npu) {
#if defined(XGBOOST_USE_NPU)
      if (item.second == "npu:0") {
        Context ctx;
        ctx.Init({{"device", item.first}});
        Require(ctx.IsNPU() && ctx.DeviceName() == "npu:0", "native NPU parser selection");
        // Parsing is host-only. Generic dispatch must still reject unported
        // callers instead of entering the CPU or CUDA branch.
        bool rejected_dispatch = false;
        try { ctx.DispatchDevice([] { return 1; }, [] { return 2; }); }
        catch (dmlc::Error const&) { rejected_dispatch = true; }
        Require(rejected_dispatch, "generic NPU dispatch silently accepted");
      } else {
        auto message = Reject(item.first);
        Require(message.find("supports device npu:0 only") != std::string::npos, message);
      }
#else
      auto message = Reject(item.first);
      Require(message.find("not compiled with NPU support (XGBOOST_USE_NPU)") != std::string::npos,
              message);
      Require(message.find("Requested device: " + item.second) != std::string::npos, message);
      Require(message.find("CPU fallback is disabled") != std::string::npos, message);
#endif
    }

    std::vector<std::string> invalid{
        "", "NPU", "NPU:0", "npu:", "npu:-1", "npu:-0", "npu:+0", "npu:0-",
        "npu:1.0", "npu:0x1", "npu: 0", "npu:0 ", " npu", "npu:0:1", "npu::0",
        "npu:gpu", "npux:0", "npu:2147483648", "npu:999999999999999999999",
        "cpu:0", "cpux:0", "cuda:-1", "gpu:-1", "cud:0", "gpu:+0", "gpu:",
        "cuda:0:1", "syclx:0", "sycl:foo:0", "sycl:cpu:", "sycl:gpu:-2",
        "sycl:gpu:0:1"};
    invalid.emplace_back("npu:" + std::to_string(
        static_cast<int>(std::numeric_limits<xgboost::bst_d_ordinal_t>::max()) + 1));
    invalid.emplace_back("npu:0\0ignored", 13);
    for (auto const& input : invalid) {
      auto message = Reject(input);
      Require(message.find("Invalid argument for `device`") != std::string::npos ||
                  message.find("Ordinal value too large") != std::string::npos,
              "incorrect malformed-device error: " + message);
    }
    std::cout << "CONTEXT_HOST_PASS: ABI, CPU dispatch, " << sycl.size()
              << " SYCL cases, " << valid_npu.size() << " fail-closed NPU cases, "
              << invalid.size() << " malformed cases; no devices used\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "CONTEXT_HOST_FAIL: " << error.what() << '\n';
    return 1;
  }
}