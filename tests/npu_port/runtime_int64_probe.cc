// Copyright 2026, XGBoost Contributors
// Standalone runtime/int64 capability probe, not an XGBoost training backend.
// Host checks and argument guards never initialize ACL or touch a device.
#include "../../src/common/npu/acl_api.h"
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_scatter_add.h>
#include <aclnnop/aclnn_sub.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace npu = xgboost::common::npu;
namespace {
void Require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
void Passed(char const* name) {
  std::cout << "{\"test\":\"" << name << "\",\"status\":\"passed\"}\n";
}
template <typename T> constexpr aclDataType DType() {
  if constexpr (std::is_same_v<T, int64_t>) return ACL_INT64;
  else if constexpr (std::is_same_v<T, float>) return ACL_FLOAT;
  else static_assert(std::is_same_v<T, void>, "unsupported probe type");
}
template <typename T> struct Tensor {
  npu::Buffer storage;
  aclTensor* descriptor{nullptr};
  int64_t size;
  int* cleanup_failures;
  Tensor(npu::Runtime const& runtime, int64_t count, int* errors,
         std::vector<T> const* host = nullptr)
      : storage{runtime.AllocateElements<T>(CheckedCount(count))}, size{count},
        cleanup_failures{errors} {
    if (host) {
      Require(host->size() == static_cast<std::size_t>(count), "tensor input shape mismatch");
      storage.Upload(host->data(), storage.Size());
    } else {
      storage.Zero();
    }
    int64_t stride = 1;
    descriptor = aclCreateTensor(&size, 1, DType<T>(), &stride, 0, ACL_FORMAT_ND, &size, 1,
                                 storage.Data());
    Require(descriptor, "aclCreateTensor failed");
  }
  static std::size_t CheckedCount(int64_t n) {
    Require(n > 0, "probe requires nonempty tensors");
    return static_cast<std::size_t>(n);
  }
  ~Tensor() {
    auto status = aclDestroyTensor(descriptor);
    if (status != ACL_SUCCESS) {
      ++*cleanup_failures;
      std::cerr << "aclDestroyTensor status=" << status << '\n';
    }
  }
  Tensor(Tensor const&) = delete;
  Tensor& operator=(Tensor const&) = delete;
  std::vector<T> Read() const {
    std::vector<T> host(size);
    storage.Download(host.data(), storage.Size());
    return host;
  }
};
struct Ops {
  npu::Runtime const& runtime;
  template <typename Prepare, typename Execute> void Run(Prepare prepare, Execute execute) const {
    auto stream = static_cast<aclrtStream>(runtime.Stream());
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    npu::CheckAcl(prepare(&bytes, &executor), "aclnn GetWorkspaceSize");
    Require(bytes <= std::numeric_limits<std::size_t>::max(), "workspace byte count overflow");
    auto workspace = runtime.Allocate(static_cast<std::size_t>(bytes));
    npu::CheckAcl(execute(workspace.Data(), bytes, executor, stream), "aclnn execute");
    runtime.Synchronize();
    workspace.Release();
  }
  void Histogram(Tensor<int64_t> const& zero, Tensor<int64_t> const& bins,
                 Tensor<int64_t> const& values, Tensor<int64_t> const& out) const {
    Run([&](uint64_t* n, aclOpExecutor** e) {
      return aclnnScatterAddGetWorkspaceSize(zero.descriptor, 0, bins.descriptor,
                                             values.descriptor, out.descriptor, n, e);
    }, aclnnScatterAdd);
  }
  void Prefix(Tensor<int64_t> const& values, Tensor<int64_t> const& out) const {
    Run([&](uint64_t* n, aclOpExecutor** e) {
      return aclnnCumsumGetWorkspaceSize(values.descriptor, 0, ACL_INT64, out.descriptor, n, e);
    }, aclnnCumsum);
  }
  void Sub(Tensor<int64_t> const& a, Tensor<int64_t> const& b,
           Tensor<int64_t> const& out) const {
    int64_t one = 1;
    auto scalar = aclCreateScalar(&one, ACL_INT64);
    Require(scalar, "aclCreateScalar failed");
    try {
      Run([&](uint64_t* n, aclOpExecutor** e) {
        return aclnnSubGetWorkspaceSize(a.descriptor, b.descriptor, scalar, out.descriptor, n, e);
      }, aclnnSub);
    } catch (...) {
      aclDestroyScalar(scalar);
      throw;
    }
    npu::CheckAcl(aclDestroyScalar(scalar), "aclDestroyScalar");
  }
  void Cast(Tensor<float> const& a, Tensor<int64_t> const& out) const {
    Run([&](uint64_t* n, aclOpExecutor** e) {
      return aclnnCastGetWorkspaceSize(a.descriptor, ACL_INT64, out.descriptor, n, e);
    }, aclnnCast);
  }
};
void Equal(char const* name, std::vector<int64_t> const& actual,
           std::vector<int64_t> const& expected) {
  Require(actual.size() == expected.size(), "integer fixture shape mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      throw std::runtime_error(std::string(name) + " mismatch at " + std::to_string(i) +
                               " actual=" + std::to_string(actual[i]) +
                               " expected=" + std::to_string(expected[i]));
    }
  }
  Passed(name);
}
std::vector<int64_t> HistogramReference(std::vector<int64_t> const& bins,
                                      std::vector<int64_t> const& values, std::size_t count) {
  Require(bins.size() == values.size(), "fixture row mismatch");
  std::vector<int64_t> hist(count);
  for (std::size_t i = 0; i < bins.size(); ++i) {
    Require(bins[i] >= 0 && static_cast<std::size_t>(bins[i]) < count, "fixture bin out of range");
    // The supplied fixtures have bounded sums; no intentional int64 overflow.
    auto& sum = hist[bins[i]];
    auto v = values[i];
    Require((v >= 0 && sum <= INT64_MAX - v) || (v < 0 && sum >= INT64_MIN - v),
            "fixture int64 overflow");
    sum += v;
  }
  return hist;
}
std::vector<int64_t> FixedBins() { return {0, 1, 1, 2, 4, 0, 2, 4}; }
std::vector<int64_t> FixedChannel(bool hessian) {
  int64_t scale = INT64_C(1) << 58;
  if (hessian) return {scale, scale, 2 * scale, 0, scale, 3 * scale, 2 * scale, 0};
  return {-2 * scale, scale, 3 * scale, 0, -scale, 2 * scale, 4 * scale, 0};
}
void HostFixtureChecks() {
  int64_t scale = INT64_C(1) << 58;
  auto bins = FixedBins();
  Equal("host_fixed_gradient_histogram", HistogramReference(bins, FixedChannel(false), 5),
        {0, 4 * scale, 4 * scale, 0, -scale});
  Equal("host_fixed_hessian_histogram", HistogramReference(bins, FixedChannel(true), 5),
        {4 * scale, 3 * scale, 2 * scale, 0, scale});
  Equal("host_unit_residual_above_2pow53",
        HistogramReference({0, 0}, {(INT64_C(1) << 57) + 3, -(INT64_C(1) << 57) + 1}, 2),
        {4, 0});
  bool rejected = false;
  try { HistogramReference({0, 0}, {INT64_MAX, 1}, 1); }
  catch (std::exception const&) { rejected = true; }
  Require(rejected, "overflowing host oracle accepted");
  Passed("host_reference_overflow_guard");
}
void DeviceChecks(int ordinal) {
  int cleanup_failures = 0;
  auto api = std::make_shared<npu::Api>(*npu::MakeAclApi());
  api->cleanup_error = [&](std::string const& error) {
    ++cleanup_failures;
    std::cerr << "NPU cleanup failure: " << error << '\n';
  };
  {
    npu::Runtime runtime{api, ordinal};
    Ops ops{runtime};
    {
      std::vector<int64_t> source{0, -1, (INT64_C(1) << 57) + 1, -(INT64_C(1) << 57) + 3};
      auto buffer = runtime.AllocateElements<int64_t>(source.size());
      buffer.Upload(source.data(), buffer.Size());
      std::vector<int64_t> actual(source.size());
      buffer.Download(actual.data(), buffer.Size());
      Equal("npu_runtime_int64_roundtrip", actual, source);
      buffer.Zero();
      buffer.Download(actual.data(), buffer.Size());
      Equal("npu_runtime_zero", actual, std::vector<int64_t>(source.size(), 0));
    }
    // Same fixed scale as cuda_contract.json. Missing rows are mapped to a
    // separate diagnostic bucket (4), never an ordinary feature bin. This only
    // checks int64 primitives; production masking/layout remain a separate task.
    auto bins = FixedBins();
    auto gradient = FixedChannel(false), hessian = FixedChannel(true);
    for (bool is_hessian : {false, true}) {
      auto const& values = is_hessian ? hessian : gradient;
      Tensor<int64_t> indices{runtime, 8, &cleanup_failures, &bins};
      Tensor<int64_t> input{runtime, 8, &cleanup_failures, &values};
      Tensor<int64_t> zero{runtime, 5, &cleanup_failures}, hist{runtime, 5, &cleanup_failures};
      auto expected = HistogramReference(bins, values, 5);
      ops.Histogram(zero, indices, input, hist);
      Equal(is_hessian ? "npu_int64_histogram_hessian" : "npu_int64_histogram_gradient",
            hist.Read(), expected);
      Tensor<int64_t> prefix{runtime, 5, &cleanup_failures};
      ops.Prefix(hist, prefix);
      std::partial_sum(expected.begin(), expected.end(), expected.begin());
      Equal(is_hessian ? "npu_int64_prefix_hessian" : "npu_int64_prefix_gradient",
            prefix.Read(), expected);
      Tensor<int64_t> exclusive{runtime, 5, &cleanup_failures};
      ops.Sub(prefix, hist, exclusive);
      expected = {0, expected[0], expected[1], expected[2], expected[3]};
      Equal(is_hessian ? "npu_int64_exclusive_hessian" : "npu_int64_exclusive_gradient",
            exclusive.Read(), expected);
      auto known = expected.back();
      auto root = std::accumulate(values.begin(), values.end(), INT64_C(0));
      // Small diagnostic summaries may be downloaded; this host assertion does
      // not claim a device missing-statistics component has been implemented.
      Require(root - known == values[4] + values[7], "fixed fixture missing residual");
    }
    {
      // Many colliding contributions, empty bins, and bounded partial sums.
      // Even the largest all-positive subtotal is below 2^62.
      std::vector<int64_t> indices(8192, 0), values(8192, 1);
      for (std::size_t i = 0; i < values.size(); i += 4) {
        values[i] = (INT64_C(1) << 50) + 1;
        values[i + 1] = -(INT64_C(1) << 50);
      }
      auto expected = HistogramReference(indices, values, 3);
      Tensor<int64_t> index{runtime, 8192, &cleanup_failures, &indices};
      Tensor<int64_t> input{runtime, 8192, &cleanup_failures, &values};
      Tensor<int64_t> zero{runtime, 3, &cleanup_failures}, out{runtime, 3, &cleanup_failures};
      for (int repeat = 0; repeat < 4; ++repeat) {
        ops.Histogram(zero, index, input, out);
        auto actual = out.Read();
        Require(actual == expected, "int64 collision precision/repeatability mismatch");
      }
      Passed("npu_int64_collision_precision_repeated");
    }
    {
      // Nonrepresentable float64 inputs expose any lossy intermediate conversion.
      std::vector<int64_t> bins{0, 0};
      std::vector<int64_t> values{(INT64_C(1) << 57) + 3, -(INT64_C(1) << 57) + 1};
      Tensor<int64_t> index{runtime, 2, &cleanup_failures, &bins};
      Tensor<int64_t> input{runtime, 2, &cleanup_failures, &values};
      Tensor<int64_t> zero{runtime, 2, &cleanup_failures}, out{runtime, 2, &cleanup_failures};
      ops.Histogram(zero, index, input, out);
      Equal("npu_int64_unit_residual_above_2pow53", out.Read(), {4, 0});
    }
    {
      std::vector<int64_t> parent{(INT64_C(1) << 60) + 3, -17, 0, 100};
      std::vector<int64_t> child{INT64_C(1) << 60, -19, 0, 40};
      Tensor<int64_t> a{runtime, 4, &cleanup_failures, &parent};
      Tensor<int64_t> b{runtime, 4, &cleanup_failures, &child}, out{runtime, 4, &cleanup_failures};
      ops.Sub(a, b, out);
      Equal("npu_int64_sibling_subtraction", out.Read(), {3, 2, 0, 60});
    }
    {
      std::vector<float> values{-3.75f, -0.75f, 0, 0.75f, 3.75f, 1048576};
      Tensor<float> input{runtime, 6, &cleanup_failures, &values};
      Tensor<int64_t> out{runtime, 6, &cleanup_failures};
      ops.Cast(input, out);
      Equal("npu_float_to_int64_truncation", out.Read(), {-3, 0, 0, 0, 3, 1048576});
    }
    Require(runtime.LiveBytes() == 0, "runtime allocation accounting is nonzero after tests");
    Passed("npu_runtime_allocations_released");
  }
  Require(cleanup_failures == 0, "NPU resource cleanup failed");
  Passed("npu_runtime_cleanup");
}
void Provenance() {
  for (auto symbol : {"aclInit", "aclnnScatterAdd", "aclnnCumsum", "aclnnCast"}) {
    Dl_info info{};
    Require(dladdr(dlsym(RTLD_DEFAULT, symbol), &info) && info.dli_fname,
            "cannot resolve runtime/operator provenance");
    std::cout << "library " << symbol << ' ' << info.dli_fname << '\n';
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    bool host = argc == 2 && std::string(argv[1]) == "--host-api-only";
    bool device = argc == 4 && std::string(argv[1]) == "--device-test" &&
                  std::string(argv[3]) == "--confirm-device-idle";
    if (!host && !device) {
      std::cerr << "Usage: runtime_int64_probe --host-api-only\n"
                   "       runtime_int64_probe --device-test DEVICE --confirm-device-idle\n"
                   "Coordinate exclusive device use before acknowledging idle status.\n";
      return 2;
    }
    int ordinal = 0;
    if (device) {
      std::size_t consumed = 0;
      ordinal = std::stoi(argv[2], &consumed);
      Require(consumed == std::string(argv[2]).size() && ordinal >= 0,
              "invalid device ordinal");
    }
    Provenance();
    npu::MakeAclApi()->Validate();  // Creates callbacks only; calls no ACL API.
    HostFixtureChecks();
    if (host) {
      std::cout << "RUNTIME_ADAPTER_HOST_PASS acl_initialization=not_run device_execution=not_run\n";
      return 0;
    }
    npu::CheckAcl(aclInit(nullptr), "aclInit");
    try {
      DeviceChecks(ordinal);
    } catch (...) {
      auto status = aclFinalize();
      if (status != ACL_SUCCESS) std::cerr << "aclFinalize after failure status=" << status << '\n';
      throw;
    }
    npu::CheckAcl(aclFinalize(), "aclFinalize");
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
      if (line.find("/Ascend/") != std::string::npos) std::cout << "mapping " << line << '\n';
    }
    std::cout << "NPU_RUNTIME_INT64_PASS standalone_primitives_only=true training_backend=false\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "RUNTIME_PROBE_FAIL: " << error.what() << '\n';
    return 1;
  }
}
