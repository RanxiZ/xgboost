// Copyright 2026, XGBoost Contributors
// Standalone test of the production ABI, not a separately compiled ASC kernel.
#include "../../src/common/npu/acl_api.h"
#include "../../src/common/npu/native_training_abi.h"

#include <cuda/std/random>
#include <cuda/std/__cccl/version.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace npu = xgboost::common::npu;
using SamplerAbi = void (*)(void*, void const*, void const*, void*, void*,
                           std::uint32_t, std::uint64_t, float);
static_assert(std::is_same_v<decltype(&XgbNpuNativeSampleRows), SamplerAbi>);
static_assert(CCCL_VERSION == 3003003, "Oracle must use actual CCCL 3.3.3 headers");
static_assert(cuda::std::philox4x64::word_size == 64 &&
              cuda::std::philox4x64::word_count == 4 &&
              cuda::std::philox4x64::round_count == 10);
static_assert(sizeof(float) == sizeof(std::uint32_t) &&
              std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24);

namespace {
constexpr std::size_t kGuard = 8;  // 32-byte-aligned payload, with guards on both ends.
constexpr std::array<std::uint32_t, 20> kPayloads{
    0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U,
    0x007fffffU, 0x807fffffU, 0x00800000U, 0x80800000U,
    0x7f800000U, 0xff800000U, 0x7fc12345U, 0xffc54321U,
    0x7f812345U, 0xff854321U, 0x7f7fffffU, 0xff7fffffU,
    0x3f800000U, 0xbf800000U, 0x3eaaaaabU, 0xbeaaaaabU};
constexpr std::size_t kExpectedCases = 573;

void Require(bool ok, std::string const& message) {
  if (!ok) throw std::runtime_error(message);
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
// This intentionally does NOT include or call production cuda_row_sampler.h.
float OracleUniform(std::uint64_t source_seed, std::uint64_t row) {
  auto seed = static_cast<std::uint32_t>(source_seed);  // CUDA RandomWeight truncation.
  cuda::std::philox4x64 rng(seed);
  rng.discard(row);
  cuda::std::uniform_real_distribution<float> distribution;
  return distribution(rng);
}
bool OracleKeep(std::uint64_t seed, std::uint64_t row, float probability) {
  return probability >= 1.0f ||
         (probability > 0.0f && OracleUniform(seed, row) <= probability);
}
struct Case {
  std::string name;
  std::uint32_t count;
  std::uint64_t seed;
  float probability;
};
std::vector<Case> Cases() {
  std::vector<Case> cases;
  auto add = [&](char const* group, std::uint32_t count, std::uint64_t seed, float p) {
    cases.push_back({std::string(group) + "_" + std::to_string(cases.size()), count, seed, p});
  };
  // Philox's four lanes, 128-thread blocks, the 16-block cap and subsequent
  // grid-stride iterations, including partial last blocks/iterations.
  for (auto count : {0U, 1U, 2U, 3U, 4U, 5U, 31U, 32U, 33U, 127U, 128U, 129U,
                     2047U, 2048U, 2049U, 4095U, 4096U, 4097U, 8193U}) {
    for (auto seed : {UINT64_C(0), UINT64_C(1), UINT64_C(20111115), UINT64_C(0xffffffff),
                      UINT64_C(0x100000000), UINT64_C(0x100000001),
                      UINT64_C(0x12345678abcdef01), UINT64_MAX}) {
      for (float p : {0.8789f, 0.75f, 1.0f}) add("geometry_seed", count, seed, p);
    }
  }
  auto inf = std::numeric_limits<float>::infinity();
  for (auto count : {1U, 129U, 4097U}) {
    for (float p : {-inf, -1.0f, -0.0f, 0.0f, std::numeric_limits<float>::denorm_min(),
                    std::numeric_limits<float>::min(), std::nextafter(1.0f, 0.0f), 1.0f,
                    std::nextafter(1.0f, 2.0f), 2.0f, inf,
                    std::numeric_limits<float>::quiet_NaN()}) {
      add("probability_guard", count, UINT64_C(0x12345678abcdef01), p);
    }
  }
  // Equality and its adjacent representable floats expose <= versus < and
  // integer-threshold/truncation mistakes, including beyond the 2048-row cap.
  for (auto seed : {UINT64_C(0), UINT64_C(20111115), UINT64_C(0x12345678abcdef01)}) {
    for (auto row : {0U, 1U, 2U, 3U, 127U, 128U, 2047U, 2048U, 4096U}) {
      auto weight = OracleUniform(seed, row);
      Require(weight > 0.0f && weight < 1.0f, "boundary fixture must have interior weight");
      for (float p : {std::nextafter(weight, 0.0f), weight, std::nextafter(weight, 1.0f)}) {
        add("inclusive_boundary", row + 1, seed, p);
      }
    }
  }
  Require(cases.size() == kExpectedCases, "case count changed; update checker expectation");
  return cases;
}
void Compare(std::vector<std::uint32_t> const& actual,
             std::vector<std::uint32_t> const& expected, Case const& c, char const* channel) {
  Require(actual.size() == expected.size(), "comparison size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "case=" << c.name << " nrows=" << c.count << " seed=" << c.seed
                << " truncated_seed=" << static_cast<std::uint32_t>(c.seed)
                << " probability_bits=0x" << std::hex << Bits(c.probability) << std::dec
                << " channel=" << channel << " guarded_index=" << i
                << " expected=0x" << std::hex << expected[i] << " actual=0x" << actual[i]
                << std::dec << '\n';
      throw std::runtime_error("bitwise output/input/canary mismatch");
    }
  }
}
void RunCase(npu::Runtime const& runtime, Case const& c) {
  auto live = runtime.LiveBytes();
  {
    auto size = static_cast<std::size_t>(c.count) + 2 * kGuard;
    std::vector<std::uint32_t> gradient(size, 0xcafebabeU), hessian(size, 0xdeadbeefU);
    std::vector<std::uint32_t> expected_g(size, 0xababababU), expected_h(size, 0xcdcdcdcdU);
    for (std::size_t row = 0; row < c.count; ++row) {
      auto index = kGuard + row;
      gradient[index] = kPayloads[row % kPayloads.size()];
      hessian[index] = kPayloads[(row * 7 + 11) % kPayloads.size()];
      bool keep = OracleKeep(c.seed, row, c.probability);
      expected_g[index] = keep ? gradient[index] : 0U;
      expected_h[index] = keep ? hessian[index] : 0U;
    }
    auto g = runtime.AllocateElements<std::uint32_t>(size);
    auto h = runtime.AllocateElements<std::uint32_t>(size);
    auto out_g = runtime.AllocateElements<std::uint32_t>(size);
    auto out_h = runtime.AllocateElements<std::uint32_t>(size);
    auto bytes = size * sizeof(std::uint32_t);
    g.Upload(gradient.data(), bytes); h.Upload(hessian.data(), bytes);
    std::vector<std::uint32_t> poison_g(size, 0xababababU), poison_h(size, 0xcdcdcdcdU);
    // Repeat with poisoned outputs: determinism and every-write behavior are
    // checked without allowing old successful output to hide an omitted store.
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      out_g.Upload(poison_g.data(), bytes); out_h.Upload(poison_h.data(), bytes);
      auto payload = [](npu::Buffer const& b) {
        return static_cast<std::uint32_t*>(b.Data()) + kGuard;
      };
      XgbNpuNativeSampleRows(runtime.Stream(), payload(g), payload(h), payload(out_g),
                            payload(out_h), c.count, c.seed, c.probability);
      runtime.Synchronize();
      std::vector<std::uint32_t> actual(size);
      out_g.Download(actual.data(), bytes); Compare(actual, expected_g, c, "gradient_output");
      out_h.Download(actual.data(), bytes); Compare(actual, expected_h, c, "hessian_output");
      g.Download(actual.data(), bytes); Compare(actual, gradient, c, "gradient_unchanged");
      h.Download(actual.data(), bytes); Compare(actual, hessian, c, "hessian_unchanged");
    }
  }
  Require(runtime.LiveBytes() == live, "case leaked live device storage");
  std::cout << "{\"test\":\"" << c.name << "\",\"status\":\"passed\",\"nrows\":" << c.count
            << ",\"seed\":\"" << c.seed << "\",\"probability_bits\":" << Bits(c.probability)
            << ",\"repeats\":2}\n";
}
}  // namespace

int main(int argc, char** argv) {
  bool host = argc == 2 && std::string(argv[1]) == "--host-api-only";
  bool device = argc == 3 && std::string(argv[1]) == "--run-npu" &&
                std::string(argv[2]) == "--confirm-device-idle";
  if (!host && !device) {
    std::cerr << "Usage: cuda_row_sampler_device_test --host-api-only | "
                 "--run-npu --confirm-device-idle\n";
    return 2;
  }
  try {
    npu::MakeAclApi()->Validate();
    if (host) {
      // The actual ABI's empty fast path must not dereference any pointer or
      // initialize/launch a device. The preload guard enforces the ACL boundary.
      XgbNpuNativeSampleRows(nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                            UINT64_MAX, 0.8789f);
      std::cout << "CUDA_ROW_SAMPLER_HOST_GUARD_PASS device_execution=not_run\n";
      return 0;
    }
    // The guard positive control exits here, before any device operation.
    npu::CheckAcl(aclInit(nullptr), "aclInit");
    try {
      int cleanup_errors = 0;
      auto api = std::make_shared<npu::Api>(*npu::MakeAclApi());
      api->cleanup_error = [&](std::string const& error) {
        ++cleanup_errors; std::cerr << "cleanup failure: " << error << '\n';
      };
      {
        npu::Runtime runtime{api, 0};
        for (auto const& c : Cases()) RunCase(runtime, c);
        Require(runtime.LiveBytes() == 0, "sampler leaked device storage");
        runtime.TrimCache();
        Require(runtime.CachedBytes() == 0, "sampler cache did not release");
      }
      Require(cleanup_errors == 0, "sampler cleanup failed");
    } catch (...) {
      auto status = aclFinalize();
      if (status) std::cerr << "aclFinalize cleanup status=" << status << '\n';
      throw;
    }
    npu::CheckAcl(aclFinalize(), "aclFinalize");
    std::cout << "CUDA_ROW_SAMPLER_NPU_PASS checks=" << kExpectedCases
              << " oracle=CCCL_3.3.3 cuda_execution=not_run\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "CUDA_ROW_SAMPLER_FAIL: " << error.what() << '\n';
    return 1;
  }
}