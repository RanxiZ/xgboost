/**
 * Copyright 2026, XGBoost Contributors
 *
 * Standalone host-only comparison with the ACTUAL pinned CCCL 3.3.3 headers.
 * No CUDA toolkit, XGBoost library, or NPU runtime is required or used.
 * env -u LD_PRELOAD g++ -std=c++17 -O2 -Wall -Wextra -Werror \
 *   -I/workspace/npu-evidence/cccl-3.3.3/libcudacxx/include \
 *   tests/npu_port/cuda_row_sampler_host_test.cc -o /tmp/cuda_row_sampler_host_test
 * env -u LD_PRELOAD /tmp/cuda_row_sampler_host_test
 */
#include "../../src/common/npu/cuda_row_sampler.h"

#include <cuda/std/random>
#include <cuda/std/__cccl/version.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

static_assert(CCCL_VERSION == 3003003, "Reference must be CCCL v3.3.3");
static_assert(cuda::std::philox4x64::word_size == 64);
static_assert(cuda::std::philox4x64::word_count == 4);
static_assert(cuda::std::philox4x64::round_count == 10);
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24);

namespace {
using xgboost::common::npu::CudaKeepRow;
using xgboost::common::npu::CudaPhiloxRowWord;
using xgboost::common::npu::CudaRowUniform;
namespace detail = xgboost::common::npu::cuda_row_sampler_detail;
std::uint64_t assertions = 0;
std::uint64_t row_cases = 0;
std::uint64_t decision_cases = 0;
std::uint64_t canonical_cases = 0;
std::uint64_t mulhi_cases = 0;

void Require(bool result, char const* message) {
  ++assertions;
  if (!result) {
    throw std::runtime_error(message);
  }
}

std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// A valid, controlled 64-bit URBG lets the real CCCL distribution expose its
// entire-word conversion, round-to-nearest ties, endpoints, and consumption.
struct ControlledWord {
  using result_type = std::uint64_t;
  std::uint64_t word;
  unsigned calls{0};
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }
  result_type operator()() {
    ++calls;
    return word;
  }
};

void CheckCanonical(std::uint64_t word) {
  ControlledWord rng{word};
  cuda::std::uniform_real_distribution<float> distribution;
  auto expected = distribution(rng);
  Require(Bits(detail::UniformFromWord(word)) == Bits(expected),
          "canonical float bits differ from actual CCCL uniform_real_distribution<float>");
  Require(rng.calls == 1, "float distribution must consume exactly one philox4x64 word");
  ++canonical_cases;
}

void TestCanonical() {
  for (std::uint64_t word = 0; word < 65536; ++word) {
    CheckCanonical(word);
    CheckCanonical(UINT64_MAX - word);
  }
  // Boundaries of integer->float rounding at every binary exponent, including
  // the halfway point where retaining only the high 24 bits would be wrong.
  for (unsigned bit = 0; bit < 64; ++bit) {
    std::uint64_t power = UINT64_C(1) << bit;
    for (std::uint64_t delta = 0; delta < 32; ++delta) {
      CheckCanonical(power + delta);
      CheckCanonical(power - delta);  // intentional modulo-2^64 wrap
    }
    if (bit >= 24) {
      std::uint64_t half_ulp = UINT64_C(1) << (bit - 24);
      CheckCanonical(power + half_ulp - 1);
      CheckCanonical(power + half_ulp);
      CheckCanonical(power + half_ulp + 1);
      CheckCanonical(power + 3 * half_ulp - 1);
      CheckCanonical(power + 3 * half_ulp);
      CheckCanonical(power + 3 * half_ulp + 1);
    }
  }
  // Exhaustive upper-16-bit set with several distinct lower 48-bit patterns.
  for (std::uint64_t high = 0; high < 65536; ++high) {
    for (auto low : {UINT64_C(0), UINT64_C(1), UINT64_C(0x7fffffffffff),
                     UINT64_C(0x800000000000), UINT64_C(0xffffffffffff)}) {
      CheckCanonical((high << 48) | low);
    }
  }
  Require(Bits(detail::UniformFromWord(0)) == Bits(0.0f), "zero maps to zero");
  Require(Bits(detail::UniformFromWord(UINT64_MAX)) == Bits(1.0f),
          "CCCL rounds UINT64_MAX to 1.0f; there must be no endpoint clamp");
  std::uint64_t halfway_to_one = UINT64_MAX - (UINT64_C(1) << 39) + 1;
  Require(Bits(detail::UniformFromWord(halfway_to_one - 1)) == Bits(std::nextafter(1.0f, 0.0f)),
          "word just below halfway rounds below one");
  Require(Bits(detail::UniformFromWord(halfway_to_one)) == Bits(1.0f),
          "halfway word rounds to even, namely one");
}

void CheckMulHi(std::uint64_t a, std::uint64_t b) {
  auto expected = static_cast<std::uint64_t>((static_cast<__uint128_t>(a) * b) >> 64);
  Require(detail::MulHi64(a, b) == expected, "32-bit-limb mulhi differs from host uint128 oracle");
  ++mulhi_cases;
}

void TestMulHi() {
  std::uint64_t values[] = {0, 1, 2, UINT32_MAX, UINT64_C(1) << 32,
                            (UINT64_C(1) << 32) + 1, UINT64_C(0x7fffffffffffffff),
                            UINT64_C(0x8000000000000000), UINT64_MAX - 1, UINT64_MAX,
                            UINT64_C(0xCA5A826395121157), UINT64_C(0xD2E7470EE14C6C93)};
  for (auto a : values) {
    for (auto b : values) {
      CheckMulHi(a, b);
    }
  }
  for (std::uint64_t limb = 0; limb < 65536; ++limb) {
    CheckMulHi((limb << 32) | UINT32_MAX, UINT64_MAX);
    CheckMulHi(UINT64_MAX, (limb << 32) | limb);
  }
  std::mt19937_64 inputs{333};
  for (unsigned i = 0; i < 1000000; ++i) {
    auto a = inputs();
    auto b = inputs();
    CheckMulHi(a, b);
  }
}

void CheckDecision(std::uint32_t seed, std::uint64_t row, float weight, float probability) {
  // Literal SamplingInfo::IsSampled expression, with weight obtained from the
  // actual CCCL distribution, rather than a duplicate of portable arithmetic.
  bool expected = probability >= 1.0f || (probability > 0.0f && weight <= probability);
  Require(CudaKeepRow(seed, row, probability) == expected, "row keep decision differs");
  ++decision_cases;
}

void CheckRow(std::uint64_t source_seed, std::uint64_t row, bool all_probabilities = false) {
  auto seed = static_cast<std::uint32_t>(source_seed);  // RandomWeight constructor's truncation
  cuda::std::philox4x64 word_rng{seed};
  word_rng.discard(row);
  auto expected_word = word_rng();
  if (CudaPhiloxRowWord(seed, row) != expected_word) {
    std::cerr << "seed=" << source_seed << " truncated=" << seed << " row=" << row << '\n';
    Require(false, "raw word differs from actual CCCL philox4x64 discard(row)");
  }
  ++assertions;
  cuda::std::philox4x64 uniform_rng{seed};
  uniform_rng.discard(row);
  cuda::std::uniform_real_distribution<float> distribution;
  float expected_uniform = distribution(uniform_rng);
  Require(Bits(CudaRowUniform(seed, row)) == Bits(expected_uniform), "row uniform float bits differ");
  Require(uniform_rng == word_rng, "distribution consumes exactly the expected one-word RNG state");
  CheckDecision(seed, row, expected_uniform, expected_uniform);
  CheckDecision(seed, row, expected_uniform, std::nextafter(expected_uniform, 0.0f));
  CheckDecision(seed, row, expected_uniform, std::nextafter(expected_uniform, 1.0f));
  CheckDecision(seed, row, expected_uniform, 0.5f);
  if (all_probabilities) {
    for (float probability : {-std::numeric_limits<float>::infinity(), -1.0f, -0.0f, 0.0f,
                              std::numeric_limits<float>::denorm_min(),
                              std::numeric_limits<float>::min(), 0.01f, 0.1f, 0.25f,
                              0.75f, 0.9f, std::nextafter(1.0f, 0.0f), 1.0f,
                              std::nextafter(1.0f, 2.0f), 2.0f,
                              std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN()}) {
      CheckDecision(seed, row, expected_uniform, probability);
    }
  }
  ++row_cases;
}

void TestRows() {
  // Exhaustive Cartesian grids, checking all four output lanes and many blocks.
  for (std::uint32_t seed = 0; seed < 1024; ++seed) {
    for (std::uint64_t row = 0; row < 1024; ++row) {
      CheckRow(seed, row, (row & 255) == 0);
      CheckRow(UINT32_MAX - seed, row, (row & 255) == 0);
    }
  }
  for (std::uint32_t seed = 0; seed < 65536; ++seed) {
    for (std::uint64_t row = 0; row < 4; ++row) {
      CheckRow(seed, row);
      CheckRow(UINT32_MAX - seed, row);
    }
  }
  // Rows immediately before/after every power-of-two boundary, including
  // uint32 wrap, high counter bits and UINT64_MAX. No iterative discard.
  for (auto source_seed : {UINT64_C(0), UINT64_C(1), UINT64_C(20111115),
                           UINT64_C(0x7fffffff), UINT64_C(0x80000000),
                           UINT64_C(0xffffffff), UINT64_C(0x100000000),
                           UINT64_C(0x100000001), UINT64_C(0x12345678abcdef01), UINT64_MAX}) {
    for (unsigned bit = 0; bit < 64; ++bit) {
      std::uint64_t base = UINT64_C(1) << bit;
      for (std::uint64_t delta = 0; delta < 8; ++delta) {
        CheckRow(source_seed, base + delta, true);
        CheckRow(source_seed, base - delta, true);  // includes modulo wrap
      }
    }
  }
  std::mt19937_64 inputs{UINT64_C(0x333a100)};
  for (unsigned i = 0; i < 200000; ++i) {
    auto source_seed = inputs();
    auto row = inputs();
    CheckRow(source_seed, row, i % 256 == 0);
  }
  // Independent engine conformance and output ordering, not just discard().
  cuda::std::philox4x64 consecutive;
  for (std::uint64_t row = 0; row < 10000; ++row) {
    auto word = consecutive();
    Require(CudaPhiloxRowWord(20111115, row) == word, "sequential engine output order differs");
    if (row == 9999) {
      Require(word == UINT64_C(3409172418970261260), "CCCL documented 10000th value differs");
    }
  }
}
}  // namespace

int main() {
  try {
    TestMulHi();
    TestCanonical();
    TestRows();
    std::cout << "PASS CCCL " << CCCL_VERSION << " cuda::std::philox4x64 + "
              << "cuda::std::uniform_real_distribution<float> host parity\n"
              << "mulhi_cases=" << mulhi_cases << " canonical_cases=" << canonical_cases
              << " row_cases=" << row_cases << " decision_cases=" << decision_cases
              << " assertions=" << assertions << '\n'
              << "Exact raw words, float bits, inclusive decisions, seed truncation, "
              << "modulo wrap, all lanes, and O(1) offsets through UINT64_MAX verified.\n";
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
