/**
 * Copyright 2026, XGBoost Contributors
 *
 * Standalone host-only test. No XGBoost library, CUDA toolkit or NPU runtime.
 * Compile against the pinned CCCL 3.3.3 tree, for example:
 * env -u LD_PRELOAD g++ -std=c++17 -O2 -Wall -Wextra -Werror \
 *   -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP \
 *   -I/workspace/npu-evidence/cccl-3.3.3/thrust \
 *   -I/workspace/npu-evidence/cccl-3.3.3/libcudacxx/include \
 *   tests/npu_port/cuda_column_sampler_host_test.cc -o /tmp/cuda_column_sampler_host_test
 * env -u LD_PRELOAD /tmp/cuda_column_sampler_host_test
 */
#include "../../src/common/npu/cuda_column_sampler.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/version.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

static_assert(THRUST_VERSION == 300303,
              "This implementation and reference test pin CCCL v3.3.3 ONLY");
static_assert(std::is_same_v<thrust::detail::largest_available_float::type, double>);
static_assert(std::is_same_v<thrust::default_random_engine, thrust::minstd_rand>);
static_assert(thrust::minstd_rand::multiplier == 48271);
static_assert(thrust::minstd_rand::modulus == 2147483647);

namespace {
using xgboost::common::npu::CudaColumnSampler;
using xgboost::common::npu::CudaFeistelSample;
using xgboost::common::npu::CudaSampleFeatures;
namespace detail = xgboost::common::npu::cuda_column_sampler_detail;
using Features = std::vector<std::uint32_t>;
std::size_t assertions = 0;
std::size_t parity_cases = 0;
std::size_t round_key_cases = 0;

void Require(bool result, std::string const& message) {
  ++assertions;
  if (!result) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Function>
void RequireThrows(Function fn, std::string const& message) {
  bool caught = false;
  try {
    fn();
  } catch (Exception const&) {
    caught = true;
  }
  Require(caught, message);
}

void SameRng(std::mt19937 const& actual, std::mt19937 const& expected) {
  // Compare the complete state, not merely the next output or a draw counter.
  Require(actual == expected, "entire MT19937 state equality");
  std::ostringstream a, b;
  a << actual;
  b << expected;
  Require(a.str() == b.str(), "serialized entire MT19937 state equality");
}

Features ReferenceShuffle(Features features, std::uint32_t seed) {
  // CCCL's generic host adapter forms a reference from begin even for an
  // empty range, which UBSan rejects for vector's null iterator. Exercise
  // actual Thrust only for nonempty production-valid feature sets; the empty
  // port result is checked separately and is not counted as Thrust parity.
  if (features.empty()) return features;
  thrust::default_random_engine rng;
  rng.seed(seed);
  thrust::shuffle(thrust::host, features.begin(), features.end(), rng);
  return features;
}

Features ReferenceSample(std::mt19937& rng, Features const& input, float rate) {
  if (rate == 1.0f) {
    return input;
  }
  int retained = std::max(1, static_cast<int>(rate * input.size()));
  auto output = ReferenceShuffle(input, static_cast<std::uint32_t>(rng()));
  output.resize(static_cast<std::size_t>(retained));
  std::sort(output.begin(), output.end());
  return output;
}

Features MakeFeatures(std::size_t size, bool sparse) {
  Features features(size);
  for (std::size_t i = 0; i < size; ++i) {
    // Descending, sparse, and above INT_MAX. These are IDs, NOT domain indices.
    features[i] = sparse ? UINT32_MAX - static_cast<std::uint32_t>(i * 7919) : i;
  }
  return features;
}

// Exercise the actual Thrust distribution with controlled valid engine draws,
// including its endpoints, not a second copy of the port's arithmetic.
struct ControlledMinstdDraw {
  static constexpr std::uint32_t min = 1;
  static constexpr std::uint32_t max = 2147483646U;
  std::uint32_t draw;
  unsigned calls{0};
  std::uint32_t operator()() {
    ++calls;
    return draw;
  }
};

void TestRoundKeyDistribution() {
  thrust::uniform_int_distribution<std::uint32_t> distribution;
  Require(distribution.a() == 0 && distribution.b() == UINT32_MAX,
          "Thrust default uint32 distribution covers entire uint32 range");
  auto compare = [&](std::uint32_t draw) {
    ControlledMinstdDraw engine{draw};
    auto expected = distribution(engine);
    Require(detail::RoundKeyFromMinstdDraw(draw) == expected,
            "round-key mapping vs actual Thrust uniform_int_distribution<uint32_t>");
    Require(engine.calls == 1, "Thrust round key consumes exactly one minstd draw");
    ++round_key_cases;
  };
  for (std::uint32_t draw = 1; draw <= 10000; ++draw) {
    compare(draw);
    compare(2147483647U - draw);
  }
  for (auto draw : {1073741822U, 1073741823U, 1073741824U, 1073741825U}) {
    compare(draw);
  }
  std::mt19937 inputs{333};
  for (int i = 0; i < 1000000; ++i) {
    compare(1U + static_cast<std::uint32_t>(inputs()) % 2147483646U);
  }
  // Verify the standard engine, which is used only as minstd_rand (not as a
  // standard distribution), against the actual Thrust engine and key stream.
  for (std::uint32_t seed = 0; seed < 256; ++seed) {
    for (auto adjusted : {seed, UINT32_MAX - seed}) {
      std::minstd_rand actual{adjusted};
      thrust::minstd_rand expected{adjusted};
      for (int round = 0; round < 1024; ++round) {
        auto draw = static_cast<std::uint32_t>(actual());
        auto reference_draw = expected();
        Require(draw == reference_draw, "standard minstd_rand vs actual Thrust engine");
        compare(draw);
      }
    }
  }
  Require(detail::RoundKeyFromMinstdDraw(1) == 0, "minimum draw maps to zero key");
  Require(detail::RoundKeyFromMinstdDraw(2147483646U) == 4294967294U,
          "maximum draw maps to the Thrust upper endpoint (not UINT32_MAX)");
}

void TestActualThrustParity() {
  std::vector<std::size_t> sizes{0,   1,   2,   3,   4,   5,   7,   8,    9,    15,
                                16,  17,  23,  31,  32,  33,  63,  64,   65,   127,
                                128, 129, 255, 256, 257, 511, 512, 513,  1023, 1024,
                                1025, 4093};
  std::vector<std::uint32_t> seeds{0, 1, 2, 42, 2026, 2147483646U, 2147483647U,
                                   2147483648U, UINT32_MAX};
  std::vector<float> rates{1.0f, std::nextafter(1.0f, 0.0f), 0.99f, 0.75f, 0.5f,
                           0.33333334f, 0.1f, 0.01f, std::numeric_limits<float>::min(),
                           std::numeric_limits<float>::denorm_min(), 0.0f, -0.0f};
  for (auto size : sizes) {
    for (bool sparse : {false, true}) {
      auto input = MakeFeatures(size, sparse);
      for (auto seed : seeds) {
        // Check the entire permutation before sorting. A wrong permutation can
        // otherwise hide behind sorting at rate 1 or near 1.
        auto shuffled = ReferenceShuffle(input, seed);
        Require(CudaFeistelSample(input, size, seed) == shuffled,
                "full unsorted Feistel permutation vs actual thrust::shuffle");
        for (auto retained : {std::size_t{0}, size / 2, size}) {
          auto prefix = shuffled;
          prefix.resize(retained);
          Require(CudaFeistelSample(input, retained, seed) == prefix,
                  "truncated Feistel compaction vs thrust prefix");
          if (size != 0) ++parity_cases;
        }
        for (auto rate : rates) {
          std::mt19937 actual{seed}, expected{seed}, one_draw{seed};
          auto untouched = input;
          if (size == 0 && rate < 1.0f) {
            RequireThrows<std::invalid_argument>([&] { CudaSampleFeatures(actual, input, rate); },
                                                 "empty sampled set rejected");
            SameRng(actual, expected);
            continue;
          }
          auto output = CudaSampleFeatures(actual, input, rate);
          auto reference = ReferenceSample(expected, input, rate);
          Require(output == reference, "sorted sampled features vs actual thrust::shuffle");
          Require(input == untouched, "sampling leaves input unchanged");
          SameRng(actual, expected);
          if (rate < 1.0f) {
            one_draw();
            Require(std::is_sorted(output.begin(), output.end()), "sample sorted ascending");
          } else {
            Require(output == input, "rate 1 preserves even unsorted IDs");
          }
          SameRng(actual, one_draw);
          if (size != 0) ++parity_cases;
        }
      }
    }
  }
  // Exhaust small inputs within the minimum 256-element cipher domain.
  for (std::size_t size = 1; size <= 256; ++size) {
    for (std::uint32_t seed = 0; seed < 64; ++seed) {
      auto input = MakeFeatures(size, true);
      Require(CudaFeistelSample(input, size, seed) == ReferenceShuffle(input, seed),
              "exhaustive small-domain permutation parity");
      ++parity_cases;
    }
  }
}

// Independently reproduce ColumnSampler's tree -> cached level -> per-node
// control flow; every random permutation here is the real Thrust implementation.
class ReferenceColumnSampler {
  using Ptr = std::shared_ptr<Features>;
  Ptr tree_;
  std::map<int, Ptr> levels_;
  float bynode_, bylevel_;
  static Ptr Sample(std::mt19937& rng, Ptr const& input, float rate) {
    return rate == 1.0f ? input : std::make_shared<Features>(ReferenceSample(rng, *input, rate));
  }

 public:
  void Init(std::mt19937& rng, std::size_t count, float bynode, float bylevel, float bytree) {
    bynode_ = bynode;
    bylevel_ = bylevel;
    levels_.clear();
    tree_ = Sample(rng, std::make_shared<Features>(MakeFeatures(count, false)), bytree);
  }
  Ptr Get(std::mt19937& rng, int depth) {
    if (bynode_ == 1.0f && bylevel_ == 1.0f) {
      return tree_;
    }
    if (levels_.count(depth) == 0) {
      levels_[depth] = Sample(rng, tree_, bylevel_);
    }
    return Sample(rng, levels_[depth], bynode_);
  }
};

void TestSamplerOrderingAndCache() {
  for (auto count : {1, 2, 15, 16, 17, 64, 129, 257, 1025}) {
    for (float bytree : {1.0f, 0.8f, 0.01f, 0.0f}) {
      for (float bylevel : {1.0f, 0.6f, 0.0f}) {
        for (float bynode : {1.0f, 0.7f, 0.0f}) {
          for (auto seed : {0U, 42U, UINT32_MAX}) {
            std::mt19937 actual{seed}, expected{seed};
            CudaColumnSampler sampler;
            ReferenceColumnSampler reference;
            // Re-init must discard cached levels and consume only tree draws.
            for (int tree = 0; tree < 3; ++tree) {
              auto columns = static_cast<std::size_t>(count + tree);
              sampler.Init(actual, columns, {}, bynode, bylevel, bytree);
              reference.Init(expected, columns, bynode, bylevel, bytree);
              SameRng(actual, expected);
              std::map<int, CudaColumnSampler::FeatureSetPtr> seen;
              // Includes a level encountered out of depth order and revisits.
              for (int depth : {0, 0, 2, 1, 2, 3, 1, 0, 7, 7}) {
                auto result = sampler.GetFeatureSet(actual, depth);
                Require(*result == *reference.Get(expected, depth), "tree/level/node parity");
                Require(std::is_sorted(result->begin(), result->end()), "sampler sorted IDs");
                SameRng(actual, expected);
                if (seen.count(depth)) {
                  Require((result == seen[depth]) == (bynode == 1.0f),
                          "cached level identity or new node allocation");
                }
                seen[depth] = result;
                ++parity_cases;
              }
            }
          }
        }
      }
    }
  }
  std::mt19937 rng{7}, initial = rng;
  CudaColumnSampler sampler;
  sampler.Init(rng, 100, {}, 1.0f, 1.0f, 1.0f);
  auto first = sampler.GetFeatureSet(rng, 0);
  Require(first == sampler.GetFeatureSet(rng, 9), "rate 1 aliases tree across depths");
  SameRng(rng, initial);
  sampler.Init(rng, 1, {}, 0.01f, 0.01f, 0.01f);
  initial();  // tree
  SameRng(rng, initial);
  sampler.GetFeatureSet(rng, 0);
  initial();  // first level
  initial();  // first node
  SameRng(rng, initial);
  sampler.GetFeatureSet(rng, 0);
  initial();  // second node, cached level
  SameRng(rng, initial);
  sampler.GetFeatureSet(rng, 1);
  initial();  // next level
  initial();  // next node
  SameRng(rng, initial);

  // Cross multiple MT19937 state refresh boundaries and interleave unrelated
  // learner draws (for example, row sampling), rather than reseeding per tree.
  std::mt19937 learner{2026}, reference_rng{2026};
  ReferenceColumnSampler reference;
  sampler.Init(learner, 33, {}, 0.5f, 0.5f, 0.5f);
  reference.Init(reference_rng, 33, 0.5f, 0.5f, 0.5f);
  for (int node = 0; node < 1400; ++node) {
    learner.discard(node % 3);
    reference_rng.discard(node % 3);
    Require(*sampler.GetFeatureSet(learner, node % 11) ==
                *reference.Get(reference_rng, node % 11),
            "persistent learner RNG with interleaved external draws");
    SameRng(learner, reference_rng);
    ++parity_cases;
  }
}

void TestValidation() {
  std::mt19937 rng{8}, initial = rng;
  CudaColumnSampler sampler;
  RequireThrows<std::logic_error>([&] { sampler.GetFeatureSet(rng, 0); }, "uninitialized sampler");
  for (auto rate : {-0.5f, 1.1f, std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::quiet_NaN()}) {
    RequireThrows<std::invalid_argument>([&] { CudaSampleFeatures(rng, {1, 2}, rate); },
                                         "invalid pure sampling rate");
    RequireThrows<std::invalid_argument>([&] { sampler.Init(rng, 2, {}, rate, 1, 1); },
                                         "invalid bynode");
    RequireThrows<std::invalid_argument>([&] { sampler.Init(rng, 2, {}, 1, rate, 1); },
                                         "invalid bylevel");
    RequireThrows<std::invalid_argument>([&] { sampler.Init(rng, 2, {}, 1, 1, rate); },
                                         "invalid bytree");
  }
  for (auto columns : {std::int64_t{0}, std::int64_t{-1}}) {
    RequireThrows<std::invalid_argument>([&] { sampler.Init(rng, columns, {}, 1, 1, 1); },
                                         "nonpositive column count");
  }
  for (auto columns : {std::int64_t{std::numeric_limits<int>::max()} + 1,
                        std::int64_t{UINT32_MAX}, std::numeric_limits<std::int64_t>::max()}) {
    RequireThrows<std::overflow_error>([&] { sampler.Init(rng, columns, {}, 1, 1, 1); },
                                       "large column count rejected before allocation");
  }
  RequireThrows<std::invalid_argument>([&] { sampler.Init(rng, 2, {1, 1}, 1, 1, 1); },
                                       "weights rejected even for rate 1");
  RequireThrows<std::invalid_argument>([&] { sampler.Init(rng, 2, {1}, .5f, .5f, .5f); },
                                       "weights rejected even with mismatched length");
  RequireThrows<std::invalid_argument>([&] { CudaFeistelSample({1}, 2, 0); },
                                       "retained count larger than input");
  auto too_large = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  RequireThrows<std::overflow_error>([&] { detail::FeistelPermutation permutation{too_large, 0}; },
                                     "large Feistel domain rejected before shifts");
  RequireThrows<std::overflow_error>([&] { detail::RetainedFeatureCount(too_large, 1); },
                                     "oversized count rejected without allocation");
  auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
  Require(detail::RetainedFeatureCount(int_max, 1.0f) == int_max, "rate1 skips float-int cast");
  auto rate = std::nextafter(1.0f, 0.0f);
  Require(detail::RetainedFeatureCount(int_max, rate) ==
              static_cast<std::size_t>(static_cast<int>(rate * static_cast<float>(int_max))),
          "large valid count uses original float multiplication");
  RequireThrows<std::invalid_argument>([&] { detail::RoundKeyFromMinstdDraw(0); },
                                       "round-key input below engine minimum");
  RequireThrows<std::invalid_argument>([&] { detail::RoundKeyFromMinstdDraw(2147483647U); },
                                       "round-key input above engine maximum");
  for (std::size_t size : {0, 1, 16, 17, 255, 256}) {
    detail::FeistelPermutation minimum{size, 0};
    Require(minimum.DomainSize() == 256, "CCCL v3.3.3 minimum eight-bit domain");
  }
  detail::FeistelPermutation above_minimum{257, 0};
  Require(above_minimum.DomainSize() == 512, "first domain above eight bits");
  detail::FeistelPermutation permutation{int_max, 0};
  Require(permutation.DomainSize() == (UINT64_C(1) << 31), "large domain is safe uint64 shift");
  SameRng(rng, initial);
  sampler.Init(rng, 2, {}, 1, 1, 1);
  RequireThrows<std::invalid_argument>([&] { sampler.GetFeatureSet(rng, -1); }, "negative depth");
  SameRng(rng, initial);
}
}  // namespace

int main() {
  try {
    TestRoundKeyDistribution();
    TestActualThrustParity();
    TestSamplerOrderingAndCache();
    TestValidation();
    std::cout << "PASS: CCCL " << THRUST_MAJOR_VERSION << '.' << THRUST_MINOR_VERSION << '.'
              << THRUST_SUBMINOR_VERSION << ", " << parity_cases << " shuffle/sampler parity cases, "
              << round_key_cases << " round-key parity cases, " << assertions
              << " assertions; host-only, no XGBoost library or device runtime\n";
    return EXIT_SUCCESS;
  } catch (std::exception const& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
