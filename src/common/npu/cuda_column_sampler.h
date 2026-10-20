/**
 * Copyright 2026, XGBoost Contributors
 *
 * The Feistel permutation and round-key distribution below are adapted from
 * NVIDIA CCCL v3.3.3 ONLY, fixed sources:
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/thrust/thrust/detail/random_bijection.h
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/thrust/thrust/random/detail/uniform_int_distribution.inl
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/thrust/thrust/random/detail/uniform_real_distribution.inl
 * Copyright (c) 2008-2025, NVIDIA Corporation. All rights reserved.
 * Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
 * Stream compaction follows the same version's generic/shuffle.inl:
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/thrust/thrust/system/detail/generic/shuffle.inl
 * Original shuffle notice: Copyright (c) 2008-20120, NVIDIA Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef XGBOOST_COMMON_NPU_CUDA_COLUMN_SAMPLER_H_
#define XGBOOST_COMMON_NPU_CUDA_COLUMN_SAMPLER_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xgboost::common::npu {
namespace cuda_column_sampler_detail {
inline void ValidateRate(float rate) {
  if (!std::isfinite(rate) || rate < 0.0f || rate > 1.0f) {
    throw std::invalid_argument("CUDA-compatible column sampling requires rates in [0, 1]");
  }
}

// Match random.cc's float multiplication and int truncation, while rejecting
// sizes for which that conversion or the subsequent CUDA resize is unsafe.
inline std::size_t RetainedFeatureCount(std::size_t size, float rate) {
  ValidateRate(rate);
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA-compatible column sampling exceeds INT_MAX features");
  }
  if (rate == 1.0f) {
    return size;
  }
  if (size == 0) {
    throw std::invalid_argument("CUDA-compatible column sampling requires nonempty features");
  }
  float product = rate * static_cast<float>(size);
  // INT_MAX can round up when represented as float. Compare in double before
  // converting, rather than comparing against static_cast<float>(INT_MAX).
  if (static_cast<double>(product) > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA-compatible retained feature count overflows int");
  }
  auto retained = static_cast<std::size_t>(std::max(1, static_cast<int>(product)));
  if (retained > size) {
    throw std::overflow_error("CUDA-compatible retained feature count exceeds input size");
  }
  return retained;
}

// Exact specialization of CCCL v3.3.3's
// thrust::uniform_int_distribution<uint32_t>{}(minstd_rand). That distribution
// uses double (largest_available_float), one engine draw, and
// uniform_real_distribution<double>(0, UINT32_MAX + 1). Its normalization is
// (draw - min) / (1 + max - min), followed by cuda::std::lerp(0, 2^32, u).
// With finite u in [0,1), that lerp is exactly u * 2^32; its other term is zero.
// This is NOT std::uniform_int_distribution, nor raw minstd_rand output.
inline std::uint32_t RoundKeyFromMinstdDraw(std::uint32_t draw) {
  if (draw < 1 || draw > 2147483646U) {
    throw std::invalid_argument("CCCL round-key mapping requires a valid minstd_rand draw");
  }
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53,
                "CCCL v3.3.3 round-key distribution requires IEEE binary64 double");
  double unit = static_cast<double>(draw - 1U);
  unit /= 1.0 + static_cast<double>(2147483646U - 1U);
  return static_cast<std::uint32_t>(unit * 4294967296.0);
}

// NVIDIA CCCL v3.3.3's 24-round generic shuffle permutation, specialized to
// column counts <= INT_MAX. The minimum domain is 8 bits (256 elements).
// All products intentionally wrap modulo 2^64, as in CCCL.
class FeistelPermutation {
  std::uint32_t left_bits_, right_bits_, left_mask_, right_mask_;
  std::array<std::uint32_t, 24> keys_;

 public:
  FeistelPermutation(std::size_t size, std::uint32_t seed) {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("CUDA-compatible Feistel domain exceeds INT_MAX features");
    }
    std::uint32_t bits = 8;
    if (size > 256) {
      bits = 0;
      for (auto remaining = size - 1; remaining != 0; remaining >>= 1) {
        ++bits;
      }
    }
    left_bits_ = bits / 2;
    right_bits_ = bits - left_bits_;
    left_mask_ = (std::uint32_t{1} << left_bits_) - 1;
    right_mask_ = (std::uint32_t{1} << right_bits_) - 1;
    // CCCL v3.3.3 thrust::default_random_engine is minstd_rand, NOT
    // minstd_rand0 or the implementation-defined std::default_random_engine.
    std::minstd_rand engine{seed};
    for (auto& key : keys_) {
      key = RoundKeyFromMinstdDraw(static_cast<std::uint32_t>(engine()));
    }
  }

  std::uint64_t DomainSize() const {
    return std::uint64_t{1} << (left_bits_ + right_bits_);
  }

  std::uint64_t operator()(std::uint64_t value) const {
    auto left = static_cast<std::uint32_t>(value >> right_bits_);
    auto right = static_cast<std::uint32_t>(value & right_mask_);
    for (auto key : keys_) {
      auto product = UINT64_C(0xD2B74407B1CE6E93) * left;
      auto hi = static_cast<std::uint32_t>(product >> 32);
      auto lo = static_cast<std::uint32_t>(product);
      lo = (lo << (right_bits_ - left_bits_)) | (right >> left_bits_);
      left = ((hi ^ key) ^ right) & left_mask_;
      right = lo & right_mask_;
    }
    return (static_cast<std::uint64_t>(left) << right_bits_) | right;
  }
};
}  // namespace cuda_column_sampler_detail

// Pure host implementation of the first `retained` outputs of CCCL's generic
// thrust::shuffle, BEFORE sorting. Stream compaction preserves permutation
// domain order; cycle walking, Fisher-Yates, and shuffling feature IDs differ.
// Sparse/noncontiguous IDs are gathered by position, never used as indices.
inline std::vector<std::uint32_t> CudaFeistelSample(std::vector<std::uint32_t> const& features,
                                                 std::size_t retained, std::uint32_t seed) {
  cuda_column_sampler_detail::FeistelPermutation permutation{features.size(), seed};
  if (retained > features.size()) {
    throw std::invalid_argument("CUDA-compatible sample count exceeds input size");
  }
  std::vector<std::uint32_t> output;
  output.reserve(retained);
  for (std::uint64_t i = 0; i < permutation.DomainSize() && output.size() < retained; ++i) {
    auto index = permutation(i);
    if (index < features.size()) {
      output.push_back(features[static_cast<std::size_t>(index)]);
    }
  }
  return output;
}

// Pass the actual learner Context::Rng() by reference, never a MakeCPU() copy.
// Like random.cu, every rate < 1 call consumes exactly one outer MT19937 draw,
// even for a singleton or a rate which retains the entire input. Rate 1 uses
// zero draws and returns the input unchanged (which production keeps sorted).
inline std::vector<std::uint32_t> CudaSampleFeatures(std::mt19937& rng,
                                                  std::vector<std::uint32_t> const& features,
                                                  float rate) {
  auto retained = cuda_column_sampler_detail::RetainedFeatureCount(features.size(), rate);
  if (rate == 1.0f) {
    return features;
  }
  auto seed = static_cast<std::uint32_t>(rng());
  auto output = CudaFeistelSample(features, retained, seed);
  std::sort(output.begin(), output.end());
  return output;
}

// Standard-library-only host storage: no CUDA/NPU runtime, Context copy, or
// common/random.* dependency. The updater must supply learner_context->Rng()
// on Init and every GetFeatureSet call, in the same node order as CUDA.
class CudaColumnSampler {
 public:
  using FeatureSet = std::vector<std::uint32_t>;
  using FeatureSetPtr = std::shared_ptr<FeatureSet>;

 private:
  FeatureSetPtr tree_;
  std::map<int, FeatureSetPtr> levels_;
  float bynode_{1.0f};
  float bylevel_{1.0f};

  static FeatureSetPtr Sample(std::mt19937& rng, FeatureSetPtr const& input, float rate) {
    if (rate == 1.0f) {
      return input;
    }
    return std::make_shared<FeatureSet>(CudaSampleFeatures(rng, *input, rate));
  }

 public:
  void Init(std::mt19937& rng, std::int64_t num_col, std::vector<float> const& weights,
            float bynode, float bylevel, float bytree) {
    if (!weights.empty()) {
      throw std::invalid_argument(
          "NPU CUDA-compatible column sampling does not implement feature_weights; "
          "weighted sampling is unsupported");
    }
    cuda_column_sampler_detail::ValidateRate(bynode);
    cuda_column_sampler_detail::ValidateRate(bylevel);
    cuda_column_sampler_detail::ValidateRate(bytree);
    if (num_col <= 0) {
      throw std::invalid_argument("CUDA-compatible column sampling requires num_col > 0");
    }
    if (num_col > std::numeric_limits<int>::max() ||
        static_cast<std::uint64_t>(num_col) > FeatureSet{}.max_size()) {
      throw std::overflow_error("CUDA-compatible column sampling column count is too large");
    }
    auto count = static_cast<std::size_t>(num_col);
    auto tree_count = cuda_column_sampler_detail::RetainedFeatureCount(count, bytree);
    auto level_count = cuda_column_sampler_detail::RetainedFeatureCount(tree_count, bylevel);
    cuda_column_sampler_detail::RetainedFeatureCount(level_count, bynode);
    auto tree = std::make_shared<FeatureSet>(count);
    std::iota(tree->begin(), tree->end(), std::uint32_t{0});
    tree = Sample(rng, tree, bytree);
    tree_ = std::move(tree);
    levels_.clear();
    bynode_ = bynode;
    bylevel_ = bylevel;
  }

  // A new level is sampled before its first node. Cached levels consume no
  // further draws; bynode < 1 always samples again, including repeated depths.
  FeatureSetPtr GetFeatureSet(std::mt19937& rng, int depth) {
    if (!tree_) {
      throw std::logic_error("CudaColumnSampler must be initialized before GetFeatureSet");
    }
    if (depth < 0) {
      throw std::invalid_argument("CUDA-compatible column sampling requires depth >= 0");
    }
    if (bylevel_ == 1.0f && bynode_ == 1.0f) {
      return tree_;
    }
    auto it = levels_.find(depth);
    if (it == levels_.end()) {
      it = levels_.emplace(depth, Sample(rng, tree_, bylevel_)).first;
    }
    return Sample(rng, it->second, bynode_);
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_CUDA_COLUMN_SAMPLER_H_
