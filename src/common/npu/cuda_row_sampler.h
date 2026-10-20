/**
 * Copyright 2026, XGBoost Contributors
 *
 * Philox and float canonical generation are adapted from NVIDIA CCCL v3.3.3:
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/libcudacxx/include/cuda/std/__random/philox_engine.h
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/libcudacxx/include/cuda/std/__random/generate_canonical.h
 * https://github.com/NVIDIA/cccl/blob/v3.3.3/libcudacxx/include/cuda/std/__random/uniform_real_distribution.h
 * Part of libcu++ / the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 */
#ifndef XGBOOST_COMMON_NPU_CUDA_ROW_SAMPLER_H_
#define XGBOOST_COMMON_NPU_CUDA_ROW_SAMPLER_H_

#include <cstdint>

// Define as __aicore__ inline before including this header in NPU SIMT code.
#ifndef XGB_NPU_SAMPLER_INLINE
#define XGB_NPU_SAMPLER_INLINE inline
#endif

namespace xgboost::common::npu {
namespace cuda_row_sampler_detail {
// Upper half of a 64x64 unsigned product, using only 32-bit limbs and 64-bit
// arithmetic. No compiler-specific 128-bit integer or device intrinsic is needed.
XGB_NPU_SAMPLER_INLINE std::uint64_t MulHi64(std::uint64_t a, std::uint64_t b) {
  constexpr std::uint64_t kMask = UINT64_C(0xffffffff);
  std::uint64_t a_lo = a & kMask;
  std::uint64_t b_lo = b & kMask;
  std::uint64_t a_hi = a >> 32;
  std::uint64_t b_hi = b >> 32;
  std::uint64_t lo_lo = a_lo * b_lo;
  std::uint64_t hi_lo = a_hi * b_lo;
  std::uint64_t lo_hi = a_lo * b_hi;
  std::uint64_t middle = (lo_lo >> 32) + (hi_lo & kMask) + (lo_hi & kMask);
  return a_hi * b_hi + (hi_lo >> 32) + (lo_hi >> 32) + (middle >> 32);
}

XGB_NPU_SAMPLER_INLINE float UniformFromWord(std::uint64_t word) {
  // CCCL's generate_canonical<float, 24>(philox4x64) consumes ONE word.
  // max-min+1 wraps to zero in uint64_t; __bit_log2(0) is UINT32_MAX,
  // so __k is still 1. __rp = float(UINT64_MAX) + 1.0f = 2^64.
  // Crucially, the ENTIRE word is rounded to float before division. Neither
  // masking low bits nor truncating high bits reproduces this distribution.
  // Rounding can produce 1.0f; CCCL does not clamp it to nextafter(1, 0).
  return static_cast<float>(word) / 18446744073709551616.0f;
}
}  // namespace cuda_row_sampler_detail

// Exact specialization of cuda::std::philox4x64{seed}.discard(row); rng().
// XGBoost RandomWeight truncates its caller's seed to uint32_t; this API makes
// that truncation explicit. Both counter lookup and generation are O(1) in row.
XGB_NPU_SAMPLER_INLINE std::uint64_t CudaPhiloxRowWord(std::uint32_t seed,
                                                    std::uint64_t row) {
  // CCCL seed(): counter = {0,0,0,0}, key = {seed,0}, output index = 3.
  // discard(row) selects counter floor(row/4) and output position row%4.
  // Its counter is little-endian internally, with x[0] least significant.
  std::uint64_t s0 = row >> 2;
  std::uint64_t s1 = 0;
  std::uint64_t s2 = 0;
  std::uint64_t s3 = 0;
  std::uint64_t k0 = seed;
  std::uint64_t k1 = 0;
  constexpr std::uint64_t kMultiplier0 = UINT64_C(0xCA5A826395121157);
  constexpr std::uint64_t kMultiplier1 = UINT64_C(0xD2E7470EE14C6C93);
  constexpr std::uint64_t kRound0 = UINT64_C(0x9E3779B97F4A7C15);
  constexpr std::uint64_t kRound1 = UINT64_C(0xBB67AE8584CAA73B);
  for (int round = 0; round < 10; ++round) {
    // CCCL first permutes S to V = {S[2], S[1], S[0], S[3]}.
    std::uint64_t next0 = cuda_row_sampler_detail::MulHi64(s2, kMultiplier0) ^ k0 ^ s1;
    std::uint64_t next1 = s2 * kMultiplier0;
    std::uint64_t next2 = cuda_row_sampler_detail::MulHi64(s0, kMultiplier1) ^ k1 ^ s3;
    std::uint64_t next3 = s0 * kMultiplier1;
    s0 = next0;
    s1 = next1;
    s2 = next2;
    s3 = next3;
    // Unsigned multiplication/addition intentionally wrap modulo 2^64.
    k0 += kRound0;
    k1 += kRound1;
  }
  switch (row & UINT64_C(3)) {
    case 0: return s0;
    case 1: return s1;
    case 2: return s2;
    default: return s3;
  }
}

// Exact RandomWeight: truncated seed, discard(row), uniform_real_distribution<float>{}(rng).
XGB_NPU_SAMPLER_INLINE float CudaRowUniform(std::uint32_t seed, std::uint64_t row) {
  return cuda_row_sampler_detail::UniformFromWord(CudaPhiloxRowWord(seed, row));
}

// Match gpu_hist/sampler.cuh SamplingInfo::IsSampled, including the inclusive
// comparison and its probability guards (zero/negative/NaN reject, >=1 keeps).
XGB_NPU_SAMPLER_INLINE bool CudaKeepRow(std::uint32_t seed, std::uint64_t row,
                                     float probability) {
  return probability >= 1.0f ||
         (probability > 0.0f && CudaRowUniform(seed, row) <= probability);
}
}  // namespace xgboost::common::npu

#endif  // XGBOOST_COMMON_NPU_CUDA_ROW_SAMPLER_H_
