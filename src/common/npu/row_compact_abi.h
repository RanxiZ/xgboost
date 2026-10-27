// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_ROW_COMPACT_ABI_H_
#define XGBOOST_COMMON_NPU_ROW_COMPACT_ABI_H_
#include <cstdint>

namespace xgboost::common::npu::rowcompact {
// Fused numerical row partition (compact) primitives.
//
// Pass 1 counts the left-child members of one node, stores a one-byte left flag
// per row and rejects bins outside [-1, bins_max). Pass 2 writes BOTH children in
// ascending row order from that single left flag: the right child's position is
// derived as (row position - left prefix), so no second count/prefix pass exists.
// Warps own contiguous row segments, therefore the (block, warp, lane) order of
// pass 2 is exactly the ascending id order the CPU reference produces.
constexpr uint32_t kRowCompactThreads = 512;
constexpr uint32_t kRowCompactWarps = kRowCompactThreads / 32;
constexpr uint32_t kRowCompactBlocks = 64;
// Host-side block-count heuristic; pass 1/2 derive the row split from it.
constexpr int64_t kRowCompactRowsPerBlockTarget = 1024;
// Control word slots. Slot 0 is unused; slot 1 carries the caller's generation
// counter when a bin was rejected, so the control word never needs clearing.
constexpr uint32_t kRowCompactControlViolation = 1;
constexpr uint32_t kRowCompactControlWords = 8;
}  // namespace xgboost::common::npu::rowcompact

// flags: row_count bytes (1 = left child). warp_counts: blocks * kRowCompactWarps
// int32 segment counts, each word owned by exactly one warp. block_counts: a
// reserved blocks-int32 slot kept for ABI stability (no kernel writes it; pass 2
// folds the per-warp counts itself). control: 8 int32.
extern "C" void XgbNpuRowCompactCount(
    void* stream, void const* row_ids, void const* column, int64_t row_count,
    int64_t bins_max, int64_t candidate, uint32_t missing_left, uint32_t blocks,
    void* flags, void* warp_counts, void* block_counts, void* control, uint32_t generation);

// Writes the left child at out_left and the right child at out_left + total_left,
// both in ascending row order. out_left must have room for row_count int64 values.
extern "C" void XgbNpuRowCompactWrite(
    void* stream, void const* row_ids, void const* flags, void const* warp_counts,
    void const* block_counts, int64_t row_count, uint32_t blocks, void* out_left);
#endif  // XGBOOST_COMMON_NPU_ROW_COMPACT_ABI_H_
