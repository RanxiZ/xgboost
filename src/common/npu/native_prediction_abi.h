// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_NATIVE_PREDICTION_ABI_H_
#define XGBOOST_COMMON_NPU_NATIVE_PREDICTION_ABI_H_

#include <cstdint>

namespace xgboost::common::npu::native_abi {

// The host and Ascend translation units intentionally exchange only fixed-width,
// trivially-copyable records. Tree nodes are flattened on the host; the device
// never dereferences a C++ RegTree.
struct Node {
  int32_t left;
  int32_t right;
  uint32_t feature;
  float threshold;
  float leaf;
  uint8_t is_leaf;
  uint8_t default_left;
  uint16_t reserved;
};
static_assert(sizeof(Node) == 24, "native prediction node ABI");

struct Tree {
  uint32_t first_node;
  uint32_t node_count;
  uint32_t group;
  uint32_t reserved;
};
static_assert(sizeof(Tree) == 16, "native prediction tree ABI");

struct Entry {
  uint32_t feature;
  float value;
};
static_assert(sizeof(Entry) == 8, "native prediction entry ABI");

enum Error : uint32_t {
  kPredictShape = 1U << 0,
  kPredictTree = 1U << 1,
  kPredictNode = 1U << 2,
  kPredictCycle = 1U << 3,
  kPredictThreshold = 1U << 4,
  kPredictChild = 1U << 5,
  kPredictPrevious = 1U << 6,
  kPredictLeaf = 1U << 7,
  kPredictAddition = 1U << 8,
  kPredictFeature = 1U << 9,
  kSquareInput = 1U << 10,
  kSquareWeight = 1U << 11,
  kSquareDifference = 1U << 12,
  kSquareGradient = 1U << 13,
  kPredictEmptyTree = 1U << 14,
  kPredictTreeOverflow = 1U << 15,
};

// All failing lanes atomically OR error bits. Exactly one lane claims the
// diagnostic record; the host reads it only after stream synchronization.
struct Status {
  uint32_t error;
  uint32_t steps;
  uint32_t claimed;
  uint32_t first_error;
  uint32_t row;
  uint32_t tree;
  int32_t node;
  uint32_t first_node;
  uint32_t node_count;
  uint32_t group;
  Node observed_node;
  float previous;
  float value;
  float result;
  uint32_t reserved;
  uint64_t row_begin;
  uint64_t row_end;
};
static_assert(sizeof(Status) == 96, "native prediction status ABI");

extern "C" void XgbNpuNativePredict(void* stream, void const* nodes, void const* trees,
                                     void const* offsets, void const* entries,
                                     void const* input, void* output,
                                     void* status, uint32_t nrows, uint32_t ngroups, uint32_t ntree,
                                     uint32_t tree_begin, uint32_t tree_end);
extern "C" void XgbNpuNativeSquareError(void* stream, void const* predictions,
                                         void const* labels, void const* weights,
                                         void* gradient, void* hessian, void* status,
                                         uint32_t count, float scale_pos_weight);

}  // namespace xgboost::common::npu::native_abi
#endif  // XGBOOST_COMMON_NPU_NATIVE_PREDICTION_ABI_H_