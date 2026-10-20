#!/usr/bin/env python3
"""Host-only regression of production prediction VF bodies; never loads ACL.

This checks arithmetic/control flow, not Ascend execution or device lifetime.
Run the actual .asc through the pinned compiler separately for frontend coverage.
"""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/common/npu/native_prediction.asc"


def vf_body(source, name):
    begin = source.index("__simt_vf__", source.index("inline void " + name) - 60)
    end = source.index('\nextern "C" __global__', begin)
    return source[begin:end]


HARNESS = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "src/common/npu/native_prediction_abi.h"
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(n)
struct { unsigned x; } threadIdx;
inline uint32_t asc_atomic_or(uint32_t* address, uint32_t value) {
  uint32_t old = *address;
  *address |= value;
  return old;
}
inline uint32_t asc_atomic_cas(uint32_t* address, uint32_t compare, uint32_t value) {
  uint32_t old = *address;
  if (old == compare) *address = value;
  return old;
}
namespace native = xgboost::common::npu::native_abi;
@VF@
int main() {
  constexpr float max = std::numeric_limits<float>::max();
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  float p[]{0.5f, 2.0f, -3.0f};
  float y[]{0.0f, 1.0f, 4.0f};
  float w[]{2.0f, 3.0f, 0.0f};
  float g[3]{}, h[3]{};
  native::Status status{};
  for (unsigned i = 0; i < 3; ++i) {
    threadIdx.x = i;
    SquareErrorVf(p, y, w, g, h, &status, 3, 4.0f, 0, 1);
  }
  assert(!status.error && g[0] == 1.0f && h[0] == 2.0f);
  assert(g[1] == 12.0f && h[1] == 12.0f && g[2] == 0.0f && h[2] == 0.0f);
  threadIdx.x = 0;
  y[0] = 1.0f;
  SquareErrorVf(p, y, nullptr, g, h, &status, 1, 0.0f, 0, 1);
  assert(!status.error && g[0] == 0.0f && h[0] == 0.0f);
  auto bad_gradient = [&](float prediction, float label, float weight, float scale) {
    status = {};
    SquareErrorVf(&prediction, &label, &weight, g, h, &status, 1, scale, 0, 1);
    assert(status.error && g[0] == 0.0f && h[0] == 0.0f);
  };
  bad_gradient(inf, 0.0f, 1.0f, 1.0f);
  bad_gradient(nan, 0.0f, 1.0f, 1.0f);
  bad_gradient(max, -max, 1.0f, 1.0f);  // subtraction overflow
  bad_gradient(max, 0.0f, 2.0f, 1.0f);  // gradient overflow
  bad_gradient(0.0f, 1.0f, max, 2.0f);  // Hessian overflow
  bad_gradient(0.0f, 0.0f, -1.0f, 1.0f);
  bad_gradient(0.0f, 0.0f, nan, 1.0f);
  bad_gradient(0.0f, 0.0f, 1.0f, inf);
  native::Node nodes[]{{1, 2, 1, 1.0f, 0.0f, 0, 1, 0},
                       {-1, -1, 0, 0.0f, 2.0f, 1, 0, 0},
                       {-1, -1, 0, 0.0f, -3.0f, 1, 0, 0}};
  native::Tree trees[]{{0, 3, 0, 0}};
  uint64_t offsets[]{0, 2, 3, 4};
  native::Entry entries[]{{0, 100.0f}, {1, 0.5f}, {1, 1.0f}, {1, nan}};
  float out[]{0.5f, 0.5f, 0.5f};
  status = {};
  for (unsigned i = 0; i < 3; ++i) {
    threadIdx.x = i;
    PredictVf(nodes, trees, offsets, entries, out, out, &status, 3, 1, 1, 0, 1, 0, 1);
  }
  assert(!status.error && out[0] == 2.5f && out[1] == -2.5f && out[2] == 2.5f);
  threadIdx.x = 0;
  // A second round adds to the resident output, not a fresh base score.
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(!status.error && out[0] == 4.5f);
  nodes[0].threshold = inf;  // valid split sentinel
  out[0] = 0.5f;
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(!status.error && out[0] == 2.5f);
  nodes[0].threshold = 1.0f;
  entries[1].feature = 2;  // feature 1 absent: default-left
  out[0] = 0.5f;
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(!status.error && out[0] == 2.5f);
  entries[1].feature = 1;
  out[0] = 4.5f;
  // Zero-tree kernels touch no node/tree data.
  PredictVf(nullptr, nullptr, offsets, entries, out, out, &status, 1, 1, 0, 0, 0, 0, 1);
  assert(!status.error && out[0] == 4.5f);
  nodes[1].leaf = max;
  out[0] = max;
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(status.error && out[0] == max);
  status = {};
  nodes[1].leaf = 2.0f;
  out[0] = inf;
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(status.error);
  status = {};
  out[0] = 0.5f;
  nodes[0].left = 0;  // cycle, bounded traversal
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(status.error && out[0] == 0.5f);
  status = {};
  trees[0].group = 1;
  PredictVf(nodes, trees, offsets, entries, out, out, &status, 1, 1, 1, 0, 1, 0, 1);
  assert(status.error);
  // Later flattened trees must contribute, not be rejected by uint32 bounds.
  native::Node many_nodes[21]{};
  native::Tree many_trees[]{{0, 7, 0, 0}, {7, 7, 0, 0}, {14, 7, 0, 0}};
  for (int tree = 0; tree < 3; ++tree) {
    many_nodes[tree * 7] = {1, 2, 1, 1.0f, 0.0f, 0, 0, 0};
    many_nodes[tree * 7 + 1] = {-1, -1, 0, 0.0f, float(tree + 1), 1, 0, 0};
    many_nodes[tree * 7 + 2] = {-1, -1, 0, 0.0f, -float(tree + 1), 1, 0, 0};
  }
  status = {};
  out[0] = 0.5f;
  PredictVf(many_nodes, many_trees, offsets, entries, out, out, &status, 1, 1, 3, 2, 3, 0, 1);
  assert(!status.error && out[0] == 3.5f);
  out[0] = 0.5f;
  PredictVf(many_nodes, many_trees, offsets, entries, out, out, &status, 1, 1, 3, 0, 3, 0, 1);
  assert(!status.error && out[0] == 6.5f);
  // CUDA scalar prediction groups leaves before adding the initial margin.
  // ((2^24 + 1) - 2^24) is zero, whereas 2^24 + (1 - 2^24) is one.
  native::Node rounding_nodes[]{
      {-1, -1, 0, 0.0f, 1.0f, 1, 0, 0},
      {-1, -1, 0, 0.0f, -16777216.0f, 1, 0, 0}};
  native::Tree rounding_trees[]{{0, 1, 0, 0}, {1, 1, 0, 0}};
  status = {};
  out[0] = 16777216.0f;
  PredictVf(rounding_nodes, rounding_trees, offsets, entries, out, out, &status,
            1, 1, 2, 0, 2, 0, 1);
  assert(!status.error && out[0] == 1.0f);
  many_trees[2].first_node = UINT32_MAX;
  out[0] = 0.5f;
  PredictVf(many_nodes, many_trees, offsets, entries, out, out, &status, 1, 1, 3, 2, 3, 0, 1);
  assert(status.error == native::kPredictTreeOverflow && out[0] == 0.5f);
  assert(status.first_error == native::kPredictTreeOverflow && status.tree == 2);
  assert(status.first_node == UINT32_MAX && status.node_count == 7);
  // Separate input/output must preserve every initial value and process each
  // row once, including the second and third 2048-row grid-stride tiles.
  for (uint32_t nrows : {129U, 2049U, 4097U}) {
    std::vector<uint64_t> empty_offsets(nrows + 1, 0);
    std::vector<float> initial(nrows), result(nrows, nan);
    for (uint32_t row = 0; row < nrows; ++row) initial[row] = float(row) / 8192.0f;
    auto snapshot = initial;
    native::Node leaf{-1, -1, 0, 0.0f, 2.0f, 1, 0, 0};
    native::Tree leaf_tree{0, 1, 0, 0};
    status = {};
    for (uint32_t block = 0; block < 16; ++block) {
      for (uint32_t lane = 0; lane < 128; ++lane) {
        threadIdx.x = lane;
        PredictVf(&leaf, &leaf_tree, empty_offsets.data(), nullptr, initial.data(),
                  result.data(), &status, nrows, 1, 1, 0, 1, block, 16);
      }
    }
    assert(!status.error && initial == snapshot);
    for (uint32_t row = 0; row < nrows; ++row) assert(result[row] == initial[row] + 2.0f);
  }
}
'''


def main():
    source = SOURCE.read_text()
    bodies = vf_body(source, "PredictVf") + vf_body(source, "SquareErrorVf")
    with tempfile.TemporaryDirectory(prefix="xgb-native-prediction-") as directory:
        path = Path(directory)
        cpp = path / "kernel_host.cc"
        cpp.write_text(HARNESS.replace("@VF@", bodies))
        executable = path / "kernel_host"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall",
                        "-Wextra", "-Werror", "-I" + str(ROOT), str(cpp), "-o", str(executable)],
                       check=True)
        subprocess.run([str(executable)], check=True)
    print("NATIVE_PREDICTION_HOST_REGRESSION_PASS (device execution not tested)")


if __name__ == "__main__":
    main()