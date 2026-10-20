// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_TESTS_NPU_PORT_ROW_PARTITION_CASES_H_
#define XGBOOST_TESTS_NPU_PORT_ROW_PARTITION_CASES_H_
#include "../../src/common/npu/row_partition_contract.h"
#include "quantizer_cases.h"
#include <algorithm>
#include <numeric>
#include <set>
namespace row_partition_test {
namespace npu = xgboost::common::npu;
using quantizer_test::Require;
struct Case {
  std::string name;
  std::vector<int64_t> bins, selected;
  int64_t bin_count;
  npu::NumericalRowSplit split;
};
struct Expected { std::vector<int64_t> left, right; };
// Test-only scalar oracle, never called by production device code.
inline Expected Oracle(Case const& c) {
  npu::ValidateRowSplit({c.bin_count}, c.split);
  for (auto b : c.bins) if (b < -1 || b >= c.bin_count) throw std::invalid_argument("fixture bin");
  Expected out; std::set<int64_t> seen;
  for (auto row : c.selected) {
    if (row < 0 || row >= static_cast<int64_t>(c.bins.size()) || !seen.insert(row).second) {
      throw std::invalid_argument("fixture row");
    }
    auto b = c.bins[row];
    bool left = b == -1 ? c.split.missing_left : b < c.split.candidate;
    (left ? out.left : out.right).push_back(row);
  }
  return out;
}
inline std::vector<Case> Cases() {
  // Fixed v3.4.1 commit 6fe8c547bdd21c73e4555d85b087d9260595d30d:
  // row_partitioner.cuh SortPositionBatch/WriteResultsFunctor define membership
  // and child count, not stable right ordering. evaluate_splits.cu Numerical
  // computes exclusive prefix at i and uses preceding cut (or -inf at i=0).
  // updater_gpu_hist.cu UpdatePositionBatch routes GetFvalue <= SplitCond;
  // ellpack_page.cuh GetFvalue reconstructs cut[bin]. These are source-derived
  // fixtures; no CUDA runtime output is claimed.
  std::vector<Case> result;
  for (int64_t i : {0, 1, 2, 3}) for (bool missing : {false, true}) {
    result.push_back({"fixed_i" + std::to_string(i) + (missing ? "_missing_left" : "_missing_right"),
        {0,1,1,2,-1,0,2,-1}, {0,1,2,3,4,5,6,7}, 4, {0,i,missing}});
  }
  result.push_back({"empty_node", {0,1,-1}, {}, 2, {0,1,false}});
  result.push_back({"all_missing_left", {-1,-1,-1}, {2,0,1}, 1, {0,0,true}});
  result.push_back({"all_missing_right", {-1,-1,-1}, {1,2,0}, 1, {0,0,false}});
  result.push_back({"single_zero_row", {0}, {0}, 1, {0,0,false}});
  result.push_back({"permuted_subset", {0,1,2,3,-1,0,2,-1}, {7,6,0,4,2}, 4, {0,2,true}});
  result.push_back({"all_left", {0,0,0}, {1,0,2}, 2, {0,1,false}});
  result.push_back({"all_right", {1,1,1}, {2,0,1}, 2, {0,1,true}});
  Case large{"scan_crosses_chunks", {}, {}, 129, {0,65,false}};
  for (int64_t i=0;i<4097;++i) { large.bins.push_back(i%13==0 ? -1 : i%129); large.selected.push_back(i); }
  std::reverse(large.selected.begin(), large.selected.end()); result.push_back(std::move(large));
  return result;
}
inline void RoutingThresholdFixture() {
  // Pinned ellpack_page.cuh SearchBin uses upper_bound with end clamped. Equal
  // raw/cut goes in the next bin. Prediction predict_fn.h uses raw < threshold,
  // whereas TRAINING uses reconstructed cut <= threshold. The distinction is
  // intentional; this component accepts only candidate-local binned routing.
  std::vector<double> cuts{1,2,4,8};
  std::vector<double> raw{-2,0.999,1,1.001,2,3.99,4,7.99,8,100};
  for (int64_t i=0;i<4;++i) for (double value : raw) {
    auto bin=std::min<int64_t>(std::upper_bound(cuts.begin(),cuts.end(),value)-cuts.begin(),3);
    auto threshold=i==0 ? -std::numeric_limits<double>::infinity() : cuts[i-1];
    Require((bin<i)==(cuts[bin]<=threshold), "binned training threshold mismatch");
    Require((bin<i)==(value<threshold), "raw prediction boundary fixture mismatch");
  }
  // Clamping means arbitrary beyond-range thresholds are NOT a candidate. The
  // final endpoint i==bins is rejected, rather than silently inventing a split.
}
}  // namespace row_partition_test
#endif  // XGBOOST_TESTS_NPU_PORT_ROW_PARTITION_CASES_H_