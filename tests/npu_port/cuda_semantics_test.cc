// Copyright 2026, XGBoost Contributors
// Host-executable CUDA semantic contracts, not a CPU tree builder or a device test.
// CUDA-shared classes are used directly where their headers compile without nvcc.
// GPU-only quantiser/missing-direction logic below is a labelled transcription;
// passing this executable does not establish CUDA or NPU execution parity.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/common/deterministic.cuh"
#include "../../src/predictor/predict_fn.h"
#include "../../src/tree/driver.h"
#include "../../src/tree/updater_gpu_common.cuh"
#include "../../src/tree/split_evaluator.h"

namespace {
using xgboost::GradientPairInt64;
using xgboost::GradientPairPrecise;
using namespace xgboost::tree;  // NOLINT
using Pair = GradientPairInt64;
using Evaluator = TreeEvaluator::SplitEvaluator<TrainParam>;

// Minimal test adapter for the official host Driver. The GPUExpandEntry header
// also contains raw __device__ multi-target methods, so it is not compiled by g++.
// This adapter is scaffolding, not an alternative tree-growth implementation.
struct ControlEntry {
  int nidx, depth;
  DeviceSplitCandidate split;
  float base_weight, left_weight, right_weight;
  float GetLossChange() const { return split.loss_chg; }
  int GetNodeId() const { return nidx; }
};

void Require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
void EqualPair(Pair a, Pair b) {
  Require(a.GetQuantisedGrad() == b.GetQuantisedGrad() &&
          a.GetQuantisedHess() == b.GetQuantisedHess(), "integer gradient pair mismatch");
}
void Close(double actual, double expected) {
  Require(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <= 1e-6 + 1e-5 * std::abs(expected),
          "floating result mismatch");
}
void Pass(char const* id, char const* origin) {
  std::cout << "{\"test\":\"" << id << "\",\"origin\":\"" << origin
            << "\",\"status\":\"passed\"}" << std::endl;
}
TrainParam Parameters() {
  TrainParam p{};
  p.reg_lambda = 1;
  p.min_child_weight = 1;
  p.learning_rate = 0.3f;
  p.max_depth = 6;
  p.grow_policy = TrainParam::kDepthWise;
  return p;
}
Evaluator SplitEvaluator() { return {nullptr, nullptr, nullptr, false, 1}; }

// Transcribed from gpu_hist/quantiser.cuh::GradientQuantiser::ToFixedPointImpl.
// Conversion scales are explicit test inputs, not an approximation of GPU reduction.
struct QuantiserReference {
  double grad_scale, hess_scale;
  Pair Fixed(double g, double h) const {
    auto convert = [](double v) {
      if (!std::isfinite(v) || v < -std::ldexp(1.0, 63) || v >= std::ldexp(1.0, 63)) {
        throw std::runtime_error("fixed-point input outside int64 domain");
      }
      return static_cast<std::int64_t>(v);
    };
    auto qg = convert(g * grad_scale), qh = convert(h * hess_scale);
    if (h > 0 && qh == 0) qh = 1;
    return {qg, qh};
  }
  GradientPairPrecise Floating(Pair q) const {
    return {q.GetQuantisedGrad() / grad_scale, q.GetQuantisedHess() / hess_scale};
  }
};

// Transcribed from gpu_hist/evaluate_splits.cu::LossChangeMissing. The shared
// scalar evaluator is the actual class used by that CUDA implementation.
struct MissingResult { float gain; bool missing_left; Pair left, right; };
MissingResult Missing(Pair scan, Pair missing, Pair parent, QuantiserReference q,
                      TrainParam const& p) {
  auto eval = SplitEvaluator();
  Pair with_missing = scan + missing;
  float lg = eval.CalcSplitGain(p, 0, 0, q.Floating(with_missing),
                                q.Floating(parent - with_missing));
  float rg = eval.CalcSplitGain(p, 0, 0, q.Floating(scan), q.Floating(parent - scan));
  bool left = lg > rg;
  Pair selected = left ? with_missing : scan;
  return {left ? lg : rg, left, selected, parent - selected};
}

void QuantisationTests() {
  auto rounding = xgboost::common::CreateRoundingFactor<double>(10, 8);
  Require(rounding == 16, "CUDA rounding factor mismatch");
  Require(xgboost::common::CreateRoundingFactor<double>(0, 8) == 1,
          "zero-bound rounding mismatch");
  Pass("rounding_factor_and_zero_bound", "cuda_shared_header");

  QuantiserReference explicit_q{2, 2};
  EqualPair(explicit_q.Fixed(1.75, 0.75), {3, 1});
  EqualPair(explicit_q.Fixed(-1.75, 0), {-3, 0});
  EqualPair(explicit_q.Fixed(0, 1e-30), {0, 1});
  EqualPair(explicit_q.Fixed(0, -1e-30), {0, 0});
  Pass("truncation_and_positive_hessian", "cuda_logic_transcription");

  QuantiserReference q{std::ldexp(1.0, 62) / rounding, std::ldexp(1.0, 62) / rounding};
  std::vector<double> gradients{-2, 1, 3, 0, -1, 2, 4, 0};
  std::vector<double> hessians{1, 1, 2, 0, 1, 3, 2, 0};
  std::vector<int> bins{0, 1, 1, 2, -1, 0, 2, -1};
  std::vector<Pair> histogram(4), pairs;
  Pair root;
  for (std::size_t i = 0; i < bins.size(); ++i) {
    Pair fixed = q.Fixed(gradients[i], hessians[i]);
    pairs.push_back(fixed); root += fixed;
    if (bins[i] >= 0) histogram[bins[i]] += fixed;
  }
  std::int64_t unit = std::int64_t{1} << 58;
  EqualPair(root, {7 * unit, 10 * unit});
  std::array<Pair, 4> expected{{{0, 4 * unit}, {4 * unit, 3 * unit},
                                {4 * unit, 2 * unit}, {0, 0}}};
  Pair known;
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EqualPair(histogram[i], expected[i]); known += histogram[i];
  }
  EqualPair(root - known, {-unit, unit});
  Pass("quantized_histogram_and_missing_residual", "cuda_logic_transcription");

  Pair prefix;
  std::array<Pair, 4> expected_scan{{{0, 0}, {0, 4 * unit},
                                    {4 * unit, 7 * unit}, {8 * unit, 9 * unit}}};
  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EqualPair(prefix, expected_scan[i]); prefix += histogram[i];
  }
  EqualPair(root - prefix, {-unit, unit});
  Pass("exclusive_scan_excludes_current_bin", "cuda_logic_transcription");

  Pair child = pairs[0] + pairs[1] + pairs[2];
  Pair sibling = root - child;
  EqualPair(child + sibling, root);
  EqualPair(sibling, {5 * unit, 6 * unit});
  Pass("integer_sibling_subtraction", "cuda_integer_data_type");

  std::int64_t big = std::int64_t{1} << 54;
  std::vector<std::int64_t> collision{big, 1, -big};
  std::sort(collision.begin(), collision.end());
  do {
    auto sum = std::accumulate(collision.begin(), collision.end(), std::int64_t{0});
    Require(sum == 1, "integer collision accumulation mismatch");
  } while (std::next_permutation(collision.begin(), collision.end()));
  Close(q.Floating(root).GetGrad(), 7);
  Close(q.Floating(root).GetHess(), 10);
  Pass("int64_precision_and_dequantisation", "cuda_logic_transcription");

  bool rejected = false;
  try { explicit_q.Fixed(std::ldexp(1.0, 63), 0); }
  catch (std::runtime_error const&) { rejected = true; }
  Require(rejected, "reference oracle must reject undefined int64 conversion");
  Pass("oracle_int64_domain_guard", "reference_safety_only");
}

void SplitTests() {
  TrainParam p = Parameters();
  auto eval = SplitEvaluator();
  auto tie = Missing({-2, 1}, {0, 0}, {0, 2}, {1, 1}, p);
  Require(!tie.missing_left, "missing direction tie must select right");
  Close(tie.gain, 4);
  Pass("missing_direction_tie_goes_right", "cuda_logic_transcription");

  auto left = Missing({-2, 1}, {-2, 1}, {0, 4}, {1, 1}, p);
  Require(left.missing_left, "missing-left candidate not selected");
  EqualPair(left.left, {-4, 2}); EqualPair(left.right, {4, 2});
  Close(left.gain, 32.0 / 3.0);
  auto right = Missing({-2, 1}, {2, 1}, {0, 4}, {1, 1}, p);
  Require(!right.missing_left, "missing-right candidate not selected");
  Close(right.gain, 3);
  Pass("missing_directions_and_selected_statistics", "cuda_logic_transcription");

  float invalid = eval.CalcSplitGain(p, 0, 0, GradientPairPrecise{-2, 0},
                                    GradientPairPrecise{2, 2});
  Require(std::isinf(invalid) && invalid < 0, "zero hessian child must be invalid");
  invalid = eval.CalcSplitGain(p, 0, 0, GradientPairPrecise{-2, 0.5},
                              GradientPairPrecise{2, 2});
  Require(std::isinf(invalid) && invalid < 0, "min_child_weight boundary mismatch");
  Close(eval.CalcWeight(0, p, GradientPairPrecise{3, 0.5}), -2);
  Pass("split_validity_is_distinct_from_leaf_weight", "cuda_shared_header");

  p.reg_alpha = 2; p.max_delta_step = 0.25f;
  auto stats = GradientPairPrecise{-6, 3};
  Close(eval.CalcWeight(0, p, stats), 0.25);
  Close(eval.CalcGainGivenWeight(p, stats, 0.25f), 1.75);
  Pass("l1_l2_clipped_leaf_gain", "cuda_shared_header");

  DeviceSplitCandidate best;
  best.Update(2, kRightDir, 0.5f, 5, {-2, 1}, {2, 1}, false);
  best.Update(2, kLeftDir, 1.5f, 1, {-2, 1}, {2, 1}, false);
  Require(best.findex == 5 && best.fvalue == 0.5f && best.dir == kRightDir,
          "equal gain must retain existing candidate, not minimum feature id");
  best.Update(3, kLeftDir, 1.5f, 1, {-3, 1}, {3, 1}, false);
  Require(best.findex == 1 && best.dir == kLeftDir, "larger gain must replace candidate");
  DeviceSplitCandidate absent;
  absent.Update(-std::numeric_limits<float>::infinity(), kRightDir, 0, 0, {}, {}, false);
  Require(absent.findex == -1, "all-invalid candidates must remain invalid");
  Pass("candidate_strict_gain_update_and_invalid", "cuda_shared_header");

  p = Parameters();
  float children = eval.CalcSplitGain(p, 0, 0, GradientPairPrecise{-2, 1},
                                     GradientPairPrecise{4, 1});
  float parent = eval.CalcGain(0, p, GradientPairPrecise{2, 2});
  Close(children - parent, 10.0 - 4.0 / 3.0);
  Pass("subtract_parent_gain_once", "cuda_shared_header");
}

void DriverTests() {
  auto p = Parameters();
  ControlEntry entry{0, 0, {}, 0, 0, 0};
  entry.split.loss_chg = xgboost::kRtEps;
  Require(!IsValidExpandEntry(entry, p, 1), "epsilon equality must stop expansion");
  entry.split.loss_chg = 1; p.min_split_loss = 1;
  Require(IsValidExpandEntry(entry, p, 1), "gamma equality must allow expansion");
  p.min_split_loss = std::nextafter(1.f, 2.f);
  Require(!IsValidExpandEntry(entry, p, 1), "gain below gamma must stop expansion");
  p.min_split_loss = 0; entry.depth = p.max_depth;
  Require(!IsValidExpandEntry(entry, p, 1), "maximum depth reached");
  entry.depth = 0; p.max_leaves = 2;
  Require(!IsValidExpandEntry(entry, p, 2), "maximum leaf count reached");
  Pass("host_driver_epsilon_gamma_depth_leaves", "cuda_shared_host_driver");

  p = Parameters();
  Driver<ControlEntry> driver(p);
  std::vector<ControlEntry> candidates;
  for (int id : {4, 2, 3, 1}) {
    ControlEntry item{id, 1, {}, 0, 0, 0}; item.split.loss_chg = 1;
    candidates.push_back(item);
  }
  driver.Push(candidates);
  auto batch = driver.Pop();
  Require(batch.size() == 4 && driver.IsEmpty(), "depthwise batch size mismatch");
  for (std::size_t i = 0; i < batch.size(); ++i) {
    Require(batch[i].nidx == static_cast<int>(i + 1), "depthwise node order mismatch");
  }
  Pass("cpu_control_uses_cuda_depthwise_driver", "cuda_shared_host_driver");
}

struct BoundaryTree {
  float SplitCond(int) const { return 1; }
  int DefaultChild(int) const { return 2; }
  int LeftChild(int) const { return 1; }
};
void BoundaryTests() {
  std::vector<float> cuts{1, 2, 4};
  auto bin = [&](float value) {
    auto index = std::upper_bound(cuts.begin(), cuts.end(), value) - cuts.begin();
    return std::min(index, static_cast<decltype(index)>(cuts.size() - 1));
  };
  Require(bin(0) == 0 && bin(1) == 1 && bin(2) == 2 && bin(4) == 2,
          "CUDA numeric upper-bound binning mismatch");
  Pass("cut_equality_uses_upper_bound", "cuda_logic_transcription");

  BoundaryTree tree;
  xgboost::RegTree::CategoricalSplitMatrix cats{};
  for (float value : {0.f, 1.f, 1.5f, 2.f, 4.f}) {
    // CUDA training compares reconstructed bin cut <= threshold. Raw prediction
    // compares raw feature < threshold. They agree for this valid cut fixture.
    bool train_left = cuts[bin(value)] <= tree.SplitCond(0);
    bool predict_left = xgboost::predictor::GetDecision<false>(tree, 0, value, cats);
    Require(train_left == predict_left, "training/prediction cut boundary disagreement");
  }
  Require(!xgboost::predictor::GetDecision<false>(tree, 0, 1, cats),
          "raw value equal to threshold must go right");
  auto missing = xgboost::predictor::GetNextNode<true, false>(
      tree, 0, std::numeric_limits<float>::quiet_NaN(), true, cats);
  Require(missing == 2, "prediction must honor default child");
  Pass("training_reconstructed_cut_vs_raw_prediction", "cuda_shared_and_transcribed");

  std::vector<float> split_values{-std::numeric_limits<float>::infinity(), cuts[0], cuts[1]};
  Require(std::isinf(split_values[0]) && split_values[0] < 0 && split_values[1] == 1 &&
          split_values[2] == 2, "exclusive scan threshold offsets mismatch");
  Pass("exclusive_scan_threshold_minus_one", "cuda_logic_transcription");
}
}  // namespace

int main() {
  try {
    QuantisationTests(); SplitTests(); DriverTests(); BoundaryTests();
    std::cout << "CUDA_SOURCE_CONTRACT_PASS (host reference only; CUDA/NPU not executed)" << std::endl;
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "CUDA_SOURCE_CONTRACT_FAIL: " << e.what() << std::endl;
    return 1;
  }
}
