// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_SPLIT_STATISTICS_H_
#define XGBOOST_COMMON_NPU_SPLIT_STATISTICS_H_
#include "integer_histogram.h"
#include "numerical_split_evaluator_abi.h"

namespace xgboost::common::npu {
// Integer portion of gpu_hist/evaluate_splits.cu::EvaluateSplitAgent::Numerical.
// One candidate BEFORE each bin: index 0 has an empty non-missing left side;
// index i>0 uses cut[i-1]. There is no candidate after the final bin.
// This component deliberately does NOT compute gains or choose a split.
class NumericalSplitStatistics {
  struct Channel {
    Tensor missing;
    Tensor left_missing_right, right_missing_right;
    Tensor left_missing_left, right_missing_left;
  };
 public:
  class Result {
    friend class NumericalSplitStatistics;
    friend class PreciseSplitScorer;
    friend class NumericalSplitEvaluator;
    friend class SplitStatisticsTestAccess;
    Channel gradient_, hessian_;
    Result(Channel&& gradient, Channel&& hessian)
        : gradient_{std::move(gradient)}, hessian_{std::move(hessian)} {}
   public:
    Result(Result&&) = default;
    Result(Result const&) = delete;
    Result& operator=(Result const&) = delete;
    int64_t Candidates() const { return gradient_.left_missing_right.Count(); }
    // Writable CANN tensor handles remain private, as for Histogram.
  };
 private:
  Ops& ops_;
  std::size_t launches_{0};
  Channel BuildChannel(Tensor const& bins, Tensor const& parent) {
    auto const& runtime = ops_.GetRuntime();
    if (bins.Owner() != &runtime || parent.Owner() != &runtime ||
        bins.Type() != ACL_INT64 || parent.Type() != ACL_INT64 || parent.Count() != 1 ||
        bins.Count() <= 0 || bins.Count() > INT32_MAX) {
      throw std::invalid_argument("NPU split statistics tensor layout mismatch");
    }
    Channel out{
        Tensor{runtime, 1, ACL_INT64},
        Tensor{runtime, bins.Count(), ACL_INT64},
        Tensor{runtime, bins.Count(), ACL_INT64},
        Tensor{runtime, bins.Count(), ACL_INT64},
        Tensor{runtime, bins.Count(), ACL_INT64}};
    XgbNpuSplitStatistics(
        runtime.Stream(), bins.Storage().Data(), parent.Storage().Data(),
        out.missing.Storage().Data(), out.left_missing_right.Storage().Data(),
        out.right_missing_right.Storage().Data(), out.left_missing_left.Storage().Data(),
        out.right_missing_left.Storage().Data(), static_cast<uint32_t>(bins.Count()));
    ++launches_;
    runtime.Synchronize();
    return out;
  }
 public:
  explicit NumericalSplitStatistics(Ops& ops) : ops_{ops} {}
  std::size_t Launches() const { return launches_; }
  // nullptr means a zero-bin feature: CUDA's scan loop has no iterations.
  // Histogram's validated immutable source bounds both sign totals, so every
  // bin prefix, missing residual, and disjoint child subset fits int64. This
  // proof also covers subtracting histograms and cancellation near the limits.
  std::unique_ptr<Result> Build(IntegerHistogramBuilder::Histogram const& histogram,
                                std::size_t feature_index) {
    if (histogram.root_gradient_.Owner() != &ops_.GetRuntime()) {
      throw std::invalid_argument("NPU split histogram belongs to a different runtime");
    }
    if (feature_index >= histogram.features_.size()) {
      throw std::invalid_argument("NPU split feature index out of range");
    }
    auto const& f = histogram.features_[feature_index];
    if (f.bin_count == 0) return nullptr;
    return std::unique_ptr<Result>{new Result{BuildChannel(f.gradient, histogram.root_gradient_),
                                              BuildChannel(f.hessian, histogram.root_hessian_)}};
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_SPLIT_STATISTICS_H_
