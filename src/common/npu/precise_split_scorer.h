// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_PRECISE_SPLIT_SCORER_H_
#define XGBOOST_COMMON_NPU_PRECISE_SPLIT_SCORER_H_
#include "precise_scoring_abi.h"
#include "split_statistics.h"
namespace xgboost::common::npu {
class PreciseSplitScorer {
 public:
  class Result {
    friend class PreciseSplitScorer;
    friend class PreciseScorerTestAccess;
    friend class CandidateSelection;
    Buffer missing_left_,missing_right_;
    int64_t candidates_;
    Result(Buffer&& l,Buffer&& r,int64_t n)
        : missing_left_{std::move(l)},missing_right_{std::move(r)},candidates_{n} {}
   public:
    Result(Result&&) = default;
    Result(Result const&) = delete;
    int64_t Candidates() const { return candidates_; }
  };
 private:
  Ops& ops_;
  std::size_t launches_{0};
  Buffer Run(Tensor const& lg,Tensor const& lh,Tensor const& rg,Tensor const& rh,Buffer const& parameters) {
    auto const& runtime=ops_.GetRuntime(); auto n=lg.Count();
    auto out=runtime.AllocateElements<precise::Score>(n);
    // Positive sentinel detects an unwritten status; the guarded histogram
    // size bounds the reduction well below int64 overflow.
    auto errors=ops_.Constant(n,4);
    XgbNpuScoreColumns(runtime.Stream(),lg.Storage().Data(),lh.Storage().Data(),rg.Storage().Data(),
                      rh.Storage().Data(),parameters.Data(),out.Data(),errors.Storage().Data(),static_cast<uint32_t>(n));
    ++launches_; runtime.Synchronize();
    if (ops_.SumControl(errors)!=0) throw std::overflow_error("NPU split scoring outside supported arithmetic domain");
    return out;
  }
 public:
  explicit PreciseSplitScorer(Ops& ops) : ops_{ops} {}
  std::size_t Launches() const { return launches_; }
  // Supplied scale exponents must be those of the immutable histogram source;
  // binding quantizer metadata to dataset ownership is native integration work.
  // No constraints/categorical/parent-gain subtraction/candidate selection here.
  Result Evaluate(NumericalSplitStatistics::Result const& statistics,precise::Parameters p) {
    auto const& g=statistics.gradient_; auto const& h=statistics.hessian_;
    if (g.left_missing_right.Owner()!=&ops_.GetRuntime()) {
      throw std::invalid_argument("NPU split statistics belong to a different runtime");
    }
    for (auto v:{p.min_child_weight,p.reg_lambda,p.reg_alpha,p.max_delta_step}) {
      if (v>=0x7f800000U) throw std::invalid_argument("NPU split scoring parameters must be finite and nonnegative, without negative zero");
    }
    if (p.gradient_scale < -1022 || p.gradient_scale > 1022 || p.hessian_scale < -1022 || p.hessian_scale > 1022) {
      throw std::invalid_argument("NPU split scoring scale out of range");
    }
    auto metadata=ops_.GetRuntime().Allocate(sizeof(p)); metadata.Upload(&p,sizeof(p));
    auto left=Run(g.left_missing_left,h.left_missing_left,g.right_missing_left,h.right_missing_left,metadata);
    auto right=Run(g.left_missing_right,h.left_missing_right,g.right_missing_right,h.right_missing_right,metadata);
    return Result{std::move(left),std::move(right),statistics.Candidates()};
  }
};
}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_PRECISE_SPLIT_SCORER_H_
