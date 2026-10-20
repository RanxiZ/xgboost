// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_CANDIDATE_SELECTION_H_
#define XGBOOST_COMMON_NPU_CANDIDATE_SELECTION_H_
#include "candidate_selection_abi.h"
#include "precise_split_scorer.h"
namespace xgboost::common::npu {
class CandidateSelection {
 public:
  class Result {
    friend class CandidateSelection;
    friend class NumericalSplitEvaluator;
    Buffer selected_;
    Runtime const* owner_;
    NumericalSplitStatistics::Result const* provenance_;
    Result(Buffer&& b,Runtime const* owner,NumericalSplitStatistics::Result const* provenance)
        : selected_{std::move(b)},owner_{owner},provenance_{provenance} {}
   public:
    Result(Result&&)=default;
    Result(Result const&)=delete;
  };
 private:
  Ops& ops_;
 public:
  explicit CandidateSelection(Ops& ops):ops_{ops} {}
  // Score and reduce together, preventing callers from pairing scores with
  // different statistics. No public writable tensor or score-array download.
  Result Select(NumericalSplitStatistics::Result const& statistics,precise::Parameters p,uint32_t feature) {
    if (feature==UINT32_MAX) throw std::invalid_argument("NPU reserved feature id");
    PreciseSplitScorer scorer{ops_}; auto scores=scorer.Evaluate(statistics,p);
    auto const& r=ops_.GetRuntime(); auto out=r.AllocateElements<precise::Candidate>(1);
    XgbNpuSelectCandidates(r.Stream(),scores.missing_left_.Data(),scores.missing_right_.Data(),
                          out.Data(),static_cast<uint32_t>(scores.Candidates()),feature);
    r.Synchronize();
    return Result{std::move(out),&r,&statistics};
  }
};
}  // namespace xgboost::common::npu
#endif