// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_NUMERICAL_SPLIT_EVALUATOR_H_
#define XGBOOST_COMMON_NPU_NUMERICAL_SPLIT_EVALUATOR_H_
#include "numerical_split_evaluator_abi.h"
#include "candidate_selection.h"
#include <cstring>
#include <climits>
#include <set>
namespace xgboost::common::npu {
// Numerical, unconstrained, single-target/single-device evaluation only.
// Each request retains the original CUDA feature input order, including the
// pinned CCCL block-striped candidate tie order. A batch shares allocation and
// the sole summary download, never gradient provenance or node reductions.
class NumericalSplitEvaluator {
  Ops& ops_;
  std::size_t control_download_bytes_{0}, launches_{0};
  static std::size_t Add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error("NPU batch count overflow");
    return a + b;
  }
  static std::size_t Bytes(std::size_t count, std::size_t size) {
    if (count > std::numeric_limits<std::size_t>::max() / size) throw std::overflow_error("NPU batch byte count overflow");
    return count * size;
  }
  struct Prepared {
    std::vector<uint64_t> metadata;
    std::size_t count, candidates, status_bytes;
    std::size_t sizes[6], offsets[6];
    Prepared(std::size_t features, std::size_t bins, std::size_t prefixes)
        : count{features}, candidates{bins}, status_bytes{Bytes(Add(Add(features, bins), features), sizeof(uint32_t))},
          sizes{Add(sizeof(precise::NodeBatchHeader), Bytes(features, sizeof(precise::NodeBatchFeature))),
                Bytes(prefixes, sizeof(int64_t)), Bytes(prefixes, sizeof(int64_t)),
                Bytes(bins, sizeof(precise::DirectionOutput)), Bytes(features, sizeof(precise::Candidate)), status_bytes},
          offsets{} {
      metadata.resize(sizes[0] / sizeof(uint64_t), 0);
    }
  };
 public:
  struct Request {
    IntegerHistogramBuilder::Histogram const* histogram;
    precise::Parameters parameters;
    std::vector<uint32_t> feature_order;
  };
  explicit NumericalSplitEvaluator(Ops& ops) : ops_{ops} {}
  std::size_t ControlDownloadBytes() const { return control_download_bytes_; }
  std::size_t Launches() const { return launches_; }
  precise::SplitSummary Evaluate(IntegerHistogramBuilder::Histogram const& histogram,
                                 precise::Parameters p, std::vector<uint32_t> const& feature_order) {
    return EvaluateBatch({Request{&histogram, p, feature_order}}).front();
  }
  std::vector<precise::SplitSummary> EvaluateBatch(std::vector<Request> const& requests) {
    auto evaluate_begin = StatsEnabled() ? NowNs() : 0;
    auto evaluate_wait = StatsEnabled() ? WaitTotalNs() : 0;
    if (requests.empty()) return {};
    auto evaluate_timer = [&] {
      if (StatsEnabled()) {
        Stats().evaluate_ns += NowNs() - evaluate_begin;
        Stats().evaluate_wait_ns += WaitTotalNs() - evaluate_wait;
        ++Stats().n_eval;
      }
    };
    auto const& r = ops_.GetRuntime();
    std::vector<Prepared> prepared;
    prepared.reserve(requests.size());
    // All requests are checked before any device allocation, upload or launch.
    for (auto const& request : requests) {
      if (!request.histogram || !request.histogram->source_ ||
          request.histogram->root_gradient_.Owner() != &r) {
        throw std::invalid_argument("NPU evaluator histogram runtime mismatch");
      }
      if (!precise::ValidParameters(request.parameters)) throw std::invalid_argument("NPU evaluator unsupported parameters");
      auto const& histogram = *request.histogram;
      auto const& order = request.feature_order;
      if (order.size() > UINT32_MAX) throw std::invalid_argument("NPU batch feature count exceeds uint32");
      std::set<uint32_t> seen;
      std::size_t candidates = 0, prefixes = 0;
      for (auto f : order) {
        if (f == UINT32_MAX || f >= histogram.FeatureCount() || !seen.insert(f).second) {
          throw std::invalid_argument("NPU evaluator invalid feature order");
        }
        auto bins = histogram.features_[f].bin_count;
        if (bins < 0 || bins > INT32_MAX) throw std::invalid_argument("NPU batch invalid bin count");
        candidates = Add(candidates, static_cast<std::size_t>(bins));
        prefixes = Add(prefixes, Add(static_cast<std::size_t>(bins), 1));
      }
      prepared.emplace_back(order.size(), candidates, prefixes);
      auto& work = prepared.back();
      precise::NodeBatchHeader header{request.parameters, static_cast<uint32_t>(order.size()), 0};
      std::memcpy(work.metadata.data(), &header, sizeof(header));
      std::size_t prefix_offset = 0, candidate_offset = 0;
      for (std::size_t position = 0; position < order.size(); ++position) {
        auto f = order[position]; auto const& bins = histogram.features_[f];
        precise::NodeBatchFeature descriptor{
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bins.gradient.Storage().Data())),
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bins.hessian.Storage().Data())),
          static_cast<uint64_t>(prefix_offset), static_cast<uint64_t>(candidate_offset),
          static_cast<uint32_t>(bins.bin_count), f};
        std::memcpy(reinterpret_cast<char*>(work.metadata.data()) + sizeof(header) + position * sizeof(descriptor),
                    &descriptor, sizeof(descriptor));
        prefix_offset += static_cast<std::size_t>(bins.bin_count) + 1;
        candidate_offset += static_cast<std::size_t>(bins.bin_count);
      }
    }
    auto summary_bytes = Bytes(requests.size(), sizeof(precise::SplitSummary));
    auto total = summary_bytes;
    for (auto& work : prepared) {
      for (std::size_t i = 0; i < 6; ++i) {
        total = Add(total, (32 - total % 32) % 32);
        work.offsets[i] = total;
        total = Add(total, work.sizes[i]);
      }
    }
    static_cast<void>(Add(total, r.LiveBytes()));
    // One allocation owns every node workspace. Bounded slices retain it through
    // the synchronized summary download and avoid per-buffer frees/waits.
    auto storage = std::make_shared<Buffer>(r.Allocate(total));
    auto view = [&](std::size_t offset, std::size_t bytes) {
      return bytes ? Buffer::Slice(storage, offset, bytes) : Buffer{};
    };
    auto summary = view(0, summary_bytes);
    std::vector<precise::SplitSummary> out(requests.size());
    for (auto& sentinel : out) {
      sentinel.best = precise::EmptyCandidate();
      sentinel.loss_change = 0xff800000U; sentinel.error = precise::kBatchIncomplete;
    }
    summary.Upload(out.data(), summary_bytes);
    // Every upload/zero completes before the first device task. Fresh status
    // retains the existing incomplete-payload checks, including zero-bin nodes.
    for (auto const& work : prepared) {
      auto metadata = view(work.offsets[0], work.sizes[0]);
      auto status = view(work.offsets[5], work.sizes[5]);
      metadata.Upload(work.metadata.data(), work.sizes[0]); status.Zero();
    }
    auto stream = r.Stream();
    auto base = static_cast<char*>(storage->Data());
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto const& work = prepared[i]; auto const& histogram = *requests[i].histogram;
      auto pointer = [&](std::size_t slot) -> void* { return work.sizes[slot] ? base + work.offsets[slot] : nullptr; };
      auto status = static_cast<char*>(pointer(5));
      void* prefix_status = status;
      void* score_status = status ? status + work.count * sizeof(uint32_t) : nullptr;
      void* selected_status = status ? status + (work.count + work.candidates) * sizeof(uint32_t) : nullptr;
      auto root_g = histogram.root_gradient_.Storage().Data();
      auto root_h = histogram.root_hessian_.Storage().Data();
      auto n = static_cast<uint32_t>(work.count);
      XgbNpuBatchedSplitPrefix(stream, pointer(0), pointer(1), pointer(2), prefix_status, n); ++launches_;
      XgbNpuBatchedSplitScore(stream, pointer(0), root_g, root_h, pointer(1), pointer(2), prefix_status, pointer(3), score_status, n); ++launches_;
      XgbNpuBatchedSplitSelect(stream, pointer(0), prefix_status, pointer(3), score_status, pointer(4), selected_status, n); ++launches_;
      XgbNpuBatchedSplitFinish(stream, pointer(0), root_g, root_h, pointer(1), pointer(2), prefix_status, pointer(4), selected_status,
                              base + i * sizeof(precise::SplitSummary)); ++launches_;
    }
    summary.Download(out.data(), summary_bytes); control_download_bytes_ += summary_bytes;
    // The summary drain already retired every producer enqueued before it, so
    // deferred structural verdicts from this expansion batch cost one extra
    // staged event wait on an otherwise idle stream.
    ops_.FlushControls();
    for (auto const& result : out) {
      if (result.error) throw std::overflow_error("NPU evaluator outside supported arithmetic domain or incomplete batch");
    }
    evaluate_timer();
    return out;
  }
};
}  // namespace xgboost::common::npu
#endif