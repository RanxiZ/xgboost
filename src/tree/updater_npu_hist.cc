// Copyright 2026, XGBoost Contributors
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/gradient.h"
#include "xgboost/json.h"
#include "xgboost/task.h"
#include "xgboost/tree_updater.h"
#include "../collective/communicator-inl.h"
#include "../common/npu/device_row_partitioner.h"
#include "../common/npu/dynamic_quantizer.h"
#include "../common/npu/native_training.h"
#include "../common/npu/cuda_column_sampler.h"
#include "../common/npu/numerical_split_evaluator.h"
#include "../common/random.h"
#include "driver.h"
#include "param.h"

namespace xgboost::tree {
namespace {
namespace npu = common::npu;

uint32_t FloatBits(float value) {
  uint32_t out;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}
float BitsFloat(uint32_t value) {
  float out;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

// CPU contains only the fixed-size selected split and growth-control metadata.
// No gradients, histograms, row membership or bin columns are downloaded.
struct NpuExpandEntry {
  bst_node_t nid;
  bst_node_t depth;
  npu::precise::SplitSummary split;
  bst_node_t GetNodeId() const { return nid; }
  float GetLossChange() const { return BitsFloat(split.loss_change); }
};

void ValidateParameters(TrainParam const& p) {
  auto finite_nonnegative = [](float v) { return std::isfinite(v) && v >= 0; };
  auto probability = [](float v) { return std::isfinite(v) && v > 0 && v <= 1; };
  if (p.max_depth < 0 || p.max_leaves < 0 || (!p.max_depth && !p.max_leaves) ||
      p.max_bin < 2 || !finite_nonnegative(p.learning_rate) ||
      !finite_nonnegative(p.min_split_loss) || !finite_nonnegative(p.min_child_weight) ||
      !finite_nonnegative(p.reg_lambda) || !finite_nonnegative(p.reg_alpha) ||
      !finite_nonnegative(p.max_delta_step) || !probability(p.subsample) ||
      !probability(p.colsample_bytree) || !probability(p.colsample_bylevel) ||
      !probability(p.colsample_bynode) ||
      (p.grow_policy != TrainParam::kDepthWise && p.grow_policy != TrainParam::kLossGuide)) {
    throw std::invalid_argument("grow_npu_hist received unsupported/invalid training parameters");
  }
  if (p.sampling_method != TrainParam::kUniform) {
    throw std::invalid_argument("grow_npu_hist supports uniform sampling only");
  }
  if (std::any_of(p.monotone_constraints.begin(), p.monotone_constraints.end(),
                  [](int v) { return v != 0; }) || !p.interaction_constraints.empty()) {
    throw std::invalid_argument("grow_npu_hist does not support monotone or interaction constraints");
  }
}

class NpuHistMaker : public TreeUpdater {
  bool square_error_task_;
  npu::NativeTrainingData data_;
  npu::CudaColumnSampler columns_;

 public:
  explicit NpuHistMaker(Context const* ctx, ObjInfo const* task)
      : TreeUpdater{ctx}, square_error_task_{task && task->task == ObjInfo::kRegression &&
                                                  task->const_hess} {}

  void Configure(Args const& args) override {
    // The learner owns TrainParam. Reject bypasses even for direct updater use.
    for (auto const& kv : args) {
      if ((kv.first == "objective" && kv.second != "reg:squarederror") ||
          (kv.first == "booster" && kv.second != "gbtree") ||
          (kv.first == "num_parallel_tree" && kv.second != "1") ||
          (kv.first == "process_type" && kv.second != "default")) {
        throw std::invalid_argument("grow_npu_hist requires square error, gbtree, one new scalar tree");
      }
    }
  }
  void LoadConfig(Json const&) override {}
  void SaveConfig(Json* out) const override { (*out)["name"] = String{Name()}; }
  char const* Name() const override { return "grow_npu_hist"; }

  void Update(TrainParam const* param, GradientContainer* gradients, DMatrix* dmat,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              std::vector<RegTree*> const& trees) override {
    if (!ctx_ || !ctx_->Device().IsNPU() || ctx_->Device().ordinal < 0 || !param ||
        !dmat || !gradients || !square_error_task_ || collective::IsDistributed()) {
      throw std::invalid_argument("grow_npu_hist requires single-card NPU scalar square-error training");
    }
    ValidateParameters(*param);
    if (trees.size() != 1 || !trees.front() || trees.front()->IsMultiTarget() ||
        trees.front()->HasCategoricalSplit() || trees.front()->NumNodes() != 1 ||
        !(*trees.front())[0].IsLeaf() || gradients->HasValueGrad()) {
      throw std::invalid_argument("grow_npu_hist grows exactly one fresh scalar tree per call");
    }
    auto native = gradients->npu_gradient;
    if (!native || native->nrows != dmat->Info().num_row_ ||
        (native->prediction_state && native->prediction_state->ngroups != 1)) {
      throw std::invalid_argument("grow_npu_hist requires the objective's native scalar gradient sidecar");
    }
    if (!native->runtime || !native->ops || !native->gradient || !native->hessian ||
        &native->ops->GetRuntime() != native->runtime.get()) {
      throw std::invalid_argument("Invalid native gradient ownership");
    }
    for (auto const* channel : {native->gradient.get(), native->hessian.get()}) {
      if (channel->Owner() != native->runtime.get() || channel->Type() != ACL_FLOAT ||
          channel->Count() != native->nrows) {
        throw std::invalid_argument("Invalid native gradient tensor dtype, shape or runtime");
      }
    }
    auto cpu = ctx_->MakeCPU();
    data_.Bind(&cpu, dmat, param->max_bin, native->runtime, native->ops);
    auto& ops = *native->ops;
    npu::DynamicQuantizer quantizer{ops};
    // Pinned CUDA Reset fits scales and quantizes BEFORE uniform sampling.
    // Fit with the full original channels to retain those scale semantics.
    // When sampling, copy/mask BOTH float32 channels on NPU, then quantize
    // those masked values with the already certified original scales. This
    // preserves CUDA's quantized survivors and exact zero sampled-out pairs,
    // while respecting the native pre-quantization float-mask contract.
    auto quantize_begin = npu::StatsEnabled() ? npu::NowNs() : 0;
    auto g = quantizer.Convert(*native->gradient, false);
    auto h = quantizer.Convert(*native->hessian, true);
    std::optional<npu::Tensor> gvalues, hvalues;
    if (param->subsample < 1) {
      // GPUHist's UniformSampling consumes exactly one Context RNG value per
      // tree. Keep that driver contract; the device kernel expands it into a
      // row-indexed counter stream so launch geometry cannot affect membership.
      auto seed = static_cast<uint64_t>(ctx_->Rng()());
      auto sampled = npu::SampleNativeGradient(*native, param->subsample, seed);
      npu::FixedQuantizer fixed{ops};
      auto sampled_g = fixed.Convert(sampled.gradient, g.certificate.scale, false);
      auto sampled_h = fixed.Convert(sampled.hessian, h.certificate.scale, true);
      gvalues.emplace(std::move(sampled_g.values));
      hvalues.emplace(std::move(sampled_h.values));
    } else {
      gvalues.emplace(std::move(g.channel.values));
      hvalues.emplace(std::move(h.channel.values));
    }
    if (npu::StatsEnabled()) npu::Stats().quantize_ns += npu::NowNs() - quantize_begin;
    Grow(*param, dmat, trees.front(), ops, std::move(*gvalues), std::move(*hvalues),
         g.certificate.scale, h.certificate.scale);
    // Deliberately disable generic CPU leaf-id/prediction-cache paths. The
    // parent gbm commits this standard RegTree and predicts on NPU next round.
    for (auto& position : out_position) position.Resize(0);
  }

 private:
  void Grow(TrainParam const& p, DMatrix* dmat, RegTree* tree,
            npu::Ops& ops, npu::Tensor&& gradient, npu::Tensor&& hessian,
            npu::QuantizerScale gscale, npu::QuantizerScale hscale) {
    // Structural row-id and subset proofs stay on the device and are read back
    // once per expansion batch (after the split-summary download), not once per
    // node. The deferred verdicts cannot change device work, only error timing.
    auto grow_begin = npu::StatsEnabled() ? npu::NowNs() : 0;
    npu::Ops::DeferScope defer_controls{ops};
    npu::IntegerHistogramBuilder builder{ops, data_.Bins(), std::move(gradient),
                                         std::move(hessian)};
    auto root = builder.BuildRoot();
    npu::DeviceRowPartitioner partitioner{ops, root};
    npu::NumericalSplitEvaluator evaluator{ops};
    npu::precise::Parameters parameters{FloatBits(p.min_child_weight), FloatBits(p.reg_lambda),
                                         FloatBits(p.reg_alpha), FloatBits(p.max_delta_step),
                                         gscale.exponent, hscale.exponent};
    columns_.Init(ctx_->Rng(), dmat->Info().num_col_,
                   dmat->Info().feature_weights.ConstHostVector(),
                   p.colsample_bynode, p.colsample_bylevel, p.colsample_bytree);
    auto request = [&](npu::IntegerHistogramBuilder::Histogram const& hist, bst_node_t depth) {
      auto features = columns_.GetFeatureSet(ctx_->Rng(), depth);
      return npu::NumericalSplitEvaluator::Request{&hist, parameters, *features};
    };
    auto hess = [&](int64_t value) {
      // A scalar selected-node summary, not host split scoring or reduction.
      return static_cast<float>(static_cast<double>(value) * hscale.to_floating);
    };
    auto root_split = evaluator.EvaluateBatch({request(root, 0)}).front();
    auto weight = BitsFloat(root_split.node.weight);
    tree->Stat(0) = {0.0f, hess(root_split.hessian), weight};
    (*tree)[0].SetLeaf(p.learning_rate * weight);

    using Rows = npu::DeviceRowPartitioner::Rows;
    using Histogram = npu::IntegerHistogramBuilder::Histogram;
    struct Pending {
      Rows rows;
      Histogram hist;
      Pending(Rows&& r, Histogram&& h) : rows{std::move(r)}, hist{std::move(h)} {}
    };
    std::map<bst_node_t, std::unique_ptr<Pending>> pending;
    pending.emplace(0, std::make_unique<Pending>(partitioner.RootBuffered(), std::move(root)));
    // Official GPUHistMaker uses a 1024-entry depthwise expansion batch.
    // Driver itself enforces loss-guided and max-leaf admission rules.
    Driver<NpuExpandEntry> driver{p, 1024U};
    struct Children {
      NpuExpandEntry parent;
      bst_node_t left_id, right_id;
      Rows left_rows, right_rows;
      Histogram left, right;
    };
    driver.Push(NpuExpandEntry{0, 0, root_split});
    while (!driver.IsEmpty()) {
      auto batch = driver.Pop();
      std::vector<Children> children;
      children.reserve(batch.size());
      for (auto const& entry : batch) {
        auto node = pending.find(entry.nid);
        if (node == pending.end() || !entry.split.best.valid) {
          throw std::logic_error("Missing native expansion state");
        }
        auto const& s = entry.split;
        auto const& best = s.best;
        auto partition = partitioner.PartitionConsume(std::move(node->second->rows),
            {best.feature, best.candidate, best.missing_left != 0});
        auto threshold = data_.Threshold(best.feature, best.candidate);
        tree->ExpandNode(entry.nid, best.feature, threshold, best.missing_left != 0,
            BitsFloat(s.node.weight), BitsFloat(best.score.left_weight) * p.learning_rate,
            BitsFloat(best.score.right_weight) * p.learning_rate, entry.GetLossChange(),
            hess(s.hessian), hess(s.left_hessian), hess(s.right_hessian));
        auto left_id = (*tree)[entry.nid].LeftChild();
        auto right_id = (*tree)[entry.nid].RightChild();
        if (driver.IsChildValid(entry)) {
          // Build only the smaller child; SubtractSibling proves membership
          // subset on NPU and validates identical immutable source/scales.
          bool build_left = partition.left.Count() <= partition.right.Count();
          auto small = partitioner.BuildHistogram(builder, build_left ? partition.left : partition.right);
          auto sibling = builder.SubtractSibling(node->second->hist, small);
          auto left = build_left ? std::move(small) : std::move(sibling);
          auto right = build_left ? std::move(sibling) : std::move(small);
          children.push_back(Children{entry, left_id, right_id, std::move(partition.left),
                                      std::move(partition.right), std::move(left), std::move(right)});
        }
        pending.erase(node);
      }
      std::vector<npu::NumericalSplitEvaluator::Request> requests;
      requests.reserve(2 * children.size());
      for (auto const& child : children) {
        auto depth = child.parent.depth + 1;
        // Preserve the former parent/left/right feature-sampling order exactly.
        requests.push_back(request(child.left, depth));
        requests.push_back(request(child.right, depth));
      }
      auto summaries = evaluator.EvaluateBatch(requests);
      for (std::size_t i = 0; i < children.size(); ++i) {
        auto& child = children[i]; auto depth = child.parent.depth + 1;
        auto const& s = child.parent.split;
        auto const& ls = summaries[2 * i]; auto const& rs = summaries[2 * i + 1];
        if (ls.gradient != s.left_gradient || ls.hessian != s.left_hessian ||
            rs.gradient != s.right_gradient || rs.hessian != s.right_hessian) {
          throw std::runtime_error("NPU partition/subtraction disagrees with selected child statistics");
        }
        for (auto const& entry : {NpuExpandEntry{child.left_id, depth, ls},
                                  NpuExpandEntry{child.right_id, depth, rs}}) {
          if (entry.split.best.valid && entry.GetLossChange() > kRtEps) driver.Push(entry);
        }
        if (ls.best.valid && BitsFloat(ls.loss_change) > kRtEps) {
          pending.emplace(child.left_id, std::make_unique<Pending>(std::move(child.left_rows), std::move(child.left)));
        }
        if (rs.best.valid && BitsFloat(rs.loss_change) > kRtEps) {
          pending.emplace(child.right_id, std::make_unique<Pending>(std::move(child.right_rows), std::move(child.right)));
        }
      }
    }
    ops.FlushControls();
    ops.GetRuntime().Synchronize();
    if (npu::StatsEnabled()) npu::Stats().grow_ns += npu::NowNs() - grow_begin;
  }
};
}  // namespace

XGBOOST_REGISTER_TREE_UPDATER(NpuHistMaker, "grow_npu_hist")
    .describe("Native single-card Ascend numerical integer-histogram tree grower.")
    .set_body([](Context const* ctx, ObjInfo const* task) { return new NpuHistMaker{ctx, task}; });
}  // namespace xgboost::tree