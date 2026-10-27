// Copyright 2026, XGBoost Contributors
#include "native_prediction.h"

#include "acl_api.h"
#include "xgboost/gbm.h"
#include "../../gbm/gbtree_model.h"
#include "xgboost/logging.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace xgboost::common::npu {
namespace {

void CheckContext(Context const* ctx) {
  if (ctx == nullptr || !ctx->Device().IsNPU() || ctx->Device().ordinal < 0) {
    throw std::invalid_argument("Native Ascend prediction requires a valid NPU Context");
  }
}

uint32_t CheckedU32(std::size_t n, char const* what) {
  if (n > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(std::string("NPU ") + what + " exceeds uint32 range");
  }
  return static_cast<uint32_t>(n);
}

void CheckFinite(float value, char const* what) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("NPU ") + what + " must be finite");
  }
}

void CheckNumericFeatures(MetaInfo const& info) {
  if (!info.feature_types.Empty()) {
    if (info.feature_types.Size() != info.num_col_) {
      throw std::invalid_argument("NPU feature type shape does not match feature count");
    }
    for (auto type : info.feature_types.ConstHostSpan()) {
      if (type != FeatureType::kNumerical) {
        throw std::invalid_argument("Native Ascend prediction supports numerical features only");
      }
    }
  }
}

void CheckModel(Context const* ctx, MetaInfo const& info, gbm::GBTreeModel const& model) {
  CheckContext(ctx);
  if (!model.Ctx() || model.Ctx()->Device() != ctx->Device()) {
    throw std::invalid_argument("NPU prediction model belongs to a different device context");
  }
  auto const* param = model.learner_model_param;
  if (!param || param->OutputLength() != 1 || param->IsVectorLeaf()) {
    throw std::invalid_argument("NPU prediction requires single-output scalar trees");
  }
  if (info.num_col_ > param->num_feature) {
    throw std::invalid_argument("NPU prediction feature count exceeds model feature count");
  }
  if (!model.weight_drop.empty()) {
    throw std::invalid_argument("Native Ascend prediction does not support weighted DART trees");
  }
  CheckNumericFeatures(info);
}

void CheckState(Context const* ctx, NativePrediction const& state) {
  CheckContext(ctx);
  if (ctx->Device().ordinal != state.device_ordinal) {
    throw std::invalid_argument("NPU prediction cache belongs to a different device ordinal");
  }
  if (!state.runtime || !state.ops || !state.status || !state.status->Data() ||
      state.status->Size() != sizeof(native_abi::Status) ||
      !state.nodes_device || !state.nodes_device->Data() ||
      !state.trees_device || !state.trees_device->Data() || state.ngroups != 1 ||
      &state.ops->GetRuntime() != state.runtime.get()) {
    throw std::invalid_argument("NPU prediction cache has incomplete device state");
  }
  if (state.nrows && (!state.predictions || state.predictions->Type() != ACL_FLOAT ||
                     state.predictions->Owner() != state.runtime.get() ||
                     state.predictions->Count() != static_cast<int64_t>(state.nrows) * state.ngroups)) {
    throw std::invalid_argument("NPU prediction tensor has invalid dtype, shape or runtime");
  }
  // Stream() activates this state's private context even when no allocation or
  // copy is needed (the common resident-cache path).
  state.runtime->Stream();
}

void ClearStatus(NativePrediction* state) {
  if (!state || !state->runtime || !state->status || !state->status->Data()) {
    throw std::invalid_argument("NPU prediction status is not initialized");
  }
  native_abi::Status clean{};
  state->status->Upload(&clean, sizeof(clean));
}

void CheckPredictionValid(NativePrediction const& state) {
  if (!state.prediction_valid) {
    throw std::runtime_error("NPU prediction cache requires initialization after a failed prediction");
  }
}

void FlattenModel(NativePrediction* state, gbm::GBTreeModel const& model) {
  if (&model != state->model_identity || model.learner_model_param == nullptr ||
      model.learner_model_param->OutputLength() != state->ngroups ||
      model.learner_model_param->num_feature != state->nfeatures ||
      model.learner_model_param->IsVectorLeaf() || !model.weight_drop.empty()) {
    throw std::invalid_argument("NPU prediction cache model identity or shape changed; reinitialize it");
  }
  if (model.tree_info.Size() != model.trees.size()) {
    throw std::invalid_argument("NPU model tree group metadata has invalid shape");
  }
  CheckedU32(model.trees.size(), "tree count");
  auto groups = model.tree_info.ConstHostSpan();
  std::vector<native_abi::Node> nodes;
  std::vector<native_abi::Tree> trees;
  trees.reserve(model.trees.size());
  for (std::size_t tid = 0; tid < model.trees.size(); ++tid) {
    auto const* tree = model.trees[tid].get();
    if (!tree || tree->IsMultiTarget() || tree->HasCategoricalSplit()) {
      throw std::invalid_argument("Native Ascend prediction supports numerical scalar trees only");
    }
    auto tree_nodes = tree->GetNodes(DeviceOrd::CPU());
    if (tree_nodes.empty() || tree_nodes.size() > static_cast<std::size_t>(INT32_MAX) ||
        tree_nodes.size() > UINT32_MAX - nodes.size()) {
      throw std::invalid_argument("NPU tree node count is invalid or exceeds ABI range");
    }
    std::vector<uint8_t> color(tree_nodes.size(), 0);
    std::vector<std::pair<int32_t, bool>> pending{{0, false}};
    while (!pending.empty()) {
      auto [node_id, leaving] = pending.back();
      pending.pop_back();
      if (node_id < 0 || static_cast<std::size_t>(node_id) >= tree_nodes.size()) {
        throw std::invalid_argument("NPU tree traversal node id is out of range");
      }
      auto index = static_cast<std::size_t>(node_id);
      if (leaving) {
        color[index] = 2;
        continue;
      }
      if (color[index] == 1) throw std::invalid_argument("NPU tree traversal contains a cycle");
      if (color[index] == 2) continue;
      color[index] = 1;
      auto const& current = tree_nodes[node_id];
      if (current.IsDeleted()) throw std::invalid_argument("NPU tree traversal reaches a deleted node");
      pending.push_back({node_id, true});
      if (current.IsLeaf()) {
        CheckFinite(current.LeafValue(), "tree leaf");
      } else {
        if (std::isnan(current.SplitCond()) || current.SplitIndex() >= state->nfeatures) {
          throw std::invalid_argument("NPU tree has an invalid numerical split");
        }
        pending.push_back({current.RightChild(), false});
        pending.push_back({current.LeftChild(), false});
      }
    }
    native_abi::Tree t{CheckedU32(nodes.size(), "node count"),
                       CheckedU32(tree_nodes.size(), "node count"), groups[tid], 0};
    if (t.group >= state->ngroups) throw std::invalid_argument("NPU tree group is out of range");
    trees.push_back(t);
    nodes.reserve(nodes.size() + tree_nodes.size());
    for (std::size_t i = 0; i < tree_nodes.size(); ++i) {
      auto const& node = tree_nodes[i];
      native_abi::Node n{};
      // Pruning can leave deleted, unreachable records in a RegTree. They are
      // never traversed and do not need to satisfy active-node invariants.
      n.is_leaf = 1;
      if (color[i]) {
        n.left = node.LeftChild();
        n.right = node.RightChild();
        n.feature = node.SplitIndex();
        n.threshold = node.IsLeaf() ? 0.0f : node.SplitCond();
        n.leaf = node.IsLeaf() ? node.LeafValue() : 0.0f;
        n.is_leaf = node.IsLeaf();
        n.default_left = node.DefaultLeft();
      }
      nodes.push_back(n);
    }
  }
  bool unchanged = nodes.size() == state->nodes.size() && trees.size() == state->trees.size() &&
                   (nodes.empty() || std::memcmp(nodes.data(), state->nodes.data(),
                                                 nodes.size() * sizeof(native_abi::Node)) == 0) &&
                   (trees.empty() || std::memcmp(trees.data(), state->trees.data(),
                                                 trees.size() * sizeof(native_abi::Tree)) == 0);
  // The first empty model still needs valid dummy allocations. An empty host
  // vector does not prove that device state has ever been initialized.
  if (unchanged && state->nodes_device && state->trees_device) return;
  auto node_bytes = nodes.size() * sizeof(native_abi::Node);
  auto tree_bytes = trees.size() * sizeof(native_abi::Tree);
  auto nodes_device = std::make_unique<Buffer>(state->runtime->Allocate(
      std::max<std::size_t>(sizeof(native_abi::Node), node_bytes)));
  auto trees_device = std::make_unique<Buffer>(state->runtime->Allocate(
      std::max<std::size_t>(sizeof(native_abi::Tree), tree_bytes)));
  if (node_bytes) nodes_device->Upload(nodes.data(), node_bytes);
  else nodes_device->Zero();
  if (tree_bytes) trees_device->Upload(trees.data(), tree_bytes);
  else trees_device->Zero();
  // The uploads are stream-ordered ahead of every prediction kernel that uses
  // them, and asynchronous uploads copy the host bytes into pinned staging
  // before returning, so no host wait is required here.
  state->nodes = std::move(nodes);
  state->trees = std::move(trees);
  state->nodes_device = std::move(nodes_device);
  state->trees_device = std::move(trees_device);
  state->uploaded_tree_count = CheckedU32(state->trees.size(), "tree count");
}

void InitializeOutput(NativePrediction* state, MetaInfo const& info, gbm::GBTreeModel const& model) {
  auto count = static_cast<std::size_t>(state->nrows) * state->ngroups;
  if (count > static_cast<std::size_t>(INT64_MAX) || count > SIZE_MAX / sizeof(float)) {
    throw std::overflow_error("NPU prediction output size exceeds tensor range");
  }
  std::vector<float> initial(count);
  if (!info.base_margin_.Empty()) {
    if (info.base_margin_.Shape(0) != state->nrows ||
        info.base_margin_.Shape(1) != state->ngroups || info.base_margin_.Size() != count) {
      throw std::invalid_argument("NPU base margin shape does not match prediction output");
    }
    auto margin = info.base_margin_.Data()->ConstHostSpan();
    std::copy(margin.begin(), margin.end(), initial.begin());
  } else {
    auto base = model.learner_model_param->BaseScore(DeviceOrd::CPU());
    if (base.Size() != 1 && base.Size() != state->ngroups) {
      throw std::invalid_argument("NPU base score shape does not match prediction output");
    }
    for (uint32_t row = 0; row < state->nrows; ++row) {
      for (uint32_t group = 0; group < state->ngroups; ++group) {
        initial[static_cast<std::size_t>(row) * state->ngroups + group] = base(base.Size() == 1 ? 0 : group);
      }
    }
  }
  for (auto value : initial) CheckFinite(value, "initial prediction");
  // Do not expose a partially reset output if allocation/copy/status clearing
  // fails. Reinitialization is the only recovery for a failed additive launch.
  state->prediction_valid = false;
  if (count) {
    if (!state->predictions) {
      state->predictions = std::make_unique<Tensor>(*state->runtime, static_cast<int64_t>(count), ACL_FLOAT);
    }
    state->predictions->Storage().Upload(initial.data(), count * sizeof(float));
  }
  ClearStatus(state);
  state->prediction_valid = true;
}

std::shared_ptr<NativePrediction> MakeState(Context const* ctx, MetaInfo const& info,
                                          gbm::GBTreeModel const& model) {
  CheckModel(ctx, info, model);
  auto state = std::make_shared<NativePrediction>();
  state->nrows = CheckedU32(info.num_row_, "row count");
  state->ngroups = CheckedU32(model.learner_model_param->OutputLength(), "output count");
  state->nfeatures = model.learner_model_param->num_feature;
  state->device_ordinal = ctx->Device().ordinal;
  state->model_identity = &model;
  state->runtime = std::make_shared<Runtime>(MakeAclApi(), state->device_ordinal);
  state->ops = std::make_shared<Ops>(*state->runtime);
  // Gradient generation on round zero does not call PredictNative/UploadRows.
  state->status = std::make_unique<Buffer>(state->runtime->Allocate(sizeof(native_abi::Status)));
  InitializeOutput(state.get(), info, model);
  FlattenModel(state.get(), model);
  return state;
}

void UploadRows(NativePrediction* state, DMatrix* dmat) {
  auto timer_begin = StatsEnabled() ? NowNs() : 0;
  struct Timer {
    std::uint64_t begin;
    ~Timer() { if (StatsEnabled()) Stats().upload_rows_ns += NowNs() - begin; }
  } timer{timer_begin};
  if (!dmat || dmat->Info().num_row_ != state->nrows) {
    throw std::invalid_argument("NPU DMatrix row count changed for prediction cache");
  }
  auto const& info = dmat->Info();
  if (info.num_col_ > state->nfeatures) {
    throw std::invalid_argument("NPU DMatrix feature count exceeds the prediction model");
  }
  CheckNumericFeatures(info);
  auto same_page = [](NativePrediction::SourcePage const& old,
                      NativePrediction::SourcePage const& now) {
    // Compare control blocks as well as addresses: an address by itself is not
    // a lifetime-safe identity. Holding old.page alive also prevents reuse.
    return now.page == old.page && !now.page.owner_before(old.page) &&
           !old.page.owner_before(now.page) && now.offsets == old.offsets &&
           now.entries == old.entries && now.nrows == old.nrows && now.nnz == old.nnz &&
           now.base_rowid == old.base_rowid;
  };
  auto describe = [](std::shared_ptr<SparsePage const> page) {
    if (!page) throw std::invalid_argument("NPU prediction source page has no shared ownership");
    return NativePrediction::SourcePage{page, page->offset.ConstHostPointer(),
                                        page->data.ConstHostPointer(), page->Size(),
                                        page->data.Size(), page->base_rowid};
  };
  std::vector<NativePrediction::SourcePage> pages;
  auto batches = dmat->GetBatches<SparsePage>();
  auto it = batches.begin();
  // The initial bind proves complete, contiguous CSR coverage. For an owned,
  // unchanged single-page DMatrix the resident path examines only O(1) storage
  // metadata, never all row offsets or feature values and never reuploads CSR.
  if (state->input_bound && dmat->SingleColBlock() && state->source_pages.size() == 1 &&
      state->input_nnz == info.num_nonzero_ && state->input_ncols == info.num_col_ &&
      !it.AtEnd() && same_page(state->source_pages.front(), describe(it.Page()))) {
    return;
  }
  for (; !it.AtEnd(); ++it) pages.push_back(describe(it.Page()));
  bool unchanged = state->input_bound && state->input_nnz == info.num_nonzero_ &&
                   state->input_ncols == info.num_col_ && pages.size() == state->source_pages.size();
  for (std::size_t i = 0; unchanged && i < pages.size(); ++i) {
    unchanged = same_page(state->source_pages[i], pages[i]);
  }
  if (unchanged) return;

  std::vector<uint64_t> offsets(static_cast<std::size_t>(state->nrows) + 1, 0);
  std::vector<native_abi::Entry> entries;
  std::size_t next_row = 0;
  for (auto const& source : pages) {
    auto const& page = *source.page;
    auto view = page.GetView();
    if (page.base_rowid != next_row || view.Size() > state->nrows - next_row ||
        view.offset.empty() || view.offset.front() != 0 || view.offset.back() != view.data.size()) {
      throw std::invalid_argument("NPU sparse page shape or row boundaries are invalid");
    }
    for (std::size_t i = 0; i < view.Size(); ++i) {
      if (view.offset[i] > view.offset[i + 1] || view.offset[i + 1] > view.data.size()) {
        throw std::invalid_argument("NPU sparse page offsets are invalid");
      }
      auto row_entries = view[i];
      uint32_t previous = 0;
      bool first = true;
      for (auto const& entry : row_entries) {
        if (entry.index >= info.num_col_ || (!first && entry.index <= previous)) {
          throw std::invalid_argument("NPU CSR feature indices must be in range, sorted and unique");
        }
        // NaN is a missing value on CUDA. Infinite values are invalid input;
        // float32 subnormals and signed zeros are preserved byte-for-byte.
        if (std::isinf(entry.fvalue)) throw std::invalid_argument("NPU feature value must not be infinite");
        entries.push_back({entry.index, entry.fvalue});
        previous = entry.index;
        first = false;
      }
      offsets[++next_row] = entries.size();
    }
  }
  if (next_row != state->nrows || entries.size() != info.num_nonzero_) {
    throw std::invalid_argument("NPU sparse pages do not cover the DMatrix shape");
  }
  auto device_offsets = std::make_shared<Buffer>(state->runtime->AllocateElements<uint64_t>(offsets.size()));
  auto device_entries = std::make_shared<Buffer>(state->runtime->AllocateElements<native_abi::Entry>(
      std::max<std::size_t>(1, entries.size())));
  device_offsets->Upload(offsets.data(), offsets.size() * sizeof(uint64_t));
  if (!entries.empty()) device_entries->Upload(entries.data(), entries.size() * sizeof(native_abi::Entry));
  state->offsets = std::move(device_offsets);
  state->entries = std::move(device_entries);
  state->source_pages = std::move(pages);
  state->input_nnz = info.num_nonzero_;
  state->input_ncols = CheckedU32(info.num_col_, "input feature count");
  state->input_bound = true;
}

void BindGradientMetadata(NativePrediction* state, MetaInfo const& info) {
  if (info.num_row_ != state->nrows || state->ngroups != 1 ||
      info.labels.Size() != state->nrows || info.labels.Shape(0) != state->nrows ||
      (state->nrows && info.labels.Shape(1) != 1) ||
      (info.weights_.Size() != 0 && info.weights_.Size() != state->nrows)) {
    throw std::invalid_argument("Native square error requires one label and optional weight per row");
  }
  auto labels = info.labels.Data()->ConstHostSpan();
  auto weights = info.weights_.ConstHostSpan();
  auto equal_bytes = [](auto values, std::vector<float> const& snapshot) {
    return values.size() == snapshot.size() &&
           (values.empty() || std::memcmp(values.data(), snapshot.data(), values.size_bytes()) == 0);
  };
  bool labels_changed = !state->gradient_metadata ||
                        !equal_bytes(labels, state->gradient_metadata->labels);
  bool weights_changed = !state->gradient_metadata ||
                         !equal_bytes(weights, state->gradient_metadata->weights);
  // HostDeviceVector exposes no mutation generation. Exact comparisons, not
  // pointer identity, are required for an in-place SetInfo on an existing matrix.
  // Unchanged metadata is never allocated or uploaded again.
  if (labels_changed) {
    for (auto label : labels) CheckFinite(label, "label");
  }
  if (weights_changed) {
    for (auto weight : weights) {
      CheckFinite(weight, "weight");
      if (weight < 0) throw std::invalid_argument("NPU weights must be nonnegative");
    }
  }
  std::shared_ptr<NativePrediction::GradientMetadata const> metadata = state->gradient_metadata;
  if (labels_changed || weights_changed) {
    auto snapshot = std::make_shared<NativePrediction::GradientMetadata>();
    snapshot->labels.assign(labels.begin(), labels.end());
    snapshot->weights.assign(weights.begin(), weights.end());
    metadata = std::move(snapshot);
  }
  // Publish a new immutable metadata snapshot only after all uploads succeed.
  // Failed updates are retried on the next call; no raw MetaInfo address is kept.
  auto device_labels = state->labels;
  auto device_weights = state->weights;
  if (state->nrows) {
    if (!state->gradient) state->gradient = std::make_shared<Tensor>(*state->runtime, state->nrows, ACL_FLOAT);
    if (!state->hessian) state->hessian = std::make_shared<Tensor>(*state->runtime, state->nrows, ACL_FLOAT);
    if (labels_changed) {
      device_labels = std::make_shared<Tensor>(*state->runtime, state->nrows, ACL_FLOAT);
      device_labels->Storage().Upload(metadata->labels.data(), metadata->labels.size() * sizeof(float));
    }
    if (weights_changed && !metadata->weights.empty()) {
      device_weights = std::make_shared<Tensor>(*state->runtime, state->nrows, ACL_FLOAT);
      device_weights->Storage().Upload(metadata->weights.data(), metadata->weights.size() * sizeof(float));
    }
  }
  // Changed metadata receives new device storage, preserving old sidecar labels
  // and making an interrupted upload transactional. The steady-state allocates
  // and uploads neither labels nor weights.
  state->labels = std::move(device_labels);
  state->weights = std::move(device_weights);
  // Empty snapshot weights mean unit weights and pass a null kernel pointer.
  if (labels_changed || weights_changed) ++state->metadata_uploads;
  state->gradient_metadata = std::move(metadata);
}

void ActivateForDestruction(std::shared_ptr<Runtime> const& runtime) noexcept {
  if (!runtime) return;
  try {
    runtime->Stream();
  } catch (std::exception const& e) {
    std::cerr << "NPU prediction context activation during cleanup failed: " << e.what() << '\n';
  } catch (...) {
    std::cerr << "NPU prediction context activation during cleanup failed\n";
  }
}

}  // namespace

NativePrediction::~NativePrediction() { ActivateForDestruction(runtime); }
NativeGradient::~NativeGradient() { ActivateForDestruction(runtime); }

void InitNativePrediction(Context const* ctx, MetaInfo const& info, PredictionCacheEntry* cache,
                          gbm::GBTreeModel const& model) {
  if (!cache) throw std::invalid_argument("Null prediction cache");
  CheckModel(ctx, info, model);
  auto state = cache->npu_predictions;
  if (state && state->model_identity == &model && state->device_ordinal == ctx->Device().ordinal &&
      state->nrows == info.num_row_ && state->ngroups == model.learner_model_param->OutputLength() &&
      state->nfeatures == model.learner_model_param->num_feature) {
    CheckState(ctx, *state);
    FlattenModel(state.get(), model);
    // A cache-prefix reset changes values, not the lifetime of the runtime,
    // input bindings or resident label/gradient allocations.
    InitializeOutput(state.get(), info, model);
  } else {
    cache->npu_predictions = MakeState(ctx, info, model);
  }
}

void PredictNative(Context const* ctx, DMatrix* dmat, PredictionCacheEntry* cache,
                   gbm::GBTreeModel const& model, bst_tree_t begin, bst_tree_t end) {
  if (!cache || !cache->npu_predictions || !dmat) {
    throw std::invalid_argument("NPU prediction cache/input is not initialized");
  }
  auto state = cache->npu_predictions;
  auto timer_begin = StatsEnabled() ? NowNs() : 0;
  CheckState(ctx, *state);
  CheckPredictionValid(*state);
  CheckModel(ctx, dmat->Info(), model);
  FlattenModel(state.get(), model);
  auto tree_end = end == 0 ? CheckedU32(model.trees.size(), "tree count") : end;
  if (begin > tree_end || tree_end > model.trees.size() || tree_end > state->trees.size()) {
    throw std::invalid_argument("Invalid NPU prediction tree range");
  }
  if (dmat->Info().num_row_ != state->nrows) {
    throw std::invalid_argument("NPU DMatrix row count changed for prediction cache");
  }
  ClearStatus(state.get());
  if (state->nrows == 0 || begin == tree_end) {
    if (StatsEnabled()) Stats().predict_ns += NowNs() - timer_begin;
    return;
  }
  auto const* data_ctx = dmat->Ctx();
  if (!data_ctx || (!data_ctx->Device().IsCPU() && data_ctx->Device() != ctx->Device())) {
    throw std::invalid_argument("NPU prediction input belongs to an incompatible device context");
  }
  UploadRows(state.get(), dmat);
  if (!state->offsets || !state->offsets->Data() || !state->entries || !state->entries->Data()) {
    throw std::invalid_argument("NPU prediction CSR device buffers are not initialized");
  }
  if (!state->prediction_scratch) {
    state->prediction_scratch = std::make_unique<Tensor>(
        *state->runtime, static_cast<int64_t>(state->nrows) * state->ngroups, ACL_FLOAT);
  }
  state->prediction_valid = false;
  native_abi::XgbNpuNativePredict(
      state->runtime->Stream(), state->nodes_device->Data(), state->trees_device->Data(),
      state->offsets->Data(), state->entries->Data(), state->predictions->Storage().Data(),
      state->prediction_scratch->Storage().Data(), state->status->Data(),
      state->nrows, state->ngroups, state->uploaded_tree_count, begin, tree_end);
  // The status download is ordered behind the prediction kernel and waits for
  // exactly that copy, so the former explicit stream drain is redundant.
  native_abi::Status result{};
  state->status->Download(&result, sizeof(result));
  if (result.error) {
    std::ostringstream message;
    message << "NPU prediction kernel error bits=0x" << std::hex << result.error
            << " first=0x" << result.first_error << std::dec << " row=" << result.row
            << " tree=" << result.tree << " node=" << result.node << " steps=" << result.steps
            << " device_tree={first=" << result.first_node << ",count=" << result.node_count
            << ",group=" << result.group << "} device_node={left=" << result.observed_node.left
            << ",right=" << result.observed_node.right << ",feature=" << result.observed_node.feature
            << ",threshold=" << result.observed_node.threshold << ",leaf=" << result.observed_node.leaf
            << ",is_leaf=" << +result.observed_node.is_leaf
            << ",default_left=" << +result.observed_node.default_left << "} previous=" << result.previous
            << " value=" << result.value << " result=" << result.result
            << " csr=[" << result.row_begin << ',' << result.row_end << ")"
            << " requested=[" << begin << ',' << tree_end << ")"
            << " uploaded_trees=" << state->uploaded_tree_count << " nodes=" << state->nodes.size()
            << " device_ngroups=" << result.reserved << " host_ngroups=" << state->ngroups;
    if (result.tree < state->trees.size()) {
      auto const& t = state->trees[result.tree];
      message << " host_tree={first=" << t.first_node << ",count=" << t.node_count << ",group=" << t.group << '}';
      if (result.node >= 0 && static_cast<uint32_t>(result.node) < t.node_count) {
        auto const& n = state->nodes[t.first_node + result.node];
        message << " host_node={left=" << n.left << ",right=" << n.right << ",feature=" << n.feature
                << ",threshold=" << n.threshold << ",leaf=" << n.leaf << ",is_leaf=" << +n.is_leaf
                << ",default_left=" << +n.default_left << '}';
      }
    }
    throw std::runtime_error(message.str());
  }
  state->predictions.swap(state->prediction_scratch);
  state->prediction_valid = true;
  ++state->prediction_launches;
  if (StatsEnabled()) Stats().predict_ns += NowNs() - timer_begin;
}

void ExportNativePrediction(PredictionCacheEntry* cache) {
  if (!cache || !cache->npu_predictions) {
    throw std::invalid_argument("NPU prediction cache is not initialized");
  }
  auto const& state = *cache->npu_predictions;
  auto timer_begin = StatsEnabled() ? NowNs() : 0;
  CheckPredictionValid(state);
  if (!state.runtime || !state.ops || &state.ops->GetRuntime() != state.runtime.get() ||
      (state.nrows && (!state.predictions || state.predictions->Owner() != state.runtime.get() ||
                      state.predictions->Type() != ACL_FLOAT ||
                      state.predictions->Count() != static_cast<int64_t>(state.nrows) * state.ngroups))) {
    throw std::invalid_argument("NPU prediction export has incomplete device state");
  }
  cache->predictions.Resize(static_cast<std::size_t>(state.nrows) * state.ngroups);
  if (state.nrows == 0) return;
  auto& host = cache->predictions.HostVector();
  state.predictions->Storage().Download(host.data(), host.size() * sizeof(float));
  if (StatsEnabled()) Stats().export_ns += NowNs() - timer_begin;
}

std::shared_ptr<NativeGradient> SquareErrorNativeGradient(
    Context const* ctx, PredictionCacheEntry* cache, MetaInfo const& info, int /*iteration*/,
    float scale_pos_weight) {
  if (!cache || !cache->npu_predictions) {
    throw std::invalid_argument("NPU prediction cache is not initialized");
  }
  // CUDA permits scale_pos_weight=0. It legitimately creates zero Hessians.
  if (!std::isfinite(scale_pos_weight) || scale_pos_weight < 0.0f) {
    throw std::invalid_argument("NPU scale_pos_weight must be finite and nonnegative");
  }
  auto state = cache->npu_predictions;
  auto timer_begin = StatsEnabled() ? NowNs() : 0;
  CheckState(ctx, *state);
  CheckPredictionValid(*state);
  CheckNumericFeatures(info);
  BindGradientMetadata(state.get(), info);
  ClearStatus(state.get());
  auto result = std::make_shared<NativeGradient>();
  result->prediction_state = state;
  result->metadata = state->gradient_metadata;
  result->runtime = state->runtime;
  result->ops = state->ops;
  result->nrows = state->nrows;
  result->labels = state->labels;
  result->weights = result->metadata->weights.empty() ? nullptr : state->weights;
  result->gradient = state->gradient;
  result->hessian = state->hessian;
  if (result->nrows == 0) return result;
  for (auto const& tensor : {result->labels, result->gradient, result->hessian}) {
    if (!tensor || tensor->Owner() != state->runtime.get() || tensor->Type() != ACL_FLOAT ||
        tensor->Count() != state->nrows) {
      throw std::invalid_argument("NPU square-error tensor has invalid runtime, shape or dtype");
    }
  }
  if (result->weights && (result->weights->Owner() != state->runtime.get() ||
                          result->weights->Type() != ACL_FLOAT ||
                          result->weights->Count() != state->nrows)) {
    throw std::invalid_argument("NPU square-error weights have invalid runtime, shape or dtype");
  }
  native_abi::XgbNpuNativeSquareError(
      state->runtime->Stream(), state->predictions->Storage().Data(),
      result->labels->Storage().Data(), result->weights ? result->weights->Storage().Data() : nullptr,
      result->gradient->Storage().Data(), result->hessian->Storage().Data(), state->status->Data(),
      result->nrows, scale_pos_weight);
  native_abi::Status status{};
  state->status->Download(&status, sizeof(status));
  if (status.error) throw std::overflow_error("NPU square-error arithmetic produced a nonfinite result");
  ++state->gradient_launches;
  if (StatsEnabled()) Stats().gradient_ns += NowNs() - timer_begin;
  return result;
}

}  // namespace xgboost::common::npu