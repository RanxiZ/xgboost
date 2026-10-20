// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_NATIVE_PREDICTION_H_
#define XGBOOST_COMMON_NPU_NATIVE_PREDICTION_H_

#include "acl_ops.h"
#include "native_prediction_abi.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/npu.h"
#include "xgboost/predictor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace xgboost::gbm {
struct GBTreeModel;
}

namespace xgboost::common::npu {

// Device state for one prediction cache entry. CSR source pages are retained
// by shared ownership; rebinding checks page identity and storage metadata, not
// every feature value. DMatrix pages are immutable while bound to a cache.
// Runtime, predictions and reusable gradient/label tensors persist across rounds.
struct NativePrediction {
  uint32_t nrows{0};
  uint32_t ngroups{0};
  uint32_t nfeatures{0};
  int device_ordinal{-1};
  gbm::GBTreeModel const* model_identity{nullptr};
  std::shared_ptr<Runtime> runtime;
  // Ops holds a reference to *runtime. Declaration order is part of the
  // lifetime contract: tensors and buffers die before Ops, then Runtime.
  std::shared_ptr<Ops> ops;
  std::unique_ptr<Tensor> predictions;
  // Separate resident output prevents an additive launch from reading values
  // it has already written. Publish only after the device status is valid.
  std::unique_ptr<Tensor> prediction_scratch;
  std::shared_ptr<Buffer> offsets;
  std::shared_ptr<Buffer> entries;
  std::unique_ptr<Buffer> status;
  std::unique_ptr<Buffer> nodes_device;
  std::unique_ptr<Buffer> trees_device;
  struct SourcePage {
    std::shared_ptr<SparsePage const> page;
    void const* offsets{nullptr};
    void const* entries{nullptr};
    std::size_t nrows{0};
    std::size_t nnz{0};
    std::size_t base_rowid{0};
  };
  std::vector<SourcePage> source_pages;
  uint64_t input_nnz{0};
  uint32_t input_ncols{0};
  bool input_bound{false};
  // A failed prediction may have partially added trees. Only explicit cache
  // initialization makes that tensor usable again; retrying would double-add.
  bool prediction_valid{false};
  // MetaInfo has no mutation generation. Own immutable snapshots and compare
  // their bytes to detect even in-place SetInfo, without reuploading unchanged
  // labels/weights. No borrowed MetaInfo pointer is a metadata validity token.
  // SquareErrorNativeGradient snapshots validated host metadata once per
  // distinct byte content and retains it through every consuming sidecar.
  // Exact comparisons are necessary until MetaInfo exposes mutation generations.
  struct GradientMetadata {
    std::vector<float> labels;
    std::vector<float> weights;
  };
  std::shared_ptr<GradientMetadata const> gradient_metadata;
  std::shared_ptr<Tensor> labels;
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> gradient;
  std::shared_ptr<Tensor> hessian;
  uint64_t metadata_uploads{0};
  uint64_t prediction_launches{0};
  uint64_t gradient_launches{0};
  // RegTree has no mutation generation. Exact owned byte snapshots detect
  // in-place edits and allocator address reuse as well as replacement/load.
  // Unchanged trees are compared, but never revalidated or flattened again.
  struct TreeSnapshot {
    void const* identity{nullptr};
    std::vector<uint8_t> nodes;
  };
  std::vector<TreeSnapshot> tree_snapshots;
  std::vector<native_abi::Node> nodes;
  std::vector<native_abi::Tree> trees;
  uint32_t uploaded_tree_count{0};
  bool model_valid{false};
  bool prediction_has_trees{false};
  uint32_t prediction_tree_begin{0};
  uint32_t prediction_tree_end{0};
  // Diagnostic counters count successful copies/allocations, including retries.
  uint64_t model_node_upload_bytes{0};
  uint64_t model_tree_upload_bytes{0};
  uint64_t model_buffer_allocations{0};
  uint64_t model_flattened_trees{0};

  NativePrediction() = default;
  NativePrediction(NativePrediction const&) = delete;
  NativePrediction& operator=(NativePrediction const&) = delete;
  ~NativePrediction();
};

// A gradient sidecar retains the prediction state, so its runtime, stream,
// prediction tensor and input buffers remain alive while an updater consumes
// gradient/hessian. The gradient tensors are device-resident and are not
// represented by HostDeviceVector. Gradient/hessian storage is reused by the
// next gradient call on the same prediction cache; consume it before that call.
struct NativeGradient {
  std::shared_ptr<NativePrediction> prediction_state;
  std::shared_ptr<NativePrediction::GradientMetadata const> metadata;
  std::shared_ptr<Runtime> runtime;
  std::shared_ptr<Ops> ops;
  std::shared_ptr<Tensor> gradient;
  std::shared_ptr<Tensor> hessian;
  std::shared_ptr<Tensor> labels;
  std::shared_ptr<Tensor> weights;
  uint32_t nrows{0};

  NativeGradient() = default;
  NativeGradient(NativeGradient const&) = delete;
  NativeGradient& operator=(NativeGradient const&) = delete;
  ~NativeGradient();
};

// These entry points implement numerical, single-output prediction. The caller
// must select SquareErrorNativeGradient only for reg:squarederror; GBTreeModel
// and MetaInfo do not contain an objective-name discriminator. Init resets the
// prediction prefix to base margin/score while retaining compatible allocations.
void InitNativePrediction(Context const* ctx, MetaInfo const& info, PredictionCacheEntry* cache,
                          gbm::GBTreeModel const& model);
// Call BEFORE selecting a reusable cache version, even for an empty tree range.
// Returns true if model replacement/edit/shrink or failed device state required
// resetting output to base margin/score. The caller must then restart its prefix
// from layer zero (cache->version is reset here). Append-only growth returns
// false and preserves the already accumulated prefix. Init always resets output.
bool RefreshNativePrediction(Context const* ctx, MetaInfo const& info,
                             PredictionCacheEntry* cache, gbm::GBTreeModel const& model);
// Optional eager notification for load/replacement paths. Exact snapshot checks
// in Refresh/Predict also detect edits; callers that skip Predict for an empty
// delta MUST Refresh first. Invalidation preserves compatible resident storage.
void InvalidateNativePrediction(PredictionCacheEntry* cache);
// Add [begin,end) to initialized output. Ranges must be contiguous; Init starts
// an independent iteration_range. end==0 retains the existing all-trees meaning.
void PredictNative(Context const* ctx, DMatrix* dmat, PredictionCacheEntry* cache,
                   gbm::GBTreeModel const& model, bst_tree_t begin, bst_tree_t end);
void ExportNativePrediction(PredictionCacheEntry* cache);
std::shared_ptr<NativeGradient> SquareErrorNativeGradient(
    Context const* ctx, PredictionCacheEntry* cache, MetaInfo const& info, int iteration,
    float scale_pos_weight = 1.0f);

}  // namespace xgboost::common::npu
#endif  // XGBOOST_COMMON_NPU_NATIVE_PREDICTION_H_