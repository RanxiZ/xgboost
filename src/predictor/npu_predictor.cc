// Copyright 2026, XGBoost Contributors
#include "xgboost/predictor.h"

#include "../common/npu/native_prediction.h"
#include "xgboost/logging.h"

#include <stdexcept>

namespace xgboost::predictor {

// The legacy Predictor interface cannot attach a PredictionCacheEntry sidecar
// in InitOutPredictions. Parent routing must call InitNativePrediction directly;
// this adapter is intentionally limited to the sidecar-bound PredictBatch path.
class NpuPredictor final : public xgboost::Predictor {
 public:
  explicit NpuPredictor(Context const* ctx) : Predictor{ctx} {}

  void Configure(Args const&) override {}

  void InitOutPredictions(MetaInfo const&, HostDeviceVector<float>*,
                          gbm::GBTreeModel const&) const override {
    throw std::logic_error("NPU InitOutPredictions cannot bind a native sidecar; "
                           "call common::npu::InitNativePrediction");
  }

  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* out_preds,
                    gbm::GBTreeModel const& model, bst_tree_t tree_begin,
                    bst_tree_t tree_end,
                    std::vector<float> const* tree_weights_override = nullptr) const override {
    if (tree_weights_override != nullptr) {
      throw std::logic_error("NPU native prediction does not support temporary tree weights");
    }
    common::npu::PredictNative(ctx_, dmat, out_preds, model, tree_begin, tree_end);
  }

  bool InplacePredict(std::shared_ptr<DMatrix>, gbm::GBTreeModel const&, float,
                      PredictionCacheEntry*, bst_tree_t, bst_tree_t) const override {
    throw std::logic_error("NPU native prediction requires an ordinary DMatrix and sidecar cache");
  }

  void PredictLeaf(DMatrix*, HostDeviceVector<float>*, gbm::GBTreeModel const&,
                   bst_tree_t) const override {
    throw std::logic_error("NPU native prediction does not support leaf prediction");
  }

  void PredictFromLeafIds(common::Span<HostDeviceVector<bst_node_t> const>,
                          common::Span<RegTree const*>,
                          linalg::MatrixView<float>) const override {
    throw std::logic_error("NPU native prediction does not support leaf-id prediction");
  }

  void PredictContribution(DMatrix*, HostDeviceVector<float>*, gbm::GBTreeModel const&,
                           bst_tree_t, bool, int, unsigned) const override {
    throw std::logic_error("NPU native prediction does not support contributions");
  }

  void PredictInteractionContributions(DMatrix*, HostDeviceVector<float>*,
                                       gbm::GBTreeModel const&, bst_tree_t,
                                       bool) const override {
    throw std::logic_error("NPU native prediction does not support interactions");
  }
};

XGBOOST_REGISTER_PREDICTOR(NpuPredictor, "npu_predictor")
    .describe("Native Ascend resident numerical tree prediction.")
    .set_body([](Context const* ctx) { return new NpuPredictor{ctx}; });

}  // namespace xgboost::predictor