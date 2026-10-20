// Copyright 2026, XGBoost Contributors
#ifndef XGBOOST_COMMON_NPU_NATIVE_TRAINING_ABI_H_
#define XGBOOST_COMMON_NPU_NATIVE_TRAINING_ABI_H_
#include <cstdint>
#include "native_prediction_abi.h"

// Device pointer table contains one int64[nrows] column per feature. Missing
// values map to -1; other bins are feature-local upper_bound(cuts, value),
// clamped to the final bin, exactly HistogramCuts::SearchBin.
extern "C" void XgbNpuNativeBinRows(void* stream, void const* offsets, void const* entries,
                                     void const* cut_ptrs, void const* cuts,
                                     void const* columns, uint32_t nrows, uint32_t nfeatures);
// CCCL v3.3.3 cuda::std::philox4x64, discard(row), float canonical
// distribution, and inclusive CUDA SamplingInfo comparison. Both float32
// channels are copied as bits or replaced by positive zero.
extern "C" void XgbNpuNativeSampleRows(void* stream, void const* gradient, void const* hessian,
                                        void* sampled_gradient, void* sampled_hessian,
                                        uint32_t nrows, uint64_t seed, float subsample);
#endif  // XGBOOST_COMMON_NPU_NATIVE_TRAINING_ABI_H_