# Copyright 2026, XGBoost Contributors
cmake_minimum_required(VERSION 3.20)
set(CANN_ROOT "/usr/local/Ascend/cann-9.1.0")
list(APPEND CMAKE_MODULE_PATH "${CANN_ROOT}/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc_modules")
set(CMAKE_ASC_ENABLE_SIMT OFF CACHE BOOL "Use the AscendC vector-core VF frontend" FORCE)
project(XgbNpuNativePredictionKernel LANGUAGES CXX ASC)
find_package(ASC REQUIRED)
set(CMAKE_ASC_COMPILE_OBJECT
  "<CMAKE_ASC_COMPILER> <DEFINES> <INCLUDES> -std=c++17 <FLAGS> -o <OBJECT> -c --cce-aicore-lang <SOURCE>")
ascendc_library(xgb_npu_native_prediction SHARED native_prediction_test_kernels.asc)
target_include_directories(xgb_npu_native_prediction PRIVATE
  "${CANN_ROOT}/asc"
  "${CANN_ROOT}/asc/include"
  "${CANN_ROOT}/asc/include/basic_api"
  "${CANN_ROOT}/asc/include/interface"
  "${CANN_ROOT}/include"
  "${CMAKE_CURRENT_LIST_DIR}/../../../src/common/npu")