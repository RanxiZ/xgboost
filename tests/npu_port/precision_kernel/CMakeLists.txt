# Copyright 2026, XGBoost Contributors
cmake_minimum_required(VERSION 3.20)
set(CANN_ROOT "/usr/local/Ascend/cann-9.1.0")
list(APPEND CMAKE_MODULE_PATH "${CANN_ROOT}/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc_modules")
# --enable-simt selects a different, pure-SIMT frontend that rejects AscendC VF headers.
set(CMAKE_ASC_ENABLE_SIMT OFF CACHE BOOL "Use the AscendC vector-core VF frontend" FORCE)
project(XgbNpuPreciseScoring LANGUAGES CXX ASC)
find_package(ASC REQUIRED)
# The CCE vector-core frontend supports __simt_vf__ and asc_vf_call.
set(CMAKE_ASC_COMPILE_OBJECT
  "<CMAKE_ASC_COMPILER> <DEFINES> <INCLUDES> -std=c++17 <FLAGS> -o <OBJECT> -c --cce-aicore-lang <SOURCE>")
ascendc_library(xgb_npu_precise SHARED ../precise_test_kernels.asc)
target_include_directories(xgb_npu_precise PRIVATE
  "${CANN_ROOT}/asc"
  "${CANN_ROOT}/asc/include"
  "${CANN_ROOT}/asc/include/basic_api"
  "${CANN_ROOT}/asc/include/interface"
  "${CANN_ROOT}/include")
