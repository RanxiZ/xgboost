# Copyright 2026, XGBoost Contributors
cmake_minimum_required(VERSION 3.20)
# Set CANN's environment before its custom language compiler detection runs.
set(XGBOOST_CANN_ROOT "/usr/local/Ascend/cann-9.1.0" CACHE PATH "Pinned CANN 9.1.0 toolkit")
set(ENV{ASCEND_HOME_PATH} "${XGBOOST_CANN_ROOT}")
list(PREPEND CMAKE_MODULE_PATH
  "${XGBOOST_CANN_ROOT}/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc_modules")
set(CMAKE_ASC_ENABLE_SIMT OFF CACHE BOOL "Use the AscendC vector-core VF frontend" FORCE)
set(XGBOOST_NPU_ARCH "dav-3510" CACHE STRING "Ascend architecture")
set(CMAKE_ASC_ARCHITECTURES "${XGBOOST_NPU_ARCH}" CACHE STRING "ASC architecture" FORCE)
project(XGBoostNpuKernels LANGUAGES CXX ASC)
include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/NPUConfig.cmake)
find_package(ASC REQUIRED)
# The default pure-SIMT frontend rejects VF headers; use the validated vector-core frontend.
set(CMAKE_ASC_COMPILE_OBJECT
  "<CMAKE_ASC_COMPILER> <DEFINES> <INCLUDES> -std=c++17 <FLAGS> -o <OBJECT> -c --cce-aicore-lang <SOURCE>")
ascendc_library(xgboost_npu_kernels SHARED production_kernels.asc)
set_target_properties(xgboost_npu_kernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_BINARY_DIR}/lib>"
  BUILD_RPATH "${XGBOOST_CANN_PLATFORM_ROOT}/lib64")
# CANN's custom ASC linker rules do not emit CMake's automatic RPATH flags.
# The kernel itself needs this path for indirect dependencies such as libprofapi.
target_link_options(xgboost_npu_kernels PRIVATE
  "-Wl,-rpath,${XGBOOST_CANN_PLATFORM_ROOT}/lib64")
target_include_directories(xgboost_npu_kernels PRIVATE
  "${XGBOOST_CANN_ROOT}/asc"
  "${XGBOOST_CANN_ROOT}/asc/include"
  "${XGBOOST_CANN_ROOT}/asc/include/basic_api"
  "${XGBOOST_CANN_ROOT}/asc/include/interface"
  "${XGBOOST_CANN_ROOT}/include"
  "${CMAKE_CURRENT_LIST_DIR}/..")