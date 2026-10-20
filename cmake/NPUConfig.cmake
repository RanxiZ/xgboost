# Copyright 2026, XGBoost Contributors
# Shared validation for the host build and the isolated AscendC project.
include_guard(GLOBAL)

set(XGBOOST_CANN_ROOT "/usr/local/Ascend/cann-9.1.0" CACHE PATH "Pinned CANN 9.1.0 toolkit")
set(XGBOOST_NPU_ARCH "dav-3510" CACHE STRING "Ascend architecture (dav-3510 only)")
if(NOT XGBOOST_CANN_ROOT STREQUAL "/usr/local/Ascend/cann-9.1.0")
  message(FATAL_ERROR "USE_NPU requires CANN at /usr/local/Ascend/cann-9.1.0")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "USE_NPU currently supports native Linux x86_64 builds only")
endif()
if(NOT XGBOOST_NPU_ARCH STREQUAL "dav-3510")
  message(FATAL_ERROR "USE_NPU supports only XGBOOST_NPU_ARCH=dav-3510")
endif()
if(CMAKE_VERSION VERSION_LESS "3.20")
  message(FATAL_ERROR "USE_NPU requires CMake 3.20 or newer for the ASC compiler project")
endif()
set(XGBOOST_CANN_PLATFORM_ROOT "${XGBOOST_CANN_ROOT}/x86_64-linux")
set(XGBOOST_ASC_MODULE_PATH
  "${XGBOOST_CANN_PLATFORM_ROOT}/tikcpp/ascendc_kernel_cmake/asc_modules")
set(XGBOOST_BISHENG "${XGBOOST_CANN_ROOT}/tools/bisheng_compiler/bin/bisheng")
foreach(required
    "${XGBOOST_ASC_MODULE_PATH}/CMakeDetermineASCCompiler.cmake"
    "${XGBOOST_BISHENG}"
    "${XGBOOST_CANN_ROOT}/include/acl/acl.h")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Incomplete pinned CANN 9.1.0 installation: missing ${required}")
  endif()
endforeach()