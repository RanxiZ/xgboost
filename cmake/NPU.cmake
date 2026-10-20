# Copyright 2026, XGBoost Contributors
include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/NPUConfig.cmake)
include(GNUInstallDirs)

# This module is included only with USE_NPU. CPU configurations never search CANN.
set(XGBOOST_NPU_KERNEL_LIBRARY "" CACHE FILEPATH
  "Optional prebuilt production libxgboost_npu_kernels.so; otherwise compile with bisheng")
set(XGBOOST_NPU_LIBRARY_DIR "${XGBOOST_CANN_PLATFORM_ROOT}/lib64")
foreach(name opapi nnopbase ascendcl)
  # Deliberately avoid toolkit stubs (devlib) and system installations.
  unset(XGBOOST_NPU_${name}_LIBRARY CACHE)
  find_library(XGBOOST_NPU_${name}_LIBRARY NAMES ${name}
    PATHS "${XGBOOST_NPU_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
  list(APPEND XGBOOST_NPU_RUNTIME_LIBRARIES "${XGBOOST_NPU_${name}_LIBRARY}")
endforeach()

if(XGBOOST_NPU_KERNEL_LIBRARY)
  if(NOT IS_ABSOLUTE "${XGBOOST_NPU_KERNEL_LIBRARY}" OR
     NOT EXISTS "${XGBOOST_NPU_KERNEL_LIBRARY}" OR
     IS_DIRECTORY "${XGBOOST_NPU_KERNEL_LIBRARY}" OR
     NOT XGBOOST_NPU_KERNEL_LIBRARY MATCHES "\\.so(\\.[0-9.]+)?$")
    message(FATAL_ERROR "XGBOOST_NPU_KERNEL_LIBRARY must be an existing absolute production .so path")
  endif()
  set(_xgboost_npu_kernel "${XGBOOST_NPU_KERNEL_LIBRARY}")
else()
  include(ExternalProject)
  set(_xgboost_npu_kernel_build "${xgboost_BINARY_DIR}/npu-kernel")
  set(_xgboost_npu_kernel "${_xgboost_npu_kernel_build}/lib/libxgboost_npu_kernels.so")
  # ASC has its own compiler and language initialization. Do not change the host CXX compiler.
  ExternalProject_Add(xgboost_npu_kernel_build
    SOURCE_DIR "${xgboost_SOURCE_DIR}/src/common/npu/kernel"
    BINARY_DIR "${_xgboost_npu_kernel_build}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_ASC_COMPILER:FILEPATH=${XGBOOST_BISHENG}"
      "-DXGBOOST_CANN_ROOT:PATH=${XGBOOST_CANN_ROOT}"
      "-DXGBOOST_NPU_ARCH:STRING=${XGBOOST_NPU_ARCH}"
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --parallel 2
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS "${_xgboost_npu_kernel}"
    INSTALL_COMMAND "")
endif()

add_library(xgboost_npu_kernels SHARED IMPORTED GLOBAL)
set_target_properties(xgboost_npu_kernels PROPERTIES IMPORTED_LOCATION "${_xgboost_npu_kernel}")
if(TARGET xgboost_npu_kernel_build)
  add_dependencies(xgboost_npu_kernels xgboost_npu_kernel_build)
endif()

function(xgboost_enable_npu target)
  target_compile_definitions(${target} PUBLIC XGBOOST_USE_NPU=1)
  target_include_directories(${target} SYSTEM PRIVATE "${XGBOOST_CANN_ROOT}/include")
  target_link_libraries(${target} PUBLIC
    "$<BUILD_INTERFACE:xgboost_npu_kernels>" ${XGBOOST_NPU_RUNTIME_LIBRARIES} ${CMAKE_DL_LIBS})
endfunction()

# Place the production kernel beside libxgboost for both build-tree and installed loading.
function(xgboost_stage_npu_kernel target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_xgboost_npu_kernel}" "$<TARGET_FILE_DIR:${target}>/libxgboost_npu_kernels.so"
    VERBATIM)
  set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH
    "$ORIGIN" "${XGBOOST_NPU_LIBRARY_DIR}")
  set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH
    "$ORIGIN" "${XGBOOST_NPU_LIBRARY_DIR}")
  if(NOT __XGBOOST_FOR_PYTHON_WHEEL)
    install(FILES "${_xgboost_npu_kernel}" DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RENAME libxgboost_npu_kernels.so)
  endif()
endfunction()