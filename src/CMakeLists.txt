file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/cli_main.cc)

# NPU host code includes CANN headers and references the separately compiled ASC library.
# Keep headers visible to IDEs, but do not compile these sources in the default CPU build.
if(NOT USE_NPU)
  list(FILTER CPU_SOURCES EXCLUDE REGEX "/common/npu/.*\\.cc$")
  list(REMOVE_ITEM CPU_SOURCES
    ${xgboost_SOURCE_DIR}/src/tree/updater_npu_hist.cc
    ${xgboost_SOURCE_DIR}/src/predictor/npu_predictor.cc)
endif()

if(PLUGIN_SYCL)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/regression_obj.cc)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/hinge.cc)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/quantile_obj.cc)
  list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/objective/multiclass_obj.cc)
endif()

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objxgboost OBJECT)
target_sources(objxgboost PRIVATE ${CPU_SOURCES})
if(USE_NPU)
  xgboost_enable_npu(objxgboost)
endif()
# Skip files with factory object
set_source_files_properties(
  predictor/predictor.cc gbm/gbm.cc tree/tree_updater.cc metric/metric.cc objective/objective.cc
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu)
  target_sources(objxgboost PRIVATE ${CUDA_SOURCES})
endif()

if(PLUGIN_SYCL)
  target_compile_definitions(objxgboost PRIVATE -DXGBOOST_USE_SYCL=1)
endif()

target_include_directories(objxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include)

if(LOG_CAPI_INVOCATION)
  target_compile_definitions(objxgboost PRIVATE -DLOG_CAPI_INVOCATION=1)
endif()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")

#-- End object library
