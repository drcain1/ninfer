# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops SHARED)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_library(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)
if(NINFER_TARGET STREQUAL "v100")
  # Some public Ops are self-contained non-RDC units. Retain their entry points
  # even when no other object inside this shared library calls them.
  target_link_libraries(ninfer_ops PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,ninfer_ops_non_rdc>")
  # CMake invokes the host linker directly, bypassing NVCC's large-fatbin
  # layout. Keep CUDA payloads after ordinary host code/data so x86-64
  # relative references remain in range even when device images exceed 2 GiB.
  set(ninfer_ops_linker_script "${PROJECT_SOURCE_DIR}/cmake/CudaFatbin.ld")
  target_link_options(ninfer_ops PRIVATE
    "$<HOST_LINK:LINKER:-T,${ninfer_ops_linker_script}>")
  set_property(TARGET ninfer_ops APPEND PROPERTY LINK_DEPENDS "${ninfer_ops_linker_script}")
  unset(ninfer_ops_linker_script)
else()
  target_link_libraries(ninfer_ops PRIVATE ninfer_ops_non_rdc)
endif()

# Non-RDC kernels keep instruction and register-allocation choices within one translation unit.
# This private PIC archive joins the shared library through host launchers.
add_library(ninfer_ops_non_rdc STATIC)
ninfer_internal_includes(ninfer_ops_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_ops_non_rdc)
target_link_libraries(ninfer_ops_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

target_sources(ninfer_ops PRIVATE weight_input.cpp)
include("${CMAKE_CURRENT_LIST_DIR}/collective/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/common/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/elementwise/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/mhc/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/sources.cmake")
