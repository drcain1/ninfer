include("${CMAKE_CURRENT_LIST_DIR}/fp8/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/nvfp4/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q4/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q5/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q6/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q8/sources.cmake")

if(NINFER_TARGET STREQUAL "v100")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/volta/prepared.cu")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/volta/prepared.cu" PROPERTIES
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/third_party/cutlass/include"
    COMPILE_OPTIONS "--expt-relaxed-constexpr")
endif()
