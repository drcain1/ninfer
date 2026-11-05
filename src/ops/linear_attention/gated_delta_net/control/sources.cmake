set(ninfer_gdn_control_target ninfer_ops)
if(NINFER_TARGET STREQUAL "v100")
  # This independent projection benefits from compilation without device linking.
  set(ninfer_gdn_control_target ninfer_ops_non_rdc)
endif()
target_sources(${ninfer_gdn_control_target} PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n96_k5120.cu"
)
unset(ninfer_gdn_control_target)

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n64_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/control.cu"
  "${CMAKE_CURRENT_LIST_DIR}/execute.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/projection.cpp"
)

include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
