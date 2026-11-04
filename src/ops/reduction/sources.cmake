set(ninfer_logprobs_target ninfer_ops)
if(NINFER_TARGET STREQUAL "v100")
  # RDC coalesces helper-local shared symbols across kernels and inflates their
  # allocation beyond Volta's 48 KiB static limit. These host-launched units
  # have no external device calls; keep their shared storage local to each TU.
  set(ninfer_logprobs_target ninfer_ops_non_rdc)
endif()
target_sources(${ninfer_logprobs_target} PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vocabulary_logprobs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/target_logprobs.cu"
)
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/argmax.cu")
unset(ninfer_logprobs_target)
