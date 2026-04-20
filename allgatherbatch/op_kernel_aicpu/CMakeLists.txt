set(allgatherbatch_kernel_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/diag_dl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/exec_op.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_allgather_batch_aicpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_exception_info.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allgather_batch_small_count_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_nhr_core.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_hd_stage_core.cc
)

if(KERNEL_MODE)
    target_sources(allgatherbatch_aicpu_kernel PRIVATE
        ${allgatherbatch_kernel_src_list}
    )
endif()
