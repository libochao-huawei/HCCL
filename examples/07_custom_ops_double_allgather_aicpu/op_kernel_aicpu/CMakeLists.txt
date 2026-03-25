set(op_kernel_aicpu_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_kernel.cc
    ${HCCL_CUSTOM_OPS_ROOT}/common/exec_double_all_gather.cc
)

if(KERNEL_MODE)
    target_sources(double_allgather_aicpu_kernel PRIVATE
        ${op_kernel_aicpu_src_list}
    )
endif()
