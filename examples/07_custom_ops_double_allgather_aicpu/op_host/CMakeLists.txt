set_source_files_properties(
    ${HCCL_CUSTOM_OPS_ROOT}/op_kernel_aicpu/aicpu_kernel_asc.aicpu
    PROPERTIES LANGUAGE AICPU
)

set(op_host_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/double_all_gather.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/resource.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/load_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_asc.asc
    ${HCCL_CUSTOM_OPS_ROOT}/common/exec_double_all_gather.cc
    ${HCCL_CUSTOM_OPS_ROOT}/op_kernel_aicpu/aicpu_kernel_asc.aicpu
)

target_sources(hccl_custom_double_allgather PRIVATE ${op_host_src_list})
