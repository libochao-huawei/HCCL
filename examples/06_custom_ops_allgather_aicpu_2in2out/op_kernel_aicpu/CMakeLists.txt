# 这里单独放 AICPU kernel 侧源码列表。
# 以后如果你把 allgather 改成 allreduce，通常只需要替换这里引用的
# device 侧协议实现文件，而不用改根目录里 .so -> tar.gz -> 签名 的外层链路。
set(op_kernel_aicpu_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_kernel.cc
    ${HCCL_CUSTOM_OPS_ROOT}/common/exec_op.cc
)

if(KERNEL_MODE)
    target_sources(allgather_2in2out_aicpu_kernel PRIVATE
        ${op_kernel_aicpu_src_list}
    )
endif()
