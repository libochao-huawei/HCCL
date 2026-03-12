#add_subdirectory(topomodel)
add_subdirectory(proxy)
#add_subdirectory(microcode_transform)

target_include_directories(hccl_proxy_level2 PRIVATE
        # cann包头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu

        # hccl_vm头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/include/runnerdb
        ${HCOMM_VM_ROOT_PATH}/include/store
        ${HCOMM_VM_ROOT_PATH}/src/ipc
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd

        # hcomm头文件
        ${HCOMM_CODE_FRAMEWORK}/inc
        ${HCOMM_CODE_FRAMEWORK}/next/comms/adpt
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/pub_inc
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp/ccu_channel
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1
        ${HCOMM_CODE_PLATFORM}/hccp/inc/network

        ${TOP_DIR}/hcomm/src/legacy/common/utils
        ${TOP_DIR}/hcomm/src/legacy/common/types
        ${TOP_DIR}/hcomm/src/legacy/common
        ${TOP_DIR}/hcomm/src/legacy/common/exception
        ${TOP_DIR}/hcomm/src/legacy/common
        ${TOP_DIR}/hcomm/src/legacy/pub_inc
        ${TOP_DIR}/hcomm/src/legacy/resource
        ${TOP_DIR}/hcomm/src/legacy/resource/buffer
        ${TOP_DIR}/hcomm/src/legacy/resource/buffer/aicpu
        ${TOP_DIR}/hcomm/src/legacy/external_system
        

        # hccl_proxy模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/topomodel
        ${CMAKE_CURRENT_SOURCE_DIR}/proxy
)

target_sources(hccl_proxy_level2 PRIVATE
        # 依赖的orion仓源码文件
        #${TOP_DIR}/ace/comop/hccl/orion/common/exception/exception_defination.cc
        #${TOP_DIR}/ace/comop/hccl/orion/common/exception/hccl_exception.cc
        ${TOP_DIR}/hcomm/src/legacy/common/exception/exception_defination.cc
        ${TOP_DIR}/hcomm/src/legacy/common/exception/hccl_exception.cc
)

target_sources(hccl_proxy_level2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_device_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_context_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_memory_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_notify_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_event_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stream_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scatter_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_runner_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_ccu_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_ra_socket.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_new_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_exception_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_exec_control.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_kernel_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_proxy_common.cc
)