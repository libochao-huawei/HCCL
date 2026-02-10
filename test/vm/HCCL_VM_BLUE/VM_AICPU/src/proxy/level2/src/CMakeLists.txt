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
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/tsd
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace

        # hccl_vm头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/include/runnerdb
        ${HCOMM_VM_ROOT_PATH}/include/store
        ${HCOMM_VM_ROOT_PATH}/src/ipc
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd

        #hal
        ${TOP_DIR}/inc/driver
        
        # runtime 
        ${TOP_DIR}/ace/npuruntime/inc/runtime/rts
        ${TOP_DIR}/ace/npuruntime/acl/runtime

        # HCCP头文件
        ${TOP_DIR}/drivers/network/inc/network
        ${TOP_DIR}/drivers/network/orion/hcomm_dev/inc/network

        # orion仓头文件
        ${TOP_DIR}/ace/comop/hccl/orion/common
        ${TOP_DIR}/ace/comop/hccl/orion/common/utils
        ${TOP_DIR}/ace/comop/hccl/orion/common/types
        ${TOP_DIR}/ace/comop/hccl/orion/common/exception


        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform
        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/pub_inc
        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource
        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer
        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer/aicpu
        ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/external_system

        # hcomm头文件
	${HCOMM_CODE_FRAMEWORK}/inc
        ${HCOMM_CODE_FRAMEWORK}/src/pub_inc
        ${HCOMM_CODE_FRAMEWORK}/src/framework/inc
        ${HCOMM_CODE_PLATFORM}/hccp/
        ${HCOMM_CODE_PLATFORM}/hccp/incRaInit
        ${HCOMM_CODE_PLATFORM}/common
        ${HCOMM_CODE_FRAMEWORK}/next/comms/adapter
	${HCOMM_CODE_FRAMEWORK}/next/comms/adpt
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/pub_inc
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp/ccu_channel
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe
        ${HCOMM_CODE_FRAMEWORK}/next/comms/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1
        ${HCOMM_CODE_PLATFORM}/hccp/inc/network
        ${TOP_DIR}/hcomm/src/legacy/common
        ${TOP_DIR}/hcomm/src/legacy/common/exception
        ${TOP_DIR}/hcomm/src/legacy/common/types
        ${TOP_DIR}/hcomm/src/legacy/common/utils       

        # hccl_proxy模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/topomodel
        ${CMAKE_CURRENT_SOURCE_DIR}/proxy
)

target_sources(hccl_proxy_level2 PRIVATE
        # 依赖的orion仓源码文件
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
	${CMAKE_CURRENT_SOURCE_DIR}/hccp_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_hal_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter_rts_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tsd_client_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aprofiling_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stub.cc
)