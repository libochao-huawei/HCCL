add_subdirectory(communicator)
add_subdirectory(topomodel)
add_subdirectory(proxy)

target_include_directories(hccl_proxy_level1 PRIVATE
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd
        ${HCOMM_VM_ROOT_PATH}/utils/base
        ${HCOMM_VM_ROOT_PATH}/utils/exception

        # hccl_proxy模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/topomodel
        ${CMAKE_CURRENT_SOURCE_DIR}/common/exception
        ${CMAKE_CURRENT_SOURCE_DIR}/communicator
        ${CMAKE_CURRENT_SOURCE_DIR}/proxy
)

target_sources(hccl_proxy_level1 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_ctrl_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_data_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_old_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scatter_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_stub.cc
)