message(STATUS "TOP_DIR: ${TOP_DIR}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

add_library(hccl_proxy_level2 SHARED "")

add_subdirectory(src)

target_include_directories(hccl_proxy_level2 PRIVATE
        # hccl_proxy依赖的HCCL业务代码头文件, 这里的头文件有要求: 开源用户可获取
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${HCOMM_CODE_ROOT}/include
        #${HCOMM_CODE_ROOT}/include/hccl
        ${HCOMM_CODE_ROOT}/pkg_inc
        ${HCOMM_CODE_ROOT}/pkg_inc/hccl
        ${HCOMM_CODE_ROOT}/src/pub_inc
        ${HCOMM_CODE_ROOT}/src/framework/communicator/impl/independent_op/rank_graph
        ${HCOMM_CODE_ROOT}/src/framework/inc
        ${HCOMM_CODE_ROOT}/src/platform/hccp/external_depends/inc/log
)

target_include_directories(hccl_proxy_level2 PRIVATE
        # hccl_proxy依赖的Checker L2其他模块的头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/src/store/hccl_shm
        ${HCOMM_VM_ROOT_PATH}/src/ipc
        ${HCOMM_VM_ROOT_PATH}/src/runnerdb
        ${HCOMM_VM_ROOT_PATH}/src/host/hccl_vm
        ${HCOMM_VM_ROOT_PATH}/src/host/hccl_check_plugin
        ${HCOMM_VM_ROOT_PATH}/src/host/hccl_check_plugin/checker/utils
)

target_compile_options(hccl_proxy_level2 PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl_proxy_level2 PRIVATE
        dl
        rt
        log
        runnerdb
        store
        ipc
)

install(TARGETS hccl_proxy_level2
    DESTINATION "."
    # 文件的权限 (755)
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)
