add_library(hccl_proxy_level1 SHARED "")
add_subdirectory(src)
add_subdirectory(src/aicpu_kernel)

target_include_directories(hccl_proxy_level1 PRIVATE
        # hccl_proxy依赖的HCCL业务代码头文件, 这里的头文件有要求: 开源用户可获取
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        
        ${HCOMM_CODE_ROOT}/include
        #${HCOMM_CODE_ROOT}/include/hccl
        ${HCOMM_CODE_ROOT}/pkg_inc
        ${HCOMM_CODE_ROOT}/pkg_inc/hccl
        ${HCOMM_CODE_ROOT}/src/pub_inc
        ${HCOMM_CODE_ROOT}/src/framework/communicator/impl/independent_op/rank_graph
        ${HCOMM_CODE_ROOT}/src/framework/inc
)

target_include_directories(hccl_proxy_level1 PRIVATE
        # hccl_proxy依赖的Checker L2其他模块的头文件
        ${HCOMM_VM_ROOT_PATH}/hccl_common
        ${HCOMM_VM_ROOT_PATH}/store/hccl_shm
        ${HCOMM_VM_ROOT_PATH}/hccl_ipc
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_vm
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_check_plugin
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_check_plugin/checker/utils
)

target_compile_options(hccl_proxy_level1 PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl_proxy_level1 PRIVATE
        store
        hccl_utils
        $<TARGET_OBJECTS:hccl_aicpu_proxy>
        hccl_aicpu_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64/libc_sec.so
)

install(TARGETS hccl_proxy_level1
    DESTINATION "."
    # 文件的权限 (755)
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)