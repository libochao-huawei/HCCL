############################## hccl_aicpu_kernel ##############################
set(INCLUDE_LIST
        ${TOP_DIR}/hccl/include
        ${TOP_DIR}/hccl/src/common
        ${TOP_DIR}/hccl/src/ops
        ${TOP_DIR}/hccl/src/ops/inc
        ${TOP_DIR}/hccl/src/ops/channel
        ${TOP_DIR}/hccl/src/ops/registry
        ${TOP_DIR}/hccl/src/ops/scatter
        ${TOP_DIR}/hccl/src/ops/scatter/algo
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template
        ${TOP_DIR}/hccl/src/ops/topo
        ${TOP_DIR}/hccl/src/ops/aicpu
)
message(STATUS "ZHF--${HCOMM_CODE_ROOT}")
list(APPEND INCLUDE_LIST
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl

        # hcomm头文件
        ${HCOMM_CODE_ROOT}/src/pub_inc
        #${HCOMM_CODE_ROOT}/include/hccl

        # runtime头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/base/
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/include/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/include/external/
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # mmpa头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace/
)

add_library(hccl_aicpu_kernel SHARED
        ${TOP_DIR}/hccl/src/common/utils.cc
        ${TOP_DIR}/hccl/src/common/adapter_acl.cc
        ${TOP_DIR}/hccl/src/common/config_log.cc
        ${TOP_DIR}/hccl/src/common/sal.cc
        ${TOP_DIR}/hccl/src/common/log.cc
        ${TOP_DIR}/hccl/src/common/adapter_error_manager_pub.cc
        ${TOP_DIR}/hccl/src/common/alg_env_config.cc


        ${TOP_DIR}/hccl/src/ops/aicpu/kernel_launch.cc
        ${TOP_DIR}/hccl/src/ops/aicpu/dfx/task_exception_fun.cc
        ${TOP_DIR}/hccl/src/ops/registry/alg_template_register.cc
        ${TOP_DIR}/hccl/src/ops/registry/coll_alg_exec_registry.cc

        ${TOP_DIR}/hccl/src/ops/topo/topo.cc
        ${TOP_DIR}/hccl/src/ops/channel/channel.cc
        ${TOP_DIR}/hccl/src/ops/channel/channel_request.cc

        ${TOP_DIR}/hccl/src/ops/executor_base.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/scatter_comm_executor.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/scatter_executor_base.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/scatter_mesh_executor.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/scatter_ring_executor.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/scatter_single_executor.cc

        ${TOP_DIR}/hccl/src/ops/alg_template_base.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template/nhr_base.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template/scatter_mesh.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template/scatter_nb.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template/scatter_nhr.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template/scatter_ring_direct.cc
        ${TOP_DIR}/hccl/src/ops/scatter/algo/template/scatter_ring.cc
)

target_include_directories(hccl_aicpu_kernel PRIVATE
        ${INCLUDE_LIST}
)

target_compile_options(hccl_aicpu_kernel PRIVATE
        $<$<CONFIG:Debug>:-g>
        -O1
        -fstack-protector-all
)

target_link_options(hccl_aicpu_kernel PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
)

target_compile_definitions(hccl_aicpu_kernel PRIVATE
        -DAICPU_COMPILE
)

target_link_libraries(hccl_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
)


############################## hccl_aicpu_proxy ##############################
add_library(hccl_aicpu_proxy OBJECT aclrt_aicpu_stub.cc)

target_include_directories(hccl_aicpu_proxy PRIVATE
        # CANN包依赖
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl

        #${HCOMM_CODE_ROOT}/include/hccl

        # hccl仓依赖
        ${TOP_DIR}/hccl/src/
        ${TOP_DIR}/hccl/src/common
        ${TOP_DIR}/hccl/src/ops/inc

        ${TOP_DIR}/hccl/src/
        ${TOP_DIR}/hccl/src/common
        ${TOP_DIR}/hccl/src/ops/inc
)

target_compile_options(hccl_aicpu_proxy PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

set_target_properties(hccl_aicpu_proxy PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE  # 需要将对象库hccl_aicpu_proxy设置为生成位置无关代码（PIC），因为最终它会被链接到一个共享库中
)

#target_link_libraries(hccl_aicpu_proxy PRIVATE
#        hccl_aicpu_kernel
#)

install(TARGETS hccl_aicpu_proxy
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/hccl_vm_install
)