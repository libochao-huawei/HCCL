# 如果根 CMake 已经准备好了统一 include 列表，这里直接复用。
# 这样 Host 库和 AICPU kernel 用的是同一套头文件搜索路径，后面维护更省心。
if(NOT HCCL_CUSTOM_OPS_INCLUDE_LIST)
    set(HCCL_CUSTOM_OPS_INCLUDE_LIST
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel_aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
        ${ASCEND_CANN_PACKAGE_PATH}/asc/include/aicpu_api
    )
endif()

add_library(hccl_custom_allgather_2in2out SHARED)
set_target_properties(hccl_custom_allgather_2in2out PROPERTIES LINKER_LANGUAGE CXX)

target_sources(hccl_custom_allgather_2in2out PRIVATE
    all_gather_2in2out.cc
    load_kernel.cc
    launch_kernel.cc
    resource.cc
    ../common/topology.cc
    ../common/small_count_policy.cc
)

target_include_directories(hccl_custom_allgather_2in2out PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)

target_compile_options(hccl_custom_allgather_2in2out PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O2
    -std=c++17
    -fstack-protector-all
)

target_compile_definitions(hccl_custom_allgather_2in2out PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)

target_link_options(hccl_custom_allgather_2in2out PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_directories(hccl_custom_allgather_2in2out PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_custom_allgather_2in2out PRIVATE
    -Wl,--no-as-needed
    hcomm
    c_sec
    unified_dlog
    ascendalog
    ascendcl
    runtime
    mmpa
    error_manager
    ascend_dump
    -Wl,--no-as-needed
)

install(TARGETS hccl_custom_allgather_2in2out
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
