set(HCCL_CUSTOM_OPS_AIV_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

add_library(hccl_custom_p2p_aiv SHARED)
set_target_properties(hccl_custom_p2p_aiv PROPERTIES LINKER_LANGUAGE CXX)

target_sources(hccl_custom_p2p_aiv PRIVATE
    send.cc
    recv.cc
    resource.cc
    launch_kernel.cc
)

target_include_directories(hccl_custom_p2p_aiv PRIVATE
    ${HCCL_CUSTOM_OPS_AIV_INCLUDE_LIST}
)

target_compile_options(hccl_custom_p2p_aiv PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O2
    -std=c++17
    -fstack-protector-all
)

target_compile_definitions(hccl_custom_p2p_aiv PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)

target_link_options(hccl_custom_p2p_aiv PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_directories(hccl_custom_p2p_aiv PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_custom_p2p_aiv PRIVATE
    -Wl,--no-as-needed
    hcomm
    hccl
    c_sec
    unified_dlog
    ascendc_runtime
    profapi
    ascendalog
    ascendcl
    runtime
    mmpa
    error_manager
    ascend_dump
    -Wl,--no-as-needed
)

add_dependencies(hccl_custom_p2p_aiv generate_kernel_obj)

install(TARGETS hccl_custom_p2p_aiv
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
