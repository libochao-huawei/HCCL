set(orion_include_list
    #2.0 - checker
    ${HCCL_ALG_ROOT}/adapter_v2
    ${HCCL_ALG_ROOT}/adapter_v2/stub_ccu
    ${HCCL_ALG_ROOT}/adapter_v2/adapter_rts
    ${HCCL_ALG_V2_CODE_ROOT}/primitive
)

set(analyzer_test_list
    ${HCCL_ALG_ROOT}/checker/cmd/src/main.cc
    ${HCCL_ALG_ROOT}/checker/cmd/src/CheckerCmd.cc
    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/cmd.pb.cc
)

add_executable(hccl_alg_analyzer_test
    ${analyzer_test_list}
)

target_compile_definitions(hccl_alg_analyzer_test PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

#头文件搜索路径
target_include_directories(hccl_alg_analyzer_test PRIVATE
    ${UTILS_PUBLIC_INCLUDE_LIST}
    ${UTILS_INNER_INCLUDE_LIST}

    ${HCOMM_PUBLIC_INCLUDE_LIST}
    ${HCOMM_PLATFORM_INCLUDE_LIST}
    ${HCOMM_ALGORITHM_INCLUDE_LIST}
    ${HCOMM_COMMON_INCLUDE_LIST}
    ${HCOMM_FRAMEWORK_INCLUDE_LIST}
    ${HCOMM_LEGACY_INCLUDE_LIST}

    ${THIRD_PARTY_PATH}/json/include

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto

    # protobuf源码
    ${THIRD_PARTY_PATH}/protobuf/include

    ${ASCEND_CANN_INCLUDE_LIST}
)

target_compile_options(hccl_alg_analyzer_test PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

target_link_libraries(hccl_alg_analyzer_test PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    hccl_alg_adaption
    hccl_alg_analyzer_legacy
    -pthread
    protobuf
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    -lrt
    -ldl
)
