set(include_list
    # hccl inc目录
    # ${TOP_DIR}/ace/comop/inc
    # ${TOP_DIR}/ace/comop/inc/hccl
    # ${TOP_DIR}/runtime/include/external
    # ${TOP_DIR}/runtime/pkg_inc/runtime
    # ${TOP_DIR}/runtime/pkg_inc/
    # ${TOP_DIR}/runtime/pkg_inc/profiling
    # ${TOP_DIR}/ace/npuruntime/inc
    # ${TOP_DIR}/ace/npuruntime/inc/runtime
    # ${TOP_DIR}/ace/npuruntime/inc/external
    # ${TOP_DIR}/ace/npuruntime/acl/inc/external/
    # ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl
    # ${TOP_DIR}/ace/comop/inc/external
    # ${TOP_DIR}/ace/comop/inc/external/hccl
    # ${TOP_DIR}/drivers/network/inc/network
    # 
    # ${TOP_DIR}/abl/msprof/inc
    # ${TOP_DIR}/abl/slog/inc/toolchain
    # ${TOP_DIR}/open_source/json/include

    ${UTILS_DIR}/adapter_v1/aiv

    ${legacy_common_include_list}
    ${legacy_framework_include_list}
    ${legacy_platform_include_list}
    ${legacy_service_include_list}
    ${checker_include_list}

    ${HCOMM_CODE_LEGACY}/common

    ${HCOMM_PLATFORM_INCLUDE_LIST}

    ${ASCEND_CANN_INCLUDE_LIST}

    // 第三方
    ${THIRD_PARTY_PATH}/json/include
    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
    ${THIRD_PARTY_PATH}/protobuf/include

    #${TOP_DIR}/inc/external/
    #${TOP_DIR}/abl/adump/inc/adump
    #${TOP_DIR}/metadef/inc/external/
    #${TOP_DIR}/metadef/pkg_inc
)

file(GLOB_RECURSE ALG_CCU_CONTEXT_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_ccu_context/*.cc")
file(GLOB_RECURSE ALG_CCU_TEMPLATE_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/ccu_alg_template/*.cc")
file(GLOB_RECURSE ALG_AIV_TEMPLATE_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/aiv_alg_template/*.cc")
file(GLOB_RECURSE ALG_PRIM_SRC "${HCOMM_LEGACY_SERVICE}/primitive/*.cc")
file(GLOB_RECURSE ALG_PRIM_EXECUTOR_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_executor/prim_alg_executor/*.cc")
file(GLOB_RECURSE ALG_PRIM_TEMPLATE_SRC "${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/prim_alg_template/*.cc")

file(GLOB_RECURSE ALG_NEXT_SRC "${HCOMM_LEGACY_SERVICE}/alg/*.cc")

# 排除不想编译的文件
file(GLOB_RECURSE EXCLUDE_BATCH_SEND_RECV_SRC "${HCOMM_LEGACY_SERVICE}/*/ins_batchSendRecv_executor.cc")
file(GLOB_RECURSE EXCLUDE_DEVICE_SRC "${HCOMM_LEGACY_SERVICE}/*/device_utils.cc")
list(REMOVE_ITEM ALG_NEXT_SRC ${EXCLUDE_BATCH_SEND_RECV_SRC})
list(REMOVE_ITEM ALG_NEXT_SRC ${EXCLUDE_DEVICE_SRC})
file(GLOB_RECURSE PRIMITIVE_SRC "${HCOMM_LEGACY_SERVICE}/primitive/*.cc")

# 被测代码编译
set(david_to_test_list
    #被测代码
    ${HCOMM_LEGACY_SERVICE}/instruction/instruction.cc
    ${HCOMM_LEGACY_SERVICE}/coll_operator.cc
    ${ALG_CCU_CONTEXT_SRC}
    ${ALG_CCU_TEMPLATE_SRC}
    ${ALG_AIV_TEMPLATE_SRC}
    ${ALG_PRIM_SRC}
    ${ALG_PRIM_EXECUTOR_SRC}
    ${ALG_PRIM_TEMPLATE_SRC}
    ${ALG_NEXT_SRC}
    ${PRIMITIVE_SRC}
    ${UTILS_DIR}/common/acl_stub/acl_stub.cc
)

add_library(hccl_orion_alg_test SHARED ${david_to_test_list})

#头文件搜索路径
target_include_directories(hccl_orion_alg_test PRIVATE
    ${include_list}
)

target_compile_definitions(hccl_orion_alg_test PRIVATE
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_orion_alg_test PRIVATE
    -U_FORTIFY_SOURCE
    -O0
    -g
    -pipe
)

target_link_libraries(hccl_orion_alg_test PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ${THIRD_PARTY_PATH}/protobuf/lib64/libprotobuf.so
    -lrt
    -ldl
)

set(test_case_list
)

add_subdirectory(legacy_alg_testcase)
add_subdirectory(aicpu_2d_testcase)
# add_subdirectory(aiv_1d_testcase)
add_subdirectory(ccu_1d_testcase_part1)
add_subdirectory(ccu_1d_testcase_part2)
add_subdirectory(ccu_reduce_testcase)
add_subdirectory(ccu_1d_hf16p_testcase)
# 2D用例跳过
add_subdirectory(ccu_2d_testcase)
add_subdirectory(function_ut_testcase)