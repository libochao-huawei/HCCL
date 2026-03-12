set(include_list
    # 测试框架的代码
    ${UTILS_DIR}/inc
    ${UTILS_DIR}/checker
    ${UTILS_DIR}/checker/common/types
    ${UTILS_DIR}/checker/rank_info_recorder
    ${UTILS_DIR}/checker/mem_layout
    ${UTILS_DIR}/checker/link_type_recorder
    ${UTILS_DIR}/checker/task_collector
    ${UTILS_DIR}/checker/semantics_check
    ${UTILS_DIR}/checker/singletask_check
    ${UTILS_DIR}/checker/ui/data_dumper
    ${UTILS_DIR}/checker/param_trans
    ${UTILS_DIR}/checker/mem_conflict_check
    ${UTILS_DIR}/adapter_v1/aiv
    ${UTILS_DIR}/adapter_v1

    ${UTILS_DIR}/adapter_v2

    ${HCOMM_CODE_LEGACY}/common/types
    ${HCOMM_CODE_LEGACY}/common/utils
    ${HCOMM_CODE_LEGACY}/common/exception
    ${HCOMM_CODE_LEGACY}/include
    ${HCOMM_ROOT_PATH}/src/pub_inc
    ${HCOMM_CODE_PLATFORM}/common

    ${HCOMM_CODE_LEGACY}/service/collective/primitive

    # hccl inc目录
    # ${TOP_DIR}/ace/comop/inc
    # ${TOP_DIR}/ace/comop/inc/hccl
    # ${TOP_DIR}/ace/comop/inc/external
    # ${TOP_DIR}/ace/comop/inc/external/hccl

    # inc目录下头文件
    # ${TOP_DIR}/inc
    # ${TOP_DIR}/inc/aicpu
    # ${TOP_DIR}/inc/driver
    # ${TOP_DIR}/metadef/inc/common/util/error_manager
    # ${TOP_DIR}/metadef/inc
    # ${TOP_DIR}/metadef/pkg_inc
    # ${TOP_DIR}/ace/npuruntime/inc
    # ${TOP_DIR}/ace/npuruntime/inc/external
    # ${TOP_DIR}/ace/npuruntime/inc/runtime
    # ${TOP_DIR}/abl/msprof/inc
    # ${TOP_DIR}/abl/atrace/inc/utrace
    # ${TOP_DIR}/abl/mmpa/inc/mmpa
    # ${TOP_DIR}/drivers/network/inc
    # ${TOP_DIR}/abl/mmpa/inc/mmpa

    ${ASCEND_CANN_INCLUDE_LIST}

    ${HCOMM_CODE_ALGORITHM}/base/inc
    ${HCOMM_CODE_ALGORITHM}/impl
    ${HCOMM_CODE_ALGORITHM}/impl/task
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor
    ${HCOMM_CODE_ALGORITHM}/impl/resource_manager
    ${HCOMM_CODE_ALGORITHM}/base/alg_template
    ${HCOMM_CODE_ALGORITHM}/base/communicator
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/component

    ${HCOMM_CODE_ALGORITHM}/impl/inc
    ${HCOMM_CODE_ALGORITHM}/impl/legacy
    ${HCOMM_CODE_ALGORITHM}/base/communicator/legacy

    ${HCOMM_CODE_COMMON}/debug/profiling
    ${HCOMM_CODE_COMMON}/debug/profiling/inc
    ${HCOMM_CODE_COMMON}/debug/profiling/inc/host
    ${HCOMM_CODE_COMMON}/debug/config

    # framework目录下的头文件
    ${HCOMM_CODE_FRAMEWORK}/inc
    ${HCOMM_CODE_FRAMEWORK}/common/src
    ${HCOMM_CODE_FRAMEWORK}/common/src/host
    ${HCOMM_CODE_FRAMEWORK}/common/src/config
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl

    # json 目录的头文件
    ${THIRD_PARTY_PATH}/json/include

    ${THIRD_PARTY_PATH}/protobuf/include
    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto
)

set(checker_to_test_list

)

add_library(hccl_alg_analyzer_legacy SHARED ${checker_to_test_list})

add_subdirectory(checker)
add_subdirectory(adapter_v2)

# 需要针对protobuf自动编译生成的文件设置GENERATED属性，否则会编译报错
set_property(SOURCE ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/analysis_result.pb.cc PROPERTY GENERATED 1)

#头文件搜索路径
target_include_directories(hccl_alg_analyzer_legacy PRIVATE
    ${include_list}
)

target_compile_definitions(hccl_alg_analyzer_legacy PRIVATE
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
    -DHCCL_ALG_ANALYZER_DAVID   #checker v2相关的编译宏
)

target_compile_options(hccl_alg_analyzer_legacy PRIVATE
    -Wall
    -Wfloat-equal
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

target_link_libraries(hccl_alg_analyzer_legacy PRIVATE
    hccl_alg_adaption_v2
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ${THIRD_PARTY_PATH}/protobuf/lib64/libprotobuf.so
    -lrt
    -ldl
)