set(HCCL_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/whole)
set(HCCL_OPEN_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/open_source)
set(HCCL_ALG_ROOT ${TOP_DIR}/asl/devtools/hccl_tools/hccl_alg_analyzer)
set(HCCL_OPEN_CODE_ALGORITHM ${HCCL_OPEN_CODE_ROOT}/src/domain/collective_communication/algorithm)
set(HCCL_OPEN_CODE_FRAMEWORK ${HCCL_OPEN_CODE_ROOT}/src/domain/collective_communication/framework)
set(HCCL_OPEN_CODE_COMMON ${HCCL_OPEN_CODE_ROOT}/src/domain/collective_communication/common)
set(HCCL_TOOLS_INSIGHT ${TOP_DIR}/asl/devtools/hccl_tools/hccl_insight)
set(HCCL_ANALYZER_LIB_ROOT ${TOP_DIR}/asl/devtools/hccl_tools/hccl_alg_analyzer/pub_inc)
set(HCCL_ALG_ANALYZER_LIB_ROOT ${TOP_DIR}/llt/ace/comop/hccl)

set(include_list
    ${HCCL_TOOLS_INSIGHT}/cpp_engine/src

    #用于本地跑llt
    ${HCCL_ALG_ANALYZER_LIB_ROOT}/hccl_alg

    # 测试框架的代码
    ${HCCL_ANALYZER_LIB_ROOT}/inc
    ${HCCL_ANALYZER_LIB_ROOT}/checker
    ${HCCL_ALG_ROOT}/checker/param_trans

    # hccl inc目录
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ace/comop/inc/hccl
    ${TOP_DIR}/ace/comop/inc/external
    ${TOP_DIR}/ace/comop/inc/external/hccl

    # inc目录下头文件
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/aicpu
    ${TOP_DIR}/inc/driver
    ${TOP_DIR}/metadef/inc/common/util/error_manager
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/pkg_inc
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/ace/npuruntime/inc/runtime
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/abl/atrace/inc/utrace
    ${TOP_DIR}/drivers/network/inc

    # 添加hccl目录
    ${HCCL_OPEN_CODE_ALGORITHM}/pub_inc
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_all_gather
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_all_reduce
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_alltoallv
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_alltoall
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_reduce_scatter
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_reduce
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_scatter
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_broadcast
    ${HCCL_CODE_ROOT}/hccl/pub_inc
    ${HCCL_CODE_ROOT}/hccl/pub_inc/inner
    ${HCCL_CODE_ROOT}/hccl/pub_inc/new

    ${HCCL_OPEN_CODE_ALGORITHM}/base/inc
    ${HCCL_OPEN_CODE_ALGORITHM}/impl
    ${HCCL_OPEN_CODE_ALGORITHM}/impl/task
    ${HCCL_OPEN_CODE_ALGORITHM}/impl/coll_executor
    ${HCCL_OPEN_CODE_ALGORITHM}/impl/resource_manager
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template
    ${HCCL_OPEN_CODE_ALGORITHM}/base/communicator
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/component/

    ${HCCL_OPEN_CODE_ALGORITHM}/impl/inc
    ${HCCL_OPEN_CODE_ALGORITHM}/impl/legacy
    ${HCCL_OPEN_CODE_ALGORITHM}/base/communicator/legacy

    ${HCCL_OPEN_CODE_COMMON}/debug/profiling
    ${HCCL_OPEN_CODE_COMMON}/debug/profiling/inc
    ${HCCL_OPEN_CODE_COMMON}/debug/profiling/inc/host
    ${HCCL_OPEN_CODE_COMMON}/debug/config

    # platform目录下的头文件
    ${HCCL_CODE_ROOT}/hccl/platform/inc
    ${HCCL_CODE_ROOT}/hccl/platform/inc/adapter
    ${HCCL_CODE_ROOT}/hccl/platform/common
    ${HCCL_CODE_ROOT}/hccl/platform/resource/stream
    ${HCCL_CODE_ROOT}/hccl/platform/resource/transport

    # framework目录下的头文件
    ${HCCL_OPEN_CODE_FRAMEWORK}/inc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/host
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo
    ${HCCL_OPEN_CODE_FRAMEWORK}/communicator/impl

    # libc_sec目录下的头文件
    ${TOP_DIR}/abl/slog/inc/toolchain

    # json 目录的头文件
    ${TOP_DIR}/open_source/json/include

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto

)

set(src_to_test_list
    ${HCCL_TOOLS_INSIGHT}/cpp_engine/src/hstudio_func.cc
    ${HCCL_TOOLS_INSIGHT}/cpp_engine/insight_adapter/main.cc
)

add_executable(insight_adapter ${src_to_test_list})

# 需要针对protobuf自动编译生成的文件设置GENERATED属性，否则会编译报错
set_property(SOURCE ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto/analysis_result.pb.cc PROPERTY GENERATED 1)

#头文件搜索路径
target_include_directories(insight_adapter PRIVATE
    ${include_list}
)

target_compile_definitions(insight_adapter PRIVATE
    google=ascend_private
    -DOPEN_HCCL_TEST
)

target_compile_options(insight_adapter PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O3
    -pipe
)

#本地是否有共享库存在
if(EXISTS "${HCCL_ALG_ANALYZER_LIB_ROOT}/hccl_alg/libhccl_alg_analyzer.so" )
    find_library(LIBRARY_ANALYZER libhccl_alg_analyzer.so ${HCCL_ALG_ANALYZER_LIB_ROOT}/hccl_alg)
    target_link_libraries(insight_adapter PRIVATE
    ${LIBRARY_ANALYZER}
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:slog_headers>
    ascend_protobuf
    c_sec_static
    -lrt
    -ldl
)
else()
    target_link_libraries(insight_adapter PRIVATE
    hccl_alg_analyzer
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:slog_headers>
    ascend_protobuf
    c_sec_static
    -lrt
    -ldl
)
add_dependencies(insight_adapter hccl_alg_analyzer_lib)
endif()

add_custom_target(insight_adapter_lib
  DEPENDS insight_adapter
  COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
  COMMAND cp insight_adapter ${CMAKE_INSTALL_PREFIX}/hccl_lib
  COMMAND cd ${CMAKE_INSTALL_PREFIX}/hccl_lib && strip insight_adapter
)
