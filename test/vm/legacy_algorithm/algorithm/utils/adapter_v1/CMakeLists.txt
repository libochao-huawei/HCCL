set(HCCL_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/whole)
set(HCCL_OPEN_CODE_ROOT ${TOP_DIR}/ace/comop/hccl/open_source)
set(HCCL_ALG_ROOT ${TOP_DIR}/asl/devtools/hccl_tools/hccl_alg_analyzer)
set(HCCL_OPEN_CODE_ALGORITHM ${HCCL_OPEN_CODE_ROOT}/src/domain/collective_communication/algorithm)
set(HCCL_OPEN_CODE_FRAMEWORK ${HCCL_OPEN_CODE_ROOT}/src/domain/collective_communication/framework)
set(HCCL_OPEN_CODE_COMMON ${HCCL_OPEN_CODE_ROOT}/src/domain/collective_communication/common)
set(HCCL_ANALYZER_INC_DIR ${TOP_DIR}/asl/devtools/hccl_tools/hccl_alg_analyzer/pub_inc)

set(include_list
    # 测试框架的代码
    ${HCCL_ALG_ROOT}/inc
    ${HCCL_ALG_ROOT}/checker
    ${HCCL_ALG_ROOT}/checker/common/types
    ${HCCL_ALG_ROOT}/checker/rank_info_recorder
    ${HCCL_ALG_ROOT}/checker/mem_layout
    ${HCCL_ALG_ROOT}/checker/link_type_recorder
    ${HCCL_ALG_ROOT}/checker/task_collector
    ${HCCL_ALG_ROOT}/checker/semantics_check
    ${HCCL_ALG_ROOT}/checker/singletask_check
    ${HCCL_ALG_ROOT}/checker/ui/data_dumper
    ${HCCL_ALG_ROOT}/checker/param_trans

    ${HCCL_ALG_ROOT}/adapter_v1
    ${HCCL_ALG_ROOT}/adapter_v1/dispatcher
    ${HCCL_ALG_ROOT}/adapter_v1/notify
    ${HCCL_ALG_ROOT}/adapter_v1/runtime
    ${HCCL_ALG_ROOT}/adapter_v1/transport
    ${HCCL_ALG_ROOT}/adapter_v1/stream
    ${HCCL_ALG_ROOT}/adapter_v1/aiv

    ${HCCL_ANALYZER_INC_DIR}/adapter_v1/aiv

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
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/ace/npuruntime/acl/inc/external
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/abl/atrace/inc/utrace
    ${TOP_DIR}/abl/mmpa/inc/mmpa
    ${TOP_DIR}/drivers/network/inc
    ${TOP_DIR}/abl/mmpa/inc/mmpa

    # 添加hccl目录
    ${HCCL_CODE_ROOT}/hccl/algorithm/pub_inc
    ${HCCL_CODE_ROOT}/hccl/pub_inc
    ${HCCL_CODE_ROOT}/hccl/pub_inc/inner
    ${HCCL_CODE_ROOT}/hccl/pub_inc/new
    
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_all_gather
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_all_reduce
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_alltoallv
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_alltoall
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_reduce_scatter
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_reduce
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_scatter
    ${HCCL_OPEN_CODE_ALGORITHM}/base/alg_template/temp_broadcast

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
    ${HCCL_OPEN_CODE_COMMON}/launch_device

    # platform目录下的头文件
    ${HCCL_CODE_ROOT}/hccl/platform/inc
    ${HCCL_CODE_ROOT}/hccl/platform/inc/adapter
    ${HCCL_CODE_ROOT}/hccl/platform/inc/adapter/host
    ${HCCL_CODE_ROOT}/hccl/platform/common

    # framework目录下的头文件
    ${HCCL_OPEN_CODE_FRAMEWORK}/inc
    ${HCCL_OPEN_CODE_FRAMEWORK}/inc/host
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/task
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/config
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo
    ${HCCL_OPEN_CODE_FRAMEWORK}/communicator/impl

    # libc_sec目录下的头文件
    ${TOP_DIR}/abl/slog/inc/toolchain

    # json 目录的头文件
    ${TOP_DIR}/open_source/json/include

    ${CMAKE_BINARY_DIR}/proto/alg_analyzer_proto/proto

    ${TOP_DIR}/ace/npuruntime/inc/runtime
)

set(adaption_to_test_list
    # 依赖平台层的代码（这边最好不依赖平台层）
    ${HCCL_ALG_ROOT}/adapter_v1/alg_adapt_v1_interface.cc
    ${HCCL_ALG_ROOT}/adapter_v1/device_info_recorder.cc
    ${HCCL_ALG_ROOT}/adapter_v1/orchestrate.cc
    ${HCCL_ALG_ROOT}/adapter_v1/transformer.cc
    ${HCCL_ALG_ROOT}/adapter_v1/utils_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/communicator_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/log_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/slog_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/network_manager_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/error_manager_stub.cpp
    ${HCCL_ALG_ROOT}/adapter_v1/hccl_aiv_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/sal_stub.cc

    # #aiv业务代码
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_base_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_copy_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_memory_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_sync_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/sync_interface.cc

    ${HCCL_CODE_ROOT}/hccl/platform/resource/mem/mem_host.cc
    ${HCCL_CODE_ROOT}/hccl/platform/resource/mem/mem_device.cc
    ${HCCL_CODE_ROOT}/hccl/platform/common/workflow.cc
    ${HCCL_CODE_ROOT}/hccl/platform/common/device_capacity.cc

    ${HCCL_CODE_ROOT}/hccl/platform/common/externalinput.cc
    ${HCCL_CODE_ROOT}/hccl/platform/common/dlhal_function.cc
    ${HCCL_CODE_ROOT}/hccl/platform/common/hccl_dl.cc
    ${HCCL_CODE_ROOT}/hccl/platform/common/hccl_ip_address.cc
    ${HCCL_CODE_ROOT}/hccl/platform/debug/traceinfo/hccl_trace_info.cc

    # 依赖框架层的代码
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/config.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/config/env_config.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/config/env_config_host.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo/topoinfo_parse.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableParser.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableConcise.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableHeterog.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo/topoinfo_ranktableStandard.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/topo/topoinfo_roletableParser.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/common/src/json_utils.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/communicator/impl/hccl_communicator_attrs.cc
    ${HCCL_OPEN_CODE_FRAMEWORK}/communicator/impl/hccl_communicator_attrs_host.cc

    # 依赖开源仓的代码
    ${HCCL_OPEN_CODE_ALGORITHM}/impl/resource_manager/ccl_buffer_manager.cc
    ${HCCL_OPEN_CODE_ALGORITHM}/impl/resource_manager/stream_active_manager.cc
    ${HCCL_OPEN_CODE_COMMON}/debug/profiling/profiler_base.cc
    ${HCCL_OPEN_CODE_COMMON}/debug/config/config_log.cc
    ${HCCL_OPEN_CODE_COMMON}/launch_device/launch_device.cc
)

#aiv相关用c++17
set_source_files_properties(
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_base_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_copy_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_memory_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/aiv_sync_stub.cc
    ${HCCL_ALG_ROOT}/adapter_v1/aiv/sync_interface.cc
    PROPERTIES
    COMPILE_OPTIONS "-std=c++17"
)

add_library(hccl_alg_adaption SHARED ${adaption_to_test_list})

add_subdirectory(dispatcher)
add_subdirectory(transport)
add_subdirectory(notify)
add_subdirectory(runtime)
add_subdirectory(stream)

#头文件搜索路径
target_include_directories(hccl_alg_adaption PRIVATE
    ${include_list}
)

target_compile_definitions(hccl_alg_adaption PRIVATE
    CHECKER_V1
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(hccl_alg_adaption PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O2
    -pipe
)

if(HOST_HCCL_LLT STREQUAL "1")
    target_link_libraries(hccl_alg_adaption PRIVATE
        hccl_alg_test
        $<BUILD_INTERFACE:intf_llt_pub>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:atrace_headers>
        c_sec_static
        -lrt
        -ldl
    )
else()
    target_link_libraries(hccl_alg_adaption PRIVATE
        hccl_alg_test
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:atrace_headers>
        c_sec_static
        -lrt
        -ldl
        ofed_headers
    )
endif()

add_custom_command(TARGET hccl_alg_adaption POST_BUILD
    COMMAND ${CMAKE_STRIP} $<TARGET_FILE:hccl_alg_adaption>
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libhccl_alg_adaption.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

install(TARGETS hccl_alg_adaption
    LIBRARY DESTINATION lib OPTIONAL
    RUNTIME DESTINATION bin OPTIONAL
)
