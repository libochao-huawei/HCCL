add_subdirectory(fakeso)

add_library(stub_runtime_aicpu SHARED "")
target_compile_options(stub_runtime_aicpu PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
    -D_GLIBCXX_USE_CXX11_ABI=0
    -DLOG_CPP
    -DDEVICE_STUB
)

set(HCCL_TOOL_STUB ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime)
FILE(GLOB_RECURSE STUB_FILES ${HCCL_TOOL_STUB}/*.cc)
target_sources(stub_runtime_aicpu PRIVATE
    ${STUB_FILES}
)

target_include_directories(stub_runtime_aicpu PRIVATE
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/driver
    ${TOP_DIR}/inc/aicpu/
    ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
    ${TOP_DIR}/abl/slog/inc
    ${TOP_DIR}/abl/mmpa/inc/mmpa
    ${TOP_DIR}/abl/mmpa/inc/mmpa/stub_inc
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/abl/slog/inc/toolchain
    ${TOP_DIR}/llt/air/vcast/depends
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/inc/hccl
    ${TOP_DIR}/open_source/json/include
    ${TOP_DIR}/metadef
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/pkg_inc
    ${TOP_DIR}/metadef/inc/register
    ${TOP_DIR}/metadef/inc/common/util
    ${TOP_DIR}/metadef/inc/common/util/error_manager
    ${TOP_DIR}/metadef/inc/common/util/trace_manager
    ${TOP_DIR}/metadef/inc/common/ge_common
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/inc/external/register
    ${TOP_DIR}/asl/hss/inc
    ${TOP_DIR}/asl/hss/inc/pub
    ${TOP_DIR}/asl/hss/common/hash_table
    ${TOP_DIR}/asl/ops/cann/ops/common/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/inc
    ${TOP_DIR}/ace/comop/inc/hccl
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/algorithm/pub_inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/transport
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/inc/adapter
    ${TOP_DIR}/drivers/network/inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/notify
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/common
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/task
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc/inner
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc/new
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/impl/resource_manager
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/base/alg_template
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/algorithm/impl/legacy
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/base/communicator/legacy
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/base/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/impl/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/base/communicator
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/impl/task
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/impl
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/transport/heterog
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/transport/host
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/common/debug/profiling/inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/common/debug/config
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/cluster_maintenance/recovery/operator_retry/
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/common/stream/
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
    ${TOP_DIR}/ace/comop/hccl/orion/framework/topo
    ${TOP_DIR}/ace/comop/hccl/whole/hccl_next/framework/topo/new_topo_builder/common

    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
    ${TOP_DIR}/ace/comop/hccl/orion/framework/topo
    ${TOP_DIR}/ace/comop/hccl/whole/hccl_next/framework/topo/new_topo_builder/common

    ${TOP_DIR}/ace/comop/hccl/orion/service/collective
    ${TOP_DIR}/ace/comop/hccl/orion/common/types
    ${TOP_DIR}/ace/comop/hccl/orion/common/utils
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/stream/aicpu
    ${TOP_DIR}/ace/comop/hccl/whole/orion/framework/communicator
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device
    ${TOP_DIR}/ace/comop/hccl/orion/common
    ${TOP_DIR}/ace/comop/hccl/orion/common/exception
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/external_system
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer/aicpu
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/pub_inc
    ${TOP_DIR}/inc/aicpu/common
    ${TOP_DIR}/ace/csruntime/inc/tsch

    ${TOP_DIR}/abl/qos
    ${TOP_DIR}/abl/libc_sec/include

    ${TOP_DIR}/drivers/network/inc/network/
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub
    ${TOP_DIR}/ace/comop/hccl/orion/common/external_system/ccu/ccu_resource
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component/ccu_channel
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_v1
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component/ccu_pfe
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub/ccu_executor

    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/virt_runtime_fwk
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/com_stub
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/ai_cpu_kernel
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device
    ${TOP_DIR}/ace/comop/hccl/orion/common
    ${TOP_DIR}/ace/comop/hccl/orion/common/exception
    ${TOP_DIR}/ace/comop/hccl/orion/common/external_system
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_microcode
    ${TOP_DIR}/ace/comop/hccl/orion/common/external_system/ccu/ccu_resource
    ${TOP_DIR}/ace/comop/hccl/orion/common/external_system/ccu
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/pub_inc/ccu
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_alg_analyzer/inc
    ${TOP_DIR}/ace/comop/hccl/orion/common/external_system/ccu/ccu_instruction
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub/ccu_executor
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/device
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc/aicpu
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/device/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/common/src/config
    ${TOP_DIR}/ace/npuruntime/acl/inc/external
    ${TOP_DIR}/ace/npuruntime/inc/runtime
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/common/debug/profiling/inc/aicpu
    ${TOP_DIR}/asc/asc-devkit
    ${TOP_DIR}/asc/asc-devkit/impl
    ${TOP_DIR}/ace/npuruntime/runtime/platform/inc
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/connection/aicpu  # udma_data_struct.h
)

add_custom_command(TARGET stub_runtime_aicpu POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libstub_runtime_aicpu.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)