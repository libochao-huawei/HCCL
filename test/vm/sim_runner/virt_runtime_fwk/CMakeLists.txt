set(HCCL_LIB_DIR ${TOP_DIR}/output/ascend/host/ubuntu20.04/x86_64/lib)

add_library(virt_runtime_fwk SHARED "")

add_subdirectory(sim_params)
add_subdirectory(ranktable)

target_sources(virt_runtime_fwk PRIVATE
    hccl_sim_situation.cc
    hccl_sim_interface.cc
    hccl_sim_op_flow.cc
    hccl_sim_prepare_verify.cc
)

target_include_directories(virt_runtime_fwk PUBLIC
    ./sim_params
    ./ranktable
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/com_stub
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/hccp_stub
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/virt_runtime_fwk

    # 依赖业务的头文件目录
    ${TOP_DIR}/inc
    ${TOP_DIR}/inc/aicpu
    ${TOP_DIR}/inc/driver
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ace/comop/inc/hccl
    ${TOP_DIR}/ace/comop/hccl/open_source/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/inc/hccl
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/pkg_inc
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/inc/common/util/error_manager
    ${TOP_DIR}/drivers/network/inc
    ${TOP_DIR}/drivers/network/inc/network
    ${TOP_DIR}/abl/slog/inc
    ${TOP_DIR}/abl/slog/inc/toolchain
    ${TOP_DIR}/abl/mmpa/inc/mmpa
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/open_source/json/include
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/ace/npuruntime/acl/inc/external
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/algorithm/pub_inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc/inner
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc/new
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/task
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/task/rtsq_interact
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/inc
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/common
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/transport
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/transport/heterog
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/inc/adapter
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/platform/resource/notify
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/ai_cpu_kernel
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/inc/host
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/impl/resource_manager
    ${TOP_DIR}/ace/comop/hccl/orion/common/utils
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/base/alg_template
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/algorithm/base/communicator

    # ${TOP_DIR}/ace/comop/hccl/orion/framework/entrance/op_base
    # ${TOP_DIR}/ace/comop/hccl/orion/framework/entrance/hcom_comm

    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
    ${TOP_DIR}/ace/comop/hccl/orion/framework/topo
    ${TOP_DIR}/ace/comop/hccl/whole/hccl_next/framework/topo/new_topo_builder/common

    ${TOP_DIR}/ace/npuruntime/inc/runtime

    #aicpu特有
    ${TOP_DIR}/ace/comop/hccl/orion/service/collective
    ${TOP_DIR}/ace/comop/hccl/orion/common/types
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer

    ${TOP_DIR}/ace/comop/hccl/whole/orion/framework/communicator
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/pub_inc
    ${TOP_DIR}/inc/aicpu/common
    ${TOP_DIR}/ace/csruntime/inc/tsch
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/device/common
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/device
    ${TOP_DIR}/ace/comop/hccl/whole/hccl/pub_inc/aicpu
    ${TOP_DIR}/ace/comop/hccl/open_source/src/domain/collective_communication/framework/device/inc

    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/external_system
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer/aicpu
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_microcode
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub/ccu_executor
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/pub_inc/ccu

    # 引用二进制需要
    ${TOP_DIR}/ace/comop/hccl/open_source/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/inc/hccl
    ${TOP_DIR}/ace/comop/hccl/orion/common/utils
    ${TOP_DIR}/open_source/json/include
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ace/comop/inc/hccl
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/inc/external

    # 引用phy_topo.h
    ${TOP_DIR}/ace/comop/hccl/orion/framework/topo/new_topo_builder/phy_topo
    ${TOP_DIR}/ace/comop/hccl/orion/framework/topo/new_topo_builder/common
    ${TOP_DIR}/ace/comop/hccl/orion/common
    ${TOP_DIR}/ace/comop/hccl/orion/common/exception
    # 引用env_config.h
    ${TOP_DIR}/ace/comop/hccl/orion/framework/env_config
    ${TOP_DIR}/ace/comop/hccl/orion/service/collective/primitive
    # aicpu_hccl_def.h->hccl/hccl_msg.h
    ${TOP_DIR}/asc/asc-devkit
    ${TOP_DIR}/asc/asc-devkit/impl
    # aicpu_hccl_def.h->dfx_extend_info.h
)

target_link_libraries(virt_runtime_fwk PUBLIC
    c_sec
    mmpa
    runtime
    qos_manager
    exe_graph
    lowering
    error_manager
    tsdclient
    ra
    ascendalog
    ascendcl
    stub_runtime
    ${HCCL_LIB_DIR}/libhccl_v2_alg_frame.so
    ${HCCL_LIB_DIR}/libhccl_v2.so
    ${HCCL_LIB_DIR}/libhccl_tbe_task.so
    ${HCCL_LIB_DIR}/libhccl_plf.so
    ${HCCL_LIB_DIR}/libhccl_alg.so
    ${HCCL_LIB_DIR}/libhccl_v2_native_alg_repo.so
    ${HCCL_LIB_DIR}/libhccl.so
)

target_compile_options(virt_runtime_fwk PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
    -D_GLIBCXX_USE_CXX11_ABI=0
    -DLOG_CPP
)

add_custom_command(TARGET virt_runtime_fwk POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libvirt_runtime_fwk.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)
