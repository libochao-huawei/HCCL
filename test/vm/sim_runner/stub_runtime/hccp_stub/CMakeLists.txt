target_include_directories(stub_runtime PRIVATE
    ${TOP_DIR}/drivers/network/inc/network/
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub
    ${TOP_DIR}/ace/comop/hccl/orion/common/external_system/ccu/ccu_resource
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component/ccu_channel
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_v1
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/ccu/ccu_device/ccu_component/ccu_pfe
    ${TOP_DIR}/asl/devtools/hccl_tools/hccl_sim_runner/stub_runtime/rts_stub/ccu_executor
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
    ${TOP_DIR}/ace/comop/hccl/orion/framework/topo
    ${TOP_DIR}/ace/comop/hccl/whole/hccl_next/framework/topo/new_topo_builder/common
)

target_sources(stub_runtime PRIVATE
    hccp_stub.cc
    fake_socket.cc
)