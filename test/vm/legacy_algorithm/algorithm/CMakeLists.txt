# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 14)

set(HCCL_ALG_ANALYZER_DAVID 1 CACHE STRING "Define HCCL_ALG_ANALYZER_DAVID as a fixed value")

project(hccl_checker_ops_stest)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(TESTCASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/testcase)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(HCOMM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(THIRD_PARTH_PATH ${HCOMM_ROOT_PATH}/output/third_party)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}/aarch64-linux")
endif()
message(STATUS "Final Ascend Path: ${ASCEND_HOME_PATH}")

set(HCOMM_CODE_COMMON    ${HCOMM_ROOT_PATH}/src/common)
set(HCOMM_CODE_PLATFORM  ${HCOMM_ROOT_PATH}/src/platform)
set(HCOMM_CODE_ALGORITHM ${HCOMM_ROOT_PATH}/src/algorithm)
set(HCOMM_CODE_FRAMEWORK ${HCOMM_ROOT_PATH}/src/framework)

set(HCOMM_CODE_LEGACY      ${HCOMM_ROOT_PATH}/src/legacy)
set(HCOMM_LEGACY_FRAMEWORK ${HCOMM_CODE_LEGACY}/framework)
set(HCOMM_LEGACY_PLATFORM  ${HCOMM_CODE_LEGACY}/unified_platform)
set(HCOMM_LEGACY_SERVICE   ${HCOMM_CODE_LEGACY}/service/collective)

set(THIRD_PARTY_PATH   ${CANN_3RD_LIB_PATH})
set(PROTOC_PROGRAM     ${THIRD_PARTY_PATH}/protobuf/bin/protoc)

# 下载并编译gtest
set(CANN_3RD_LIB_PATH ${HCOMM_ROOT_PATH}/third_party)
include(${HCOMM_ROOT_PATH}/cmake/third_party/gtest.cmake)

# Include List
set(ASCEND_CANN_INCLUDE_LIST
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/asc/impl/basic_api
)

set(legacy_framework_include_list
    ${HCOMM_LEGACY_FRAMEWORK}
    ${HCOMM_LEGACY_FRAMEWORK}/entrance
    ${HCOMM_LEGACY_FRAMEWORK}/env_config
    ${HCOMM_LEGACY_FRAMEWORK}/topo
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/phy_topo
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/rank_graph
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/common
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/rank_table_info
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/rank_graph_builder
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/topo_info
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/phy_topo_builder
    ${HCOMM_LEGACY_FRAMEWORK}/topo/new_topo_builder/phy_topo_builder

    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/transport/aicpu
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/buffer/aicpu
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/buffer
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/socket
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/connection
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/connection/ub_ci_updater
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/notify
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/stream
    ${HCOMM_LEGACY_FRAMEWORK}/resource_manager/transport

    ${HCOMM_LEGACY_FRAMEWORK}/misc/mask_event
    ${HCOMM_LEGACY_FRAMEWORK}/misc/json_parser
    ${HCOMM_LEGACY_FRAMEWORK}/misc/op_info

    ${HCOMM_LEGACY_FRAMEWORK}/fault_recovery

    ${HCOMM_LEGACY_FRAMEWORK}/communicator
    ${HCOMM_LEGACY_FRAMEWORK}/communicator/aicpu
    ${HCOMM_LEGACY_FRAMEWORK}/dfx/common
    ${HCOMM_LEGACY_FRAMEWORK}/dfx/task_exception
    ${HCOMM_LEGACY_FRAMEWORK}/ccu
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_ins
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_manager
    ${HCOMM_LEGACY_FRAMEWORK}/aiv/aiv_ins
    ${HCOMM_LEGACY_FRAMEWORK}/service
    ${HCOMM_LEGACY_FRAMEWORK}/dfx/profiling
    ${HCOMM_LEGACY_FRAMEWORK}/service/one_sided_service
    ${HCOMM_LEGACY_FRAMEWORK}/device_mode
    ${HCOMM_LEGACY_FRAMEWORK}/ccu/ccu_mc2
)

set(legacy_platform_include_list
    # 平台层公共头文件路径
    ${HCOMM_LEGACY_PLATFORM}/pub_inc
    ${HCOMM_LEGACY_PLATFORM}/pub_inc/ccu
    ${HCOMM_LEGACY_PLATFORM}/pub_inc/resource
    ${HCOMM_LEGACY_PLATFORM}/pub_inc/resource/net_dev
    ${HCOMM_LEGACY_PLATFORM}/pub_inc/resource/transport

    # 平台层CCU实现头文件路径
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_context
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_microcode
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/context
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/interface
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/arithmetic
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/common
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/control
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/data
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/loop
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/sync
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_representation/reps/translator
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_device
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_device/ccu_component
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_device/ccu_component/ccu_channel
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_device/ccu_component/ccu_channel/ccu_channel_v1
    ${HCOMM_LEGACY_PLATFORM}/ccu/ccu_device/ccu_component/ccu_pfe
    ${HCOMM_LEGACY_PLATFORM}/ccu/dfx
    ${HCOMM_LEGACY_PLATFORM}/aiv
    ${HCOMM_LEGACY_PLATFORM}/hccp/inc/network

    ${HCOMM_LEGACY_PLATFORM}
    ${HCOMM_LEGACY_PLATFORM}/common
    ${HCOMM_LEGACY_PLATFORM}/resource
    ${HCOMM_LEGACY_PLATFORM}/resource/buffer
    ${HCOMM_LEGACY_PLATFORM}/resource/task
    ${HCOMM_LEGACY_PLATFORM}/resource/stream
    ${HCOMM_LEGACY_PLATFORM}/resource/buffer/aicpu
    ${HCOMM_LEGACY_PLATFORM}/resource/transport/aicpu
    ${HCOMM_LEGACY_PLATFORM}/resource/stream/aicpu
    ${HCOMM_LEGACY_PLATFORM}/resource/notify/aicpu
    ${HCOMM_LEGACY_PLATFORM}/resource/connection
    ${HCOMM_LEGACY_PLATFORM}/resource/connection/aicpu
    ${HCOMM_LEGACY_PLATFORM}/resource/socket
    ${HCOMM_LEGACY_PLATFORM}/resource/notify
    ${HCOMM_LEGACY_PLATFORM}/resource/connection/ub_ci_updater
    ${HCOMM_LEGACY_PLATFORM}/resource/transport
    ${HCOMM_LEGACY_PLATFORM}/external_system
)

set(legacy_service_include_list
    ${HCOMM_LEGACY_SERVICE}
    ${HCOMM_LEGACY_SERVICE}/primitive
    ${HCOMM_LEGACY_SERVICE}/alg/interface
    ${HCOMM_LEGACY_SERVICE}/alg/interface/host
    ${HCOMM_LEGACY_SERVICE}/instruction
    ${HCOMM_LEGACY_SERVICE}/alg/selector
    ${HCOMM_LEGACY_SERVICE}/alg/alg_registry
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_executor
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_executor/ins_alg_executor
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_executor/prim_alg_executor
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/ins_alg_template
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/prim_alg_template
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/ccu_alg_template
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_template/aiv_alg_template
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_topo_match
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/utils/common
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_ccu_context
    ${HCOMM_LEGACY_SERVICE}/alg/coll_alg_factory/alg_aiv_instruction
)

set(legacy_common_include_list
    ${HCOMM_CODE_LEGACY}/common
    ${HCOMM_CODE_LEGACY}/common/types
    ${HCOMM_CODE_LEGACY}/common/utils
    ${HCOMM_CODE_LEGACY}/common/exception
    ${HCOMM_CODE_LEGACY}/include
)

set(checker_include_list
    ${UTILS_DIR}/adapter_v2
    ${UTILS_DIR}/adapter_v2/stub_ccu
    ${UTILS_DIR}/adapter_v2/adapter_rts
    ${UTILS_DIR}/adapter_v2/transport
    ${UTILS_DIR}/adapter_v2/communicator
    ${UTILS_DIR}/adapter_v2/common

    ${UTILS_DIR}/inc
    ${UTILS_DIR}/checker
    ${UTILS_DIR}/checker/rank_info_recorder
    ${UTILS_DIR}/checker/mem_layout
    ${UTILS_DIR}/checker/task_collector
    ${UTILS_DIR}/checker/link_type_recorder
    ${UTILS_DIR}/checker/mem_conflict_check
    ${UTILS_DIR}/common
    ${UTILS_DIR}/common/acl_stub
    ${UTILS_DIR}/common/virtual_topo_stub
)

set(HCOMM_PLATFORM_INCLUDE_LIST
    ${HCOMM_CODE_PLATFORM}/hccp/inc/network
    ${HCOMM_CODE_PLATFORM}/hccp/external_depends/inc/log
)

add_subdirectory(testcase)
add_subdirectory(utils)