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
project(hccl)

if(BUILD_OPEN_PROJECT)
    message(STATUS "open TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
else()
    message(STATUS "close TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${TOP_DIR}/hccl/)
    set(INSTALL_AICPU_KERNEL_JSON_DIR ${INSTALL_LIBRARY_DIR})
endif()

set(INCLUDE_LIST
    ${OPS_BASE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hcomm_dlsym

    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/inc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/selector

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aiv
    # ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/hostdpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/dpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/interface_graph_mode

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template/host_nic

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template/host_nic

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/selector
)

if(BUILD_OPEN_PROJECT)
    list(APPEND INCLUDE_LIST
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl

        # hcomm头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm

        # runtime头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/
        ${ASCEND_CANN_PACKAGE_PATH}/include/base/
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/include/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/include/external/
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace/
    )
    # pkg_inc/hcomm/ccu 仅 9.0.0 CANN 存在，8.5.0 下直接不加入 include 路径
    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu)
        list(APPEND INCLUDE_LIST ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu)
    endif()
else()
    list(APPEND INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/external/acl
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/inc/aicpu/common
        ${TOP_DIR}/ace/csruntime/inc/tsch
        ${TOP_DIR}/asl/hss/inc
        ${TOP_DIR}/asl/hss/inc/pub
        ${TOP_DIR}/asl/hss/common/hash_table
        ${TOP_DIR}/metadef
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/dfx
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/base

        ${TOP_DIR}/open_source/json/include

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc/hcomm/ccu
        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
    )
endif()

add_definitions(-DORION_MODE)

if(PRODUCT_SIDE STREQUAL "host")
    include(hccl.cmake)
else()
    include(scatter_aicpu_kernel.cmake)
endif()

add_subdirectory(common)
add_subdirectory(ops)

# 定义AIV算子聚合目标
set(AIV_TARGETS
    hccl_aiv_all_gather_op_910_95
    hccl_aiv_all_reduce_op_910_95
    hccl_aiv_all_to_all_v_op_910_95
    hccl_aiv_all_to_all_op_910_95
    hccl_aiv_broadcast_op_910_95
    hccl_aiv_reduce_op_910_95
    hccl_aiv_reduce_scatter_op_910_95
    hccl_aiv_scatter_op_910_95
    hccl_aiv_send_op_910_95
    hccl_aiv_recv_op_910_95
    hccl_ar_superkernel_mesh_1d_oneshot_int8_v2
    hccl_ar_superkernel_mesh_1d_oneshot_int16_v2
    hccl_ar_superkernel_mesh_1d_oneshot_int32_v2
    hccl_ar_superkernel_mesh_1d_oneshot_float16_v2
    hccl_ar_superkernel_mesh_1d_oneshot_float32_v2
    hccl_ar_superkernel_mesh_1d_oneshot_bfloat16_v2
    hccl_ar_superkernel_mesh_1d_twoshot_int8_v2
    hccl_ar_superkernel_mesh_1d_twoshot_int16_v2
    hccl_ar_superkernel_mesh_1d_twoshot_int32_v2
    hccl_ar_superkernel_mesh_1d_twoshot_float16_v2
    hccl_ar_superkernel_mesh_1d_twoshot_float32_v2
    hccl_ar_superkernel_mesh_1d_twoshot_bfloat16_v2
    hccl_rs_superkernel_mesh_1d_int8_v2
    hccl_rs_superkernel_mesh_1d_int16_v2
    hccl_rs_superkernel_mesh_1d_int32_v2
    hccl_rs_superkernel_mesh_1d_float16_v2
    hccl_rs_superkernel_mesh_1d_float32_v2
    hccl_rs_superkernel_mesh_1d_bfloat16_v2
    hccl_ag_superkernel_mesh_1d_int8_v2
    hccl_ag_superkernel_mesh_1d_int16_v2
    hccl_ag_superkernel_mesh_1d_int32_v2
    hccl_ag_superkernel_mesh_1d_float16_v2
    hccl_ag_superkernel_mesh_1d_float32_v2
    hccl_ag_superkernel_mesh_1d_bfloat16_v2
    hccl_ag_superkernel_mesh_1d_uint8_v2
    hccl_ag_superkernel_mesh_1d_uint16_v2
    hccl_ag_superkernel_mesh_1d_uint32_v2
    hccl_ag_superkernel_mesh_1d_int64_v2
    hccl_ag_superkernel_mesh_1d_uint64_v2
    hccl_a2a_superkernel_mesh_1d_int8_v2
    hccl_a2a_superkernel_mesh_1d_int16_v2
    hccl_a2a_superkernel_mesh_1d_int32_v2
    hccl_a2a_superkernel_mesh_1d_float16_v2
    hccl_a2a_superkernel_mesh_1d_float32_v2
    hccl_a2a_superkernel_mesh_1d_bfloat16_v2
    hccl_a2a_superkernel_mesh_1d_uint8_v2
    hccl_a2a_superkernel_mesh_1d_uint16_v2
    hccl_a2a_superkernel_mesh_1d_uint32_v2
    hccl_a2a_superkernel_mesh_1d_int64_v2
    hccl_a2a_superkernel_mesh_1d_uint64_v2
)

add_custom_target(aiv_all_targets)
foreach(target ${AIV_TARGETS})
    if(TARGET ${target})
        add_dependencies(aiv_all_targets ${target})
    endif()
endforeach()