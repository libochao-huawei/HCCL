# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT ENABLE_EXPERIMENTAL)
    return()
endif()

add_subdirectory(op_common)
add_subdirectory(reduce_scatter)

set(EXPERIMENTAL_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/op_common/
    ${CMAKE_CURRENT_SOURCE_DIR}/op_common/template
    ${CMAKE_CURRENT_SOURCE_DIR}/op_common/topo
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/birs/
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/birs/template/
)

if(TARGET scatter_aicpu_kernel)
    target_include_directories(scatter_aicpu_kernel PRIVATE ${EXPERIMENTAL_INCLUDE_LIST})
    target_compile_definitions(scatter_aicpu_kernel PRIVATE ENABLE_EXPERIMENTAL)
    target_sources(scatter_aicpu_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_common/template/alg_template_base_experimental.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op_common/topo/topo_experimental.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/birs/reduce_scatter_birs_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/birs/reduce_scatter_executor_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/birs/template/reduce_scatter_birs.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter/birs/template/reduce_scatter_birs_inter.cc
    )
endif()
target_include_directories(hccl PRIVATE ${EXPERIMENTAL_INCLUDE_LIST})
target_compile_definitions(hccl PRIVATE ENABLE_EXPERIMENTAL)