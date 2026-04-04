# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(TARGET_NAME hccl_mc2_test)

set(UT_HCCL_MC2_SRC_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_mc2_test.cc
)

set(UT_HCCL_MC2_INCLUDE_LIST
    ${CANN_3RD_LIB_PATH}/gtest/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/topo
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/executor/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/template/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hcomm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hcomm/pkg_inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hcomm/include/hccl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hcomm/pkg_inc/hccl
)

add_executable(${TARGET_NAME} ${UT_HCCL_MC2_SRC_LIST})

target_include_directories(${TARGET_NAME} PRIVATE ${UT_HCCL_MC2_INCLUDE_LIST})

target_compile_options(${TARGET_NAME} PRIVATE 
    -U_FORTIFY_SOURCE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_compile_options(${TARGET_NAME} PRIVATE 
    -U_FORTIFY_SOURCE
    -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_link_libraries(${TARGET_NAME} PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -lpthread
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(
    TARGET  ${TARGET_NAME}
    TASK_NUM 1
)
