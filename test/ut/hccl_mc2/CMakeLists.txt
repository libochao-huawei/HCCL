# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    hccl_mc2_test.cc
)

add_executable(hccl_mc2_test ${src_list})

if (BUILD_OPEN_PROJECT)
    target_include_directories(hccl_mc2_test PUBLIC
        # 第三方头文件
        ${CANN_3RD_LIB_PATH}/gtest/include/

        # HCCL仓头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/ops/op_common/inc

        # Ascend安装包头文件
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_HOME_PATH}/hcomm/include/hccl
        ${ASCEND_HOME_PATH}/include/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hccl
        ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu
        # 本仓代码头文件
        ${HCCL_DEV_BASE}/src/common
        ${HCCL_DEV_BASE}/src/ops
        ${HCCL_DEV_BASE}/src/ops/op_common
        ${HCCL_DEV_BASE}/src/ops/op_common/inc
        ${HCCL_DEV_BASE}/src/common/hcomm_dlsym
        ${HCCL_DEV_BASE}/include

        ${UTILS_DIR}/sim_world
        ${UTILS_DIR}/common/utils
        ${UTILS_DIR}/common/exception
        ${UTILS_DIR}/hccl_verifier
        ${UTILS_DIR}/hccl_verifier/semantics_check
        ${UTILS_DIR}/hccl_verifier/task_graph_generator
        ${UTILS_DIR}/hccl_proxy
        ${UTILS_DIR}/hccl_proxy/communicator
        ${UTILS_DIR}/hccl_proxy/topo_model
    )

    target_link_libraries(hccl_mc2_test PRIVATE
        gtest
        pthread
        hccl-model-proxy
        hccl-verifier
        # 链接hccl库
        ${ASCEND_HOME_PATH}/lib64/libhccl.so
        -ldl
    )

    target_compile_options(hccl_mc2_test PRIVATE
        -std=c++14
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
    )
endif()

run_llt_test(
    TARGET  hccl_mc2_test
    TASK_NUM 1
)