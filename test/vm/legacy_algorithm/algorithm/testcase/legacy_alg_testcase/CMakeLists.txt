# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_executable(legacy_alg_testcase ${test_case_list})

#头文件搜索路径
target_include_directories(legacy_alg_testcase PRIVATE
    ${include_list}
)

target_compile_definitions(legacy_alg_testcase PRIVATE
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
    -DCCL_KERNEL_AICPU
    -DHCCL_ALG_ANALYZER_DAVID
)

target_compile_options(legacy_alg_testcase PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O0 -g
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

add_dependencies(legacy_alg_testcase
    hccl_alg_adaption_v2
)

target_link_libraries(legacy_alg_testcase PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    hccl_alg_analyzer_legacy
    hccl_orion_alg_test
    hccl_alg_adaption_v2
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    ${THIRD_PARTY_PATH}/protobuf/lib64/libprotobuf.so
    -lrt
    -ldl
)

target_sources(legacy_alg_testcase
        PUBLIC
        main.cc
        testcase_all_reduce_aicpu_mesh1d_nhr.cc
        testcase_all_reduce_aicpu_mesh2d_nhr.cc
        testcase_reduce_scatter_aicpu_mesh2d_nhr.cc
        testcase_reduce_scatter_aicpu_mesh1d_nhr.cc
        testcase_all_reduce_aicpu_1d.cc
        testcase_utils.cc
        testcase_all_gather.cc
        testcase_all_gather_2d.cc
        testcase_all_reduce.cc
        testcase_alltoall.cc
        testcase_alltoallv.cc
        testcase_alltoallvc.cc
        testcase_reduce_scatter.cc
        testcase_scatter.cc
        testcase_reduce.cc
        testcase_reduce_aicpu_mesh1d_nhr.cc
        testcase_broadcast.cc
        testcase_all_reduce_aicpu_1d_mesh_chunk.cc
        testcase_all_gather_aicpu_mesh2d_nhr.cc
        ${UTILS_DIR}/common/virtual_topo_stub/virtual_topo_stub.cc
)