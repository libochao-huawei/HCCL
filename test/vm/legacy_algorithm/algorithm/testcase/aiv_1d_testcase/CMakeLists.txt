# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_executable(orion_alg_aiv_1d_testcase ${test_case_list})
 
#头文件搜索路径
target_include_directories(orion_alg_aiv_1d_testcase PRIVATE
    ${include_list}
)
 
target_compile_definitions(orion_alg_aiv_1d_testcase PRIVATE
    -DOPEN_HCCL_TEST
    _GLIBCXX_USE_CXX11_ABI=0
    -DCCL_KERNEL_AICPU
    -DHCCL_ALG_ANALYZER_DAVID
)
 
target_compile_options(orion_alg_aiv_1d_testcase PRIVATE
    -U_FORTIFY_SOURCE
    -Os
    -O0 -g
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)

add_dependencies(orion_alg_aiv_1d_testcase
    hccl_alg_adaption_v2
)
 
target_link_libraries(orion_alg_aiv_1d_testcase PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    hccl_alg_analyzer_legacy
    hccl_orion_alg_test
    hccl_alg_adaption_v2
    $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
    -pthread
    protobuf
    -lrt
    -ldl
)

target_sources(orion_alg_aiv_1d_testcase
    PUBLIC
    main.cc
    testcase_utils.cc
    testcase_all_reduce_aiv_1d_twoshot.cc
    testcase_all_reduce_aiv_1d_oneshot.cc
    testcase_all_gather_aiv_1d.cc
    testcase_scatter_aiv_1d.cc
    testcase_broadcast_aiv_1d.cc
    testcase_all_to_all_aiv_1d.cc
    testcase_all_to_all_v_aiv_1d.cc
    testcase_reduce_aiv_1d.cc
    testcase_reduce_scatter_aiv_1d.cc
    ${UTILS_DIR}/common/virtual_topo_stub/virtual_topo_stub.cc
)