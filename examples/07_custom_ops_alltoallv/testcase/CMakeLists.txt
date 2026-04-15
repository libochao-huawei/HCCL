# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(hccl_custom_alltoallv_testcase LANGUAGES CXX)

add_executable(hccl_custom_alltoallv_test
    main.cc
)

target_include_directories(hccl_custom_alltoallv_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CUSTOM_OPS_OPP_INC_PATH}
)

target_link_libraries(hccl_custom_alltoallv_test PRIVATE
    Ascend::acl
    Ascend::hccl
    hccl_custom_alltoallv
)

set_target_properties(hccl_custom_alltoallv_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(TARGETS hccl_custom_alltoallv_test DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)