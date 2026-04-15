# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(hccl_custom_alltoallv_host LANGUAGES CXX)

add_library(hccl_custom_alltoallv SHARED
    alltoallv.cc
)

target_include_directories(hccl_custom_alltoallv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CUSTOM_OPS_OPP_INC_PATH}
)

target_link_libraries(hccl_custom_alltoallv PRIVATE
    Ascend::acl
    Ascend::hccl
)

set_target_properties(hccl_custom_alltoallv PROPERTIES
    OUTPUT_NAME "hccl_custom_alltoallv"
    PREFIX ""
    SUFFIX ".so"
)

install(TARGETS hccl_custom_alltoallv DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH})