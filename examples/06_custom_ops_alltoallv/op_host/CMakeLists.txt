# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN AS IS BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

add_library(hccl_custom_alltoallv SHARED)
set_target_properties(hccl_custom_alltoallv PROPERTIES LINKER_LANGUAGE CXX)

target_sources(hccl_custom_alltoallv PRIVATE
    all_to_all_v.cc
    launch_kernel.cc
)

target_include_directories(hccl_custom_alltoallv PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)

target_compile_options(hccl_custom_alltoallv PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O2
    -std=c++17
    -fstack-protector-all
)
target_compile_definitions(hccl_custom_alltoallv PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)
target_link_options(hccl_custom_alltoallv PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_compile_options(hccl_custom_alltoallv PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>
)

target_link_directories(hccl_custom_alltoallv PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_custom_alltoallv PRIVATE
    -Wl,--no-as-needed
    hcomm
    c_sec
    unified_dlog
    ascendc_runtime
    profapi
    ascendalog
    ascendcl
    runtime
    mmpa
    error_manager
    ascend_dump
    -Wl,--no-as-needed
)

# Ensure kernel object is generated when building host lib
add_dependencies(hccl_custom_alltoallv generate_kernel_obj)

install(TARGETS hccl_custom_alltoallv
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)