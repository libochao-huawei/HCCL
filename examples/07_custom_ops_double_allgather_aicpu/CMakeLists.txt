# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
project(hccl_examples_custom_ops_double_allgather_aicpu LANGUAGES CXX AICPU ASC)

set(HCCL_CUSTOM_OPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include/aicpu_api
)

if (NOT DEFINED CUSTOM_OPS_OPP_LIB_PATH)
    set(CUSTOM_OPS_OPP_LIB_PATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if (NOT DEFINED CUSTOM_OPS_OPP_INC_PATH)
    set(CUSTOM_OPS_OPP_INC_PATH ${CMAKE_INSTALL_PREFIX}/include)
endif()

add_library(hccl_custom_double_allgather SHARED)
set_target_properties(hccl_custom_double_allgather PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(hccl_custom_double_allgather PRIVATE ${HCCL_CUSTOM_OPS_INCLUDE_LIST})
target_compile_options(hccl_custom_double_allgather PRIVATE
    -U_FORTIFY_SOURCE -Werror -fno-common -fno-strict-aliasing -pipe -g -O0 -std=c++17 -fstack-protector-all
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
target_compile_definitions(hccl_custom_double_allgather PRIVATE _GLIBCXX_USE_CXX11_ABI=0 -DHOST_COMPILE)
target_link_options(hccl_custom_double_allgather PRIVATE -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)
target_link_directories(hccl_custom_double_allgather PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64 ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64)
target_link_libraries(hccl_custom_double_allgather PRIVATE
    hcomm c_sec unified_dlog ascendc_runtime profapi ascendalog ascendcl runtime mmpa error_manager ascend_dump
)
install(TARGETS hccl_custom_double_allgather LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH})
install(FILES inc/hccl_custom_double_allgather.h DESTINATION ${CUSTOM_OPS_OPP_INC_PATH})

set(AICPU_KERNEL_TAR_FILE ${CMAKE_BINARY_DIR}/aicpu_hccl_custom_double_allgather.tar.gz)
set(AICPU_KERNEL_JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/libdouble_allgather_aicpu_kernel.json)

if(KERNEL_MODE)
    add_library(double_allgather_aicpu_kernel SHARED)
    target_include_directories(double_allgather_aicpu_kernel PRIVATE ${HCCL_CUSTOM_OPS_INCLUDE_LIST})
    target_compile_definitions(double_allgather_aicpu_kernel PRIVATE -DAICPU_COMPILE)
    target_compile_options(double_allgather_aicpu_kernel PRIVATE -g -O0)
    target_link_directories(double_allgather_aicpu_kernel PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64 ${ASCEND_CANN_PACKAGE_PATH}/devlib/device)
    target_link_libraries(double_allgather_aicpu_kernel PRIVATE ccl_kernel)
    register_custom_aicpu_target(double_allgather_aicpu_kernel)
    package_aicpu_kernel(double_allgather_aicpu_kernel ${AICPU_KERNEL_TAR_FILE} ${AICPU_KERNEL_JSON_FILE})
else()
    install(FILES ${AICPU_KERNEL_JSON_FILE} DESTINATION ${CMAKE_INSTALL_PREFIX}/aicpu_config)
endif()

add_subdirectory(op_host)
add_subdirectory(op_kernel_aicpu)
add_subdirectory(testcase)
