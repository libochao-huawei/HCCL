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

find_package(ASC REQUIRED)

project(hccl_examples_custom_ops_allgather LANGUAGES CXX ASC)
set(CMAKE_SYSTEM_PROCESSOR "x86_64")

if (NOT DEFINED CUSTOM_OPS_OPP_LIB_PATH)
    set(CUSTOM_OPS_OPP_LIB_PATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if (NOT DEFINED CUSTOM_OPS_OPP_INC_PATH)
    set(CUSTOM_OPS_OPP_INC_PATH ${CMAKE_INSTALL_PREFIX}/include)
endif()

set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

add_library(hccl_custom_allgather SHARED)
set_target_properties(hccl_custom_allgather PROPERTIES LINKER_LANGUAGE CXX)

target_sources(hccl_custom_allgather PRIVATE
    op_host/all_gather.cc
    op_host/launch_kernel.cc
)

# Manual compilation of ASC kernel to object file for dynamic loading
set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/op_host/launch_kernel_asc.asc)
set(KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hccl_custom_allgather_kernels.o)

# Include paths for ASC compiler
set(ASC_INCLUDE_FLAGS 
    -I${CMAKE_CURRENT_SOURCE_DIR}/inc
    -I${CMAKE_CURRENT_SOURCE_DIR}/op_host
    -I${ASCEND_CANN_PACKAGE_PATH}/include
    -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    -I${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

# Use CMAKE_ASC_COMPILER if available, otherwise default to 'asc'
if(NOT CMAKE_ASC_COMPILER)
    set(CMAKE_ASC_COMPILER "asc")
endif()

add_custom_command(
    OUTPUT ${KERNEL_OBJ}
    COMMAND ${CMAKE_ASC_COMPILER} -c -o ${KERNEL_OBJ} ${KERNEL_SRC} --npu-arch=dav-c310 ${ASC_INCLUDE_FLAGS}
    DEPENDS ${KERNEL_SRC}
    COMMENT "Compiling ASC kernel: ${KERNEL_SRC} -> ${KERNEL_OBJ}"
    VERBATIM
)

add_custom_target(generate_kernel_obj ALL DEPENDS ${KERNEL_OBJ})
add_dependencies(hccl_custom_allgather generate_kernel_obj)

# Copy the generated .o file to testcase directory for runtime loading
add_custom_command(TARGET hccl_custom_allgather POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${KERNEL_OBJ}
        ${CMAKE_CURRENT_SOURCE_DIR}/testcase/hccl_custom_allgather_kernels.o
)

target_include_directories(hccl_custom_allgather PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)

target_compile_options(hccl_custom_allgather PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O2
    -std=c++17
    -fstack-protector-all
)
target_compile_definitions(hccl_custom_allgather PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)
target_link_options(hccl_custom_allgather PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_compile_options(hccl_custom_allgather PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>
)

target_link_directories(hccl_custom_allgather PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_custom_allgather PRIVATE
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

install(TARGETS hccl_custom_allgather
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
install(FILES ${KERNEL_OBJ}
    DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
install(FILES
    inc/hccl_custom_allgather.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
)

add_subdirectory(testcase)
