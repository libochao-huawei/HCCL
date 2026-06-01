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
option(BUILD_OPEN_PROJECT "Build open hccl project." ON)
option(STATIC_MODE "Build static library mode." OFF)
option(ENABLE_BUILD_DEVICE "Build hccl device project." OFF)
option(ENABLE_BUILD_AARCH "Build hccl aarch project." OFF)
message(STATUS "STATIC_MODE=${STATIC_MODE}")

if(STATIC_MODE AND NOT ENABLE_BUILD_DEVICE)
    set(ENABLE_BUILD_DEVICE ON)
endif()

# 默认只编译 host 侧
if(NOT DEFINED PRODUCT_SIDE)
    set(PRODUCT_SIDE "host")
endif()

# 构建 Device 侧需设置 TOOLCHAIN_DIR 环境变量
if(DEFINED ENV{TOOLCHAIN_DIR} AND ENABLE_BUILD_DEVICE)
    set(ENABLE_DEVICE ON)
else()
    set(ENABLE_DEVICE OFF)
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")
message(STATUS "PRODUCT_SIDE=${PRODUCT_SIDE}")
message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "ENABLE_BUILD_DEVICE=${ENABLE_BUILD_DEVICE}")
message(STATUS "ENABLE_BUILD_AARCH=${ENABLE_BUILD_AARCH}")

set(HCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HCC_TOOLCHAIN_DIR "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc")
set(HCC_CXX_COMPILER  "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++")
set(HCC_C_COMPILER    "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc")
set(HCC_C_AR          "${HCC_TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-ar")

# 交叉编译场景(x86机器上编译arm包)
if(ENABLE_BUILD_AARCH)
    message(STATUS "TOOLCHAIN_DIR=${HCC_TOOLCHAIN_DIR}")
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(TOOLCHAIN_DIR      ${HCC_TOOLCHAIN_DIR})
    set(CMAKE_CXX_COMPILER ${HCC_CXX_COMPILER})
    set(CMAKE_C_COMPILER   ${HCC_C_COMPILER})
endif()

# 引入CANN/cmake库
include(cmake/fetch_cann_cmake.cmake)
project(hccl)

# 初始化 CANN 项目配置
init_cann_project()

# 采用 C++14 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

if(ENABLE_TEST)
    include(cmake/config.cmake)
    include(cmake/intf_pub_gccnative.cmake)
    include(cmake/function.cmake)
    include(cmake/third_party/gtest.cmake)
    add_subdirectory(test)

elseif(ENABLE_CUSTOM)
    # 自定义算子打包
    if(NOT CUSTOM_OPS_NAME)
        message(FATAL_ERROR "Custom ops name is not set")
    endif()
    if(NOT CUSTOM_OPS_PATH)
        message(FATAL_ERROR "Custom ops path is not set")
    endif()
    if(NOT CUSTOM_OPS_VENDOR)
        message(FATAL_ERROR "Custom ops vendor is not set")
    endif()

    if(NOT EXISTS ${CUSTOM_OPS_PATH})
        message(FATAL_ERROR "Custom ops path does not exist: ${CUSTOM_OPS_PATH}")
    endif()
    if(NOT EXISTS "${CUSTOM_OPS_PATH}/CMakeLists.txt")
        message(FATAL_ERROR "CMakeLists.txt not found. Please ensure it's a CMake project: ${CUSTOM_OPS_PATH}")
    endif()

    message(STATUS "Custom ops name: ${CUSTOM_OPS_NAME}")
    message(STATUS "Custom ops vendor: ${CUSTOM_OPS_VENDOR}")
    message(STATUS "Custom ops path: ${CUSTOM_OPS_PATH}")

    set(BUILD_DEVICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build_device)
    set(BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
    set(CUSTOM_OPS_OPP_PATH "opp/vendors/${CUSTOM_OPS_VENDOR}")
    set(CUSTOM_OPS_OPP_INC_PATH "${CUSTOM_OPS_OPP_PATH}/include")
    set(CUSTOM_OPS_OPP_LIB_PATH "${CUSTOM_OPS_OPP_PATH}/lib64")
    set(CUSTOM_OPS_OPP_SCRIPTS_PATH "${CUSTOM_OPS_OPP_PATH}/scripts")
    set(CUSTOM_OPS_OPP_AICPU_CONFIG_PATH "${CUSTOM_OPS_OPP_PATH}/aicpu/config")
    set(CUSTOM_OPS_OPP_AICPU_KERNEL_PATH "${CUSTOM_OPS_OPP_PATH}/aicpu/kernel")

    include(cmake/config.cmake)
    include(cmake/func.cmake)
    include(cmake/package.cmake)

    # AICPU Device 构建目标
    add_custom_target(custom_aicpu)
    # CPack 打包
    pack_custom()
    # 添加自定义算子工程
    add_subdirectory(${CUSTOM_OPS_PATH})

elseif(BUILD_OPEN_PROJECT)
    include(cmake/config.cmake)
    include(cmake/package.cmake)
    include(cmake/func.cmake)
    include(version.cmake)
    
    # 检查构建依赖
    check_pkg_build_deps("hccl")
    add_version_info_targets()
    # 配置打包
    pack_built_in()
    # 添加子目录
    add_subdirectory(src)

    # Device 侧编译构建
    if(ENABLE_DEVICE)
        include(ExternalProject)
        ExternalProject_Add(hccl_device
            SOURCE_DIR ${HCCL_DIR}/cmake/device
            BINARY_DIR ${HCCL_DEVICE_BUILD_PATH}
            CMAKE_ARGS
                -DTOOLCHAIN_DIR=${HCC_TOOLCHAIN_DIR}
                -DCMAKE_TOOLCHAIN_FILE=${HCCL_DIR}/cmake/aarch64-hcc-toolchain.cmake
                -DCMAKE_C_COMPILER=${HCC_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${HCC_CXX_COMPILER}
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                -DCMAKE_C_AR=${HCC_C_AR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${HCCL_DEVICE_BUILD_PATH}/_install
                -DASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}
                -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}
                -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}
                -DENABLE_SIGN=${ENABLE_SIGN}
                -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT}
                -DVERSION_INFO=${VERSION_INFO}
            INSTALL_COMMAND ${CMAKE_CPACK_COMMAND}
            BUILD_ALWAYS TRUE
        )

        # 安装 device 侧编译产物
        install(FILES
            ${HCCL_DEVICE_BUILD_PATH}/signatures/aicpu_hccl.tar.gz
            DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/kernel
            OPTIONAL
            COMPONENT hccl
        )
    endif()

    if(STATIC_MODE)
        include(cmake/static_postprocess.cmake)
    endif()

    set(HCCL_HEAD
        include/hccl.h
        include/hccl_mc2.h
    )

    set(HCCL_JSON
        # hccl json 待补充
    )

    install(FILES
        ${HCCL_HEAD}
        ${HCCL_JSON}
        DESTINATION ${INSTALL_INCLUDE_DIR}/hccl/ 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()

# this is a test commit
