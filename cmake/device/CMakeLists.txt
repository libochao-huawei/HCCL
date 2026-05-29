# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(PRODUCT_SIDE "device")
set(HCCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE PATH "Project top directory")
include(${HCCL_DIR}/cmake/fetch_cann_cmake.cmake)
project(hccl_device)

# 初始化 CANN 项目配置
init_cann_project(PREPEND_MODULE_PATH)

message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
message(STATUS "ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}")
message(STATUS "CANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}")

# cmake 配置
include(${HCCL_DIR}/cmake/config.cmake)
include(${HCCL_DIR}/cmake/func.cmake)

# AIV kernel 已在 host 侧（ENABLE_BUILD_AARCH=ON）编译安装，device 侧跳过避免重复部署
set(HCCL_CANN_COMPAT_850 ON)

# 添加子目录
add_cann_subdirectories_relative(${HCCL_DIR}
    ${HCCL_DIR}/src
)

# 打包并签名：aicpu_hccl.tar.gz
set(CCL_KERNEL_TAR_DIR ${HCCL_DIR}/build_device/ccl_kernel_tar_pkg/)
set(CCL_KERNEL_TAR_LIBS
    ${CCL_KERNEL_TAR_DIR}/aicpu_kernels_device/libscatter_aicpu_kernel.so
    ${CCL_KERNEL_TAR_DIR}/aicpu_kernels_device/libhccl_kernel_compat.so
)
pack_targets_and_files(
    OUTPUT_TARGET "generate_device_aicpu_package"
    OUTPUT "aicpu_hccl.tar.gz"
    FILES ${CCL_KERNEL_TAR_LIBS}
    MANIFEST "bin_hash.cfg"
    TAR_ROOT_DIR "aicpu_kernels_device"
)
add_dependencies(generate_device_aicpu_package scatter_aicpu_kernel)
sign_file(
    INPUT "aicpu_hccl.tar.gz"
    CONFIG "${HCCL_DIR}/scripts/sign/hccl_check_cfg.xml"
    RESULT_VAR "aicpu_hccl_sign_file"
    DEPENDS generate_device_aicpu_package
)
add_dependencies(sign_aicpu_hccl generate_device_aicpu_package)

# 安装 aicpu_hccl.tar.gz
install(FILES ${aicpu_hccl_sign_file}
    DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/kernel 
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

# 设置子工程打包
set_cann_subprj_package()