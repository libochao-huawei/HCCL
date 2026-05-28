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

 # 查找依赖包
find_cann_package(msprof MODULE REQUIRED)
find_cann_package(securec MODULE REQUIRED)
find_cann_package(unified_dlog MODULE REQUIRED)
find_cann_package(mmpa MODULE REQUIRED)
find_cann_package(runtime MODULE REQUIRED)
find_cann_package(error_manager MODULE REQUIRED)

# 添加子目录
add_cann_subdirectories_relative(${HCCL_DIR}
    ${HCCL_DIR}/src
)

# 打包并签名：aicpu_hccl.tar.gz
pack_targets_and_files(
    OUTPUT_TARGET "pack_aicpu_hccl"
    OUTPUT "aicpu_hccl.tar.gz"
    TARGETS scatter_aicpu_kernel hccl_kernel_compat
    MANIFEST "bin_hash.cfg"
    TAR_ROOT_DIR "aicpu_kernels_device"
)
sign_file(
    OUTPUT_TARGET sign_aicpu_hccl
    INPUT "aicpu_hccl.tar.gz"
    CONFIG "${HCCL_DIR}/scripts/sign/hccl_check_cfg.xml"
    RESULT_VAR "aicpu_hcomm_signed_file"
    DEPENDS pack_aicpu_hccl
)

# 安装 cann-hcomm-compat.tar.gz
install(FILES ${cann_hcomm_compat_signed_file}
    DESTINATION ${INSTALL_DEVICE_TAR_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    PERMISSIONS OWNER_READ GROUP_READ
)
# 安装 cann-hccd-compat.tar.gz
install(FILES ${cann_hccd_compat_signed_file}
    DESTINATION ${INSTALL_DEVICE_TAR_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    PERMISSIONS OWNER_READ GROUP_READ
)
# 安装 aicpu_hcomm.tar.gz
install(FILES ${aicpu_hcomm_signed_file}
    DESTINATION ${INSTALL_CCL_KERNEL_JSON_DIR}/kernel ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)

# 设置子工程打包
set_cann_subprj_package()