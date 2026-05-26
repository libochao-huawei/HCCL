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

# 添加子目录
add_cann_subdirectories_relative(${HCCL_DIR}
    ${HCCL_DIR}/src
)

# 打包并签名：cann-hcomm-compat.tar.gz 这下面需要更改吗
pack_targets_and_files(
    OUTPUT_TARGET "pack_cann_hcomm_compat"
    OUTPUT "cann-hcomm-compat.tar.gz"
    TARGETS rs_device ra_adp net_co hccp_service.bin tls_adp ascend_kms
    MANIFEST "hccp_compat_bin_hash.cfg"
)
sign_file(
    OUTPUT_TARGET sign_cann_hcomm_compat
    INPUT "cann-hcomm-compat.tar.gz"
    CONFIG "${HCOMM_DIR}/scripts/sign/hcomm_check_cfg.xml"
    RESULT_VAR "cann_hcomm_compat_signed_file"
    DEPENDS pack_cann_hcomm_compat
)

# 打包并签名：cann-hccd-compat.tar.gz
pack_targets_and_files(
    OUTPUT_TARGET "pack_cann_hccd_compat"
    OUTPUT "cann-hccd-compat.tar.gz"
    TARGETS rs ra ra_hdc ra_peer hccd tls_adp ascend_kms
    MANIFEST "bin_hash.cfg"
    TAR_ROOT_DIR "lib64"
)
sign_file(
    OUTPUT_TARGET sign_cann_hccd_compat
    INPUT "cann-hccd-compat.tar.gz"
    CONFIG "${HCOMM_DIR}/scripts/sign/hcomm_check_cfg.xml"
    RESULT_VAR "cann_hccd_compat_signed_file"
    DEPENDS pack_cann_hccd_compat sign_cann_hcomm_compat
)

# 打包并签名：aicpu_hcomm.tar.gz
pack_targets_and_files(
    OUTPUT_TARGET "pack_aicpu_hcomm"
    OUTPUT "aicpu_hcomm.tar.gz"
    TARGETS ccl_kernel ccl_kernel_plf
    MANIFEST "bin_hash.cfg"
    TAR_ROOT_DIR "aicpu_kernels_device"
)
sign_file(
    OUTPUT_TARGET sign_aicpu_hcomm
    INPUT "aicpu_hcomm.tar.gz"
    CONFIG "${HCOMM_DIR}/scripts/sign/hcomm_check_cfg.xml"
    RESULT_VAR "aicpu_hcomm_signed_file"
    DEPENDS pack_aicpu_hcomm sign_cann_hccd_compat
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