# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(HCCL_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HCCL_REPO_ROOT}/src/common
    ${HCCL_REPO_ROOT}/src/common/hcomm_dlsym
    ${HCCL_REPO_ROOT}/src/ops/op_common/inc
    ${HCCL_REPO_ROOT}/src/ops/op_common/template
    ${HCCL_REPO_ROOT}/src/ops/op_common/template/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/acl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/base
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/include/dump
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/platform
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/dump
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/mmpa
)

add_library(hccl_custom_allgather_batch SHARED)
set_target_properties(hccl_custom_allgather_batch PROPERTIES LINKER_LANGUAGE CXX)

target_sources(hccl_custom_allgather_batch PRIVATE
    all_gather_batch.cc
    host_utils.cc
    launch_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/ccu_kernel_all_gather_batch_mesh1d.cc
    ${HCCL_REPO_ROOT}/src/ops/op_common/template/ccu/ccu_kernel_alg_base.cc
    ${HCCL_REPO_ROOT}/src/ops/op_common/template/ccu/ccu_kernel_utils.cc
)

target_include_directories(hccl_custom_allgather_batch PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)

target_compile_options(hccl_custom_allgather_batch PRIVATE
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O2
    -std=c++17
    -fstack-protector-all
)
target_compile_definitions(hccl_custom_allgather_batch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)
target_link_options(hccl_custom_allgather_batch PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_directories(hccl_custom_allgather_batch PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(hccl_custom_allgather_batch PRIVATE
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
install(TARGETS hccl_custom_allgather_batch
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
