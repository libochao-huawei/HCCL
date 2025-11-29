# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_LIB_DIR ${TOP_DIR}/output/ascend/host/ubuntu20.04/x86_64/lib)
add_library(checker_fwk SHARED "")

add_subdirectory(fakeso)
add_subdirectory(framework)
add_subdirectory(sim_world)
add_subdirectory(hccl_proxy)
add_subdirectory(common)
add_subdirectory(pub_inc)

target_link_libraries(checker_fwk PUBLIC
    $<BUILD_INTERFACE:intf_llt_pub>
    c_sec
    # 链接fakeso
    hccl_fwk
    hccl_alg
    hccl_plf
    ascendcl
    unified_dlog
    ascendalog
    mmpa
    # 链接hccl库
    ${HCCL_LIB_DIR}/libhccl.so
    ${HCCL_LIB_DIR}/libhccl_scatter.so
)

target_include_directories(checker_fwk PUBLIC
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/pub_inc
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/pub_inc/utils
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/framework/mem_conflict_check
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/framework/semantics_check
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/framework/singletask_check
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/framework/task_graph_generator
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/sim_world
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/common/exception
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/common/utils
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/framework

    ${TOP_DIR}/hcomm/inc
    ${TOP_DIR}/hcomm/inc/hccl
    ${TOP_DIR}/hcomm/pkg_inc  # hccl/base.h
    ${TOP_DIR}/hcomm/pkg_inc/hccl
    ${TOP_DIR}/hcomm/src/pub_inc
    ${TOP_DIR}/inc/external
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/ace/npuruntime/inc/runtime
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/slog/inc/toolchain  # dlog_pub.h
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/hccl_proxy
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/hccl_proxy/communicator
    ${TOP_DIR}/hccl/test/utils/vm-plugins/verifier/src/hccl_proxy/topo_model
    ${TOP_DIR}/hcomm/src/framework/inc
    ${TOP_DIR}/hcomm/inc/hccl/hccl
    ${TOP_DIR}/hcomm/externel_depends/toolchain
    ${TOP_DIR}/hcomm/src/algorithm/pub_inc
    ${TOP_DIR}/hcomm/src/algorithm/base/alg_template
    ${TOP_DIR}/hcomm/src/algorithm/base/communicator
)