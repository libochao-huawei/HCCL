/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>  // 包含getenv函数
#include <cstring>  // 包含strcmp函数
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "mmpa_api.h"
#include "param_check.h"
#include "executor_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo_host.h"
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"
#include "alg_param.h"
#include "alg_type.h"
#include "op_common_graph_mode.h"
#include "op_common.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "dpu/kernel_launch.h"

namespace ops_hccl {

HcclResult HcclExecOpGraphMode(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, std::string &algName, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute HcclExecOpGraphMode.");
    param.hcclComm = comm;
    // 在原先的commName中添加执行模式，得到commModeTag
    bool isOpBase = true;
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_%s", param.commName, opModeStr);
    if (ret <= 0) {
        HCCL_ERROR("[%s] failed to fill param.commModeTag", __func__);
        return HCCL_E_INTERNAL;
    }

    std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 资源结构体
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    // 资源序列化结果
    void *resCtxSequence;
    bool isResourceReused = false;

    ThreadHandle cpuTsThread;
    ThreadHandle exportedAicpuTsThread;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        // Export cpuTsThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    }

    CHK_RET(HcclGetAlgRes(comm, param, executor, topoInfo.get(), resCtxHost, &resCtxSequence, isResourceReused));

    ThreadHandle exportedCpuTsThread;
    ThreadHandle mainThread;
    u32 notifyNumOnMainThread;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // 获取主流信息
        CHK_RET(GetMainThreadInfo(comm, param, mainThread, notifyNumOnMainThread));
        // Export mainThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
        // cpuTsThread 添加到param里
        param.opThread = exportedAicpuTsThread;
    }

    // 算法执行
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclAicpuKernelEntranceLaunch(comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread,
            resCtxSequence, algName));
    } else if (param.engine == COMM_ENGINE_AIV) {
        param.resCtx = resCtxSequence;
        AlgResourceCtxSerializable &resCtxHost = *static_cast<AlgResourceCtxSerializable *>(resCtxSequence);
        CHK_RET(HcclAivKernelEntranceLaunch(param, topoInfo, resCtxHost));
        CHK_RET(executor->Orchestrate(param, resCtxHost));
    } else {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char *ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    }
    HCCL_INFO("Execute HcclExecOp success.");
    return HCCL_SUCCESS;
}
}  // namespace ops_hccl