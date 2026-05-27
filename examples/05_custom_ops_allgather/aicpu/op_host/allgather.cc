/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <iostream>

#include "log.h"
#include "utils.h"
#include "common.h"
#include "hccl_custom_allgather.h"
#include "load_kernel.h"
#include "launch_kernel.h"

using namespace ops_hccl_allgather;

HcclResult HcclAllGatherCustom(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    if (ret <= 0) {
        HCCL_ERROR("[HcclAllGatherCustom] Failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(GetDeviceType(&param.devType));
    if (param.devType != DEVICE_TYPE_A5) {
        HCCL_ERROR("[HcclAllGatherCustom] Not Support Device Type [%u]", param.devType);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_INFO("[HcclAllGatherCustom] commName: %s", param.commName);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;

    void * ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        // device资源已经存在
        HCCL_INFO("[HcclAllGatherCustom] Engine context already exists");
        param.resCtxDevice = static_cast<AlgResourceCtx *>(ctx);
    } else {
        // 不存在，新创建Context
        HCCL_INFO("[HcclAllGatherCustom] Creating engine context");
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, size, &ctx));
        param.resCtxDevice = static_cast<AlgResourceCtx *>(ctx);
        AlgResourceCtx resCtxHost;

        // ==============================================
        // STEP 2.1: 申请host/device同步thread
        // ==============================================
        // 将传入的stream转换为thread，并申请1个notify；同时导出为AICPU上可用的thread
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &param.cpuThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, COMM_ENGINE_AICPU_TS, &resCtxHost.cpuThreadOnAicpu));
        
        // 创建一个AICPU_TS类型的thread，并申请1个notify；同时导出为CPU上可用的thread
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &resCtxHost.aicpuThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================
        CHK_RET(HcclAllocAlgResourceAICPU(comm, param, resCtxHost));
        CHK_RET(HcclMemcpyCtxHostToDevice(comm, param, resCtxHost, &param.resCtxDevice, &param.ctxSize));
    }
    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(LaunchKernel(param, stream));

    return HCCL_SUCCESS;
}
