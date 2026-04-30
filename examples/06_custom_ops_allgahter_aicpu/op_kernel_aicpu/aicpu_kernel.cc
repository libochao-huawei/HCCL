/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <hccl/hcomm_primitives.h>
#include "log.h"
#include "utils.h"
#include "common.h"
#include "exec_op.h"

using namespace ops_hccl_allgather;

extern "C" unsigned int HcclLaunchCustomAllGatherAicpuKernel(OpParam *param)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    AlgResourceCtx resCtxDevice;

    char *ctx = static_cast<char *>(param->resCtxDevice);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    resCtxDevice.DeSerialize(seq);
    
    // === 打印 resCtxDevice 所有字段 ===
    HCCL_INFO("resCtxDevice.aicpuThread: %p", resCtxDevice.aicpuThread);
    HCCL_INFO("resCtxDevice.cpuThreadOnAicpu: %p", resCtxDevice.cpuThreadOnAicpu);
    HCCL_INFO("resCtxDevice.cclMem.addr: %p, size: %lu", resCtxDevice.cclMem.addr, resCtxDevice.cclMem.size);
    HCCL_INFO("resCtxDevice.notifyNumOnMainThread: %u", resCtxDevice.notifyNumOnMainThread);
    HCCL_INFO("resCtxDevice.slaveThreadNum: %u", resCtxDevice.slaveThreadNum);
    
    HCCL_INFO("resCtxDevice.notifyNumPerThread size: %lu", resCtxDevice.notifyNumPerThread.size());
    for (size_t i = 0; i < resCtxDevice.notifyNumPerThread.size(); i++) {
        HCCL_INFO("  notifyNumPerThread[%lu]: %u", i, resCtxDevice.notifyNumPerThread[i]);
    }
    
    HCCL_INFO("resCtxDevice.threads size: %lu", resCtxDevice.threads.size());
    for (size_t i = 0; i < resCtxDevice.threads.size(); i++) {
        HCCL_INFO("  threads[%lu]: %p", i, resCtxDevice.threads[i]);
    }
    
    HCCL_INFO("resCtxDevice.channels size: %lu", resCtxDevice.channels.size());
    for (size_t i = 0; i < resCtxDevice.channels.size(); i++) {
        const auto& ch = resCtxDevice.channels[i];
        HCCL_INFO("  channels[%lu]: isValid=%d, remoteRank=%u, protocol=%u, locationType=%u, notifyNum=%u, handle=%lu",
            i, ch.isValid, ch.remoteRank, static_cast<uint32_t>(ch.protocol), 
            static_cast<uint32_t>(ch.locationType), ch.notifyNum, ch.handle);
        HCCL_INFO("    remoteCclMem.addr: %p, size: %lu", ch.remoteCclMem.addr, ch.remoteCclMem.size);
    }
    // ====================================

    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed start batch mode");
        return 1;
    }

    // 主thread等待Host stream的通知
    if (HcommThreadNotifyWaitOnThread(resCtxDevice.aicpuThread, 0, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to wait notify from host main stream");
        return 1;
    }

    // 执行算法编排
    if (ExecOp(*param, resCtxDevice) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for op:%d", param->opType);
        return 1;
    }

    // 主thread通知Host stream
    if (HcommThreadNotifyRecordOnThread(resCtxDevice.aicpuThread, resCtxDevice.cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to record host main stream");
        return 1;
    }

    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed end batch mode");
        return 1;
    }
    HCCL_INFO("%s success, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    return 0;
}
