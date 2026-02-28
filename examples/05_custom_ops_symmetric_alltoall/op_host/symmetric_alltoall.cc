/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "hccl/hcomm_primitives.h"
#include "log.h"
#include "common.h"
#include "hccl_custom_symmetric_alltoall.h"
#include "load_kernel.h"
#include "launch_kernel.h"

using namespace std;
using namespace ops_hccl_symmetric_alltoall;

HcclResult HcclAllToAllCustom(
    void* sendBuf, uint64_t count, HcclDataType dataType, void* recvBuf, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_symmetric_alltoall");
    if (ret <= 0) {
        HCCL_ERROR("[HcclAllToAllCustom] Failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(HcclGetCommName(comm, param.commName));

    param.sendBuf = sendBuf;
    param.recvBuf = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLTOALL;

    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    param.rank = rank;
    param.rankSize = rankSize;

    uint64_t dataSize = count * SIZE_TABLE[static_cast<uint32_t>(dataType)];

    CommSymWindow sendWin;
    CommSymWindow recvWin;
    size_t sendOffset;
    size_t recvOffset;
    CHK_RET(HcclCommSymWinGet(comm, sendBuf, dataSize, &sendWin, &sendOffset));
    CHK_RET(HcclCommSymWinGet(comm, recvBuf, dataSize, &recvWin, &recvOffset));
    param.sendWin = sendWin;
    param.recvWin = recvWin;
    param.sendOffset = sendOffset;
    param.recvOffset = recvOffset;

    HCCL_INFO("[HcclAllToAllCustom] rank=%u, rankSize=%u, count=%lu, dataType=%d, "
              "sendBuf=%p, recvBuf=%p, sendWin=%p, recvWin=%p",
              rank, rankSize, static_cast<unsigned long>(count), static_cast<int>(dataType), sendBuf, recvBuf, sendWin, recvWin);

    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;
    CHK_RET(LoadAICPUKernel());

    void* ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = static_cast<AlgResourceCtx*>(ctx);
    } else {
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, size, &ctx));
        param.resCtx = static_cast<AlgResourceCtx*>(ctx);

        ACLCHECK(aclrtCreateNotify(&(ops_hccl_symmetric_alltoall::g_notifies[0]), ACL_NOTIFY_DEFAULT));
        ACLCHECK(aclrtCreateNotify(&(ops_hccl_symmetric_alltoall::g_notifies[1]), ACL_NOTIFY_DEFAULT));
        AlgResourceCtx resCtxHost;
        for (uint32_t idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; idx++) {
            ACLCHECK(aclrtGetNotifyId(ops_hccl_symmetric_alltoall::g_notifies[idx], &(resCtxHost.notifyIds[idx])));
        }
        CHK_RET(HcclThreadAcquire(comm, engine, 1, 0, &(resCtxHost.threadHandle)));

        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = (rank + 1) % rankSize;
        channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
        channelDesc.notifyNum = 2;
        CHK_RET(HcclChannelAcquire(comm, engine, &channelDesc, 1, &(resCtxHost.channelHandle)));

        CHK_RET(HcclGetHcclBuffer(comm, &(resCtxHost.localBuffer.addr), &(resCtxHost.localBuffer.size)));
        CHK_RET(HcclChannelGetHcclBuffer(comm, resCtxHost.channelHandle, &(resCtxHost.remoteBuffer.addr),
                                         &(resCtxHost.remoteBuffer.size)));

        resCtxHost.sendWin = sendWin;
        resCtxHost.recvWin = recvWin;
        resCtxHost.sendOffset = sendOffset;
        resCtxHost.recvOffset = recvOffset;

        ACLCHECK(aclrtMemcpy(param.resCtx, size, &resCtxHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    CHK_RET(LaunchKernel(param, stream));

    return HCCL_SUCCESS;
}
