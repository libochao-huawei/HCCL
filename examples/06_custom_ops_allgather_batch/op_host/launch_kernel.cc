/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_kernel.h"

#include <memory>

#include "ccu_kernel_all_gather_batch_mesh1d.h"
#include "hccl_res_dl.h"
#include <hccl_ccu_res.h>
#include <hccl_rank_graph.h>

namespace ops_hccl_allgather_batch {

namespace {

HcclResult BuildChannelRequests(HcclComm comm, uint32_t rank, uint32_t rankSize,
                                std::vector<HcclChannelDesc> &channelRequests)
{
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == rank) {
            continue;
        }
        uint32_t netLayer = 0;
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, rank, remoteRank, &linkList, &listSize));
        CHK_PRT_RET(listSize == 0,
                    HCCL_ERROR("[BuildChannelRequests] no link between rank=%u and remoteRank=%u", rank, remoteRank),
                    HCCL_E_NOT_SUPPORT);
        for (uint32_t idx = 0; idx < listSize; ++idx) {
            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank = remoteRank;
            desc.localEndpoint.protocol = linkList[idx].srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = linkList[idx].srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = linkList[idx].srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = linkList[idx].dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = linkList[idx].dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = linkList[idx].dstEndpointDesc.loc;
            desc.channelProtocol = linkList[idx].linkAttr.linkProtocol;
            desc.notifyNum = 3;
            channelRequests.push_back(desc);
        }
    }
    return HCCL_SUCCESS;
}

uint64_t GetSliceSizeBytes(const HcclAllGatherItem &item)
{
    return item.sendCount * static_cast<uint64_t>(GetDataTypeSize(item.dataType));
}

uint64_t GetToken(const HcclAllGatherItem &item)
{
    return hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(item.sendBuf), GetSliceSizeBytes(item));
}

} // namespace

HcclResult InitCcuContext(HcclComm comm, const OpParam &param, CcuContext &ctx)
{
    if (ctx.initialized) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channelRequests;
    CHK_RET(BuildChannelRequests(comm, param.rank, param.rankSize, channelRequests));
    ctx.channels.resize(channelRequests.size());
    if (!channelRequests.empty()) {
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
                                   channelRequests.data(), channelRequests.size(), ctx.channels.data()));
    }

    ctx.kernelArg = std::make_shared<CcuKernelArgAllGatherBatchMesh1D>(param.rankSize, param.rank, param.itemCount);
    ctx.kernelArg->channels = ctx.channels;
    ctx.kernelCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllGatherBatchMesh1D>(arg);
    };

    void *creatorPtr = static_cast<void *>(&ctx.kernelCreator);
    void *kernelArgPtr = static_cast<void *>(ctx.kernelArg.get());
    CHK_RET(HcclCcuKernelRegister(comm, &ctx.kernelHandle, creatorPtr, kernelArgPtr));
    CHK_RET(HcclCcuKernelRegisterFinish(comm));
    ctx.initialized = true;
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(HcclComm comm, const OpParam &param, const CcuContext &ctx, aclrtStream stream)
{
    ThreadHandle thread = 0;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &thread));

    CcuAllGatherBatchItem batchItems[MAX_ITEM_COUNT] = {};
    for (uint32_t i = 0; i < param.itemCount; ++i) {
        const HcclAllGatherItem &item = param.items[i];
        const uint64_t sliceSize = GetSliceSizeBytes(item);
        batchItems[i].inputAddr = reinterpret_cast<uint64_t>(item.sendBuf);
        batchItems[i].outputAddr = reinterpret_cast<uint64_t>(item.recvBuf);
        batchItems[i].token = GetToken(item);
        batchItems[i].offset = static_cast<uint64_t>(param.rank) * sliceSize;
        batchItems[i].sliceSize = sliceSize;
    }

    auto taskArg = std::make_unique<CcuTaskArgAllGatherBatchMesh1D>(param.itemCount, batchItems);
    void *taskArgPtr = static_cast<void *>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(comm, thread, ctx.kernelHandle, taskArgPtr));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_allgather_batch
