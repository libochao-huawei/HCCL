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
#include <mutex>
#include <string>
#include <unordered_map>

#include "ccu_kernel_all_gather_batch_mesh1d.h"
#include "ccu_resource_flow.h"
#include "host_utils.h"
#include <hccl_ccu_res.h>
#include <hccl_rank_graph.h>

namespace ops_hccl_allgather_batch {

namespace {

std::mutex g_ctxCacheMutex;
std::unordered_map<std::string, CcuContextData> g_ctxByTag;

HcclResult BuildChannelRequests(HcclComm comm, uint32_t rank, uint32_t rankSize,
                                std::vector<HcclChannelDesc> &channelRequests)
{
    HCCL_INFO("[BuildChannelRequests] begin, rank=%u rankSize=%u", rank, rankSize);
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
        HcclChannelDesc desc;
        HcclChannelDescInit(&desc, 1);
        desc.remoteRank = remoteRank;
        desc.localEndpoint.protocol = linkList[0].srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = linkList[0].srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = linkList[0].srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = linkList[0].dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = linkList[0].dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = linkList[0].dstEndpointDesc.loc;
        desc.channelProtocol = linkList[0].linkAttr.linkProtocol;
        desc.notifyNum = 3;
        channelRequests.push_back(desc);
        HCCL_INFO("[BuildChannelRequests] add remoteRank=%u protocol=%u notifyNum=%u",
                  remoteRank, desc.channelProtocol, desc.notifyNum);
    }
    HCCL_INFO("[BuildChannelRequests] end, channelCount=%zu", channelRequests.size());
    return HCCL_SUCCESS;
}

} // namespace

HcclResult InitCcuContext(HcclComm comm, const char *engineCtxTag, const OpParam &param, aclrtStream stream,
                          CcuContextData *&ctx)
{
    HCCL_INFO("[InitCcuContext] begin, tag=%s rank=%u rankSize=%u itemCount=%u",
              engineCtxTag, param.rank, param.rankSize, param.itemCount);
    {
        std::lock_guard<std::mutex> guard(g_ctxCacheMutex);
        ctx = &g_ctxByTag[engineCtxTag];
        if (ctx->initialized) {
            HCCL_INFO("[InitCcuContext] context already initialized, kernelCount=%zu",
                      ctx->resCtx.ccuKernels.size());
            return HCCL_SUCCESS;
        }
    }

    std::vector<HcclChannelDesc> channelRequests;
    CHK_RET(BuildChannelRequests(comm, param.rank, param.rankSize, channelRequests));

    AlgResourceRequest resRequest;
    resRequest.notifyNumOnMainThread = 0;
    resRequest.slaveThreadNum = 0;
    resRequest.ccuKernelNum.push_back(1);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllGatherBatchMesh1D>(arg);
    };
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllGatherBatchMesh1D>(param.rankSize, param.rank, param.itemCount);
    kernelInfo.channels = channelRequests;
    resRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_INFO("[InitCcuContext] allocating CCU resources");
    CHK_RET(HcclAllocAlgResourceCcu(comm, stream, resRequest, ctx->resCtx));
    ctx->initialized = true;
    HCCL_INFO("[InitCcuContext] success");
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(HcclComm comm, const OpParam &param, const CcuContextData &ctx, aclrtStream stream)
{
    (void)comm;
    (void)stream;

    CcuAllGatherBatchItem batchItems[MAX_ITEM_COUNT] = {};
    PackedBatchItem packedItems[MAX_ITEM_COUNT] = {};
    PackBatchItemsForLaunch(param, packedItems, MAX_ITEM_COUNT);
    for (uint32_t i = 0; i < param.itemCount; ++i) {
        batchItems[i].inputAddr = packedItems[i].inputAddr;
        batchItems[i].outputAddr = packedItems[i].outputAddr;
        batchItems[i].token = packedItems[i].token;
        batchItems[i].offset = packedItems[i].offset;
        batchItems[i].sliceSize = packedItems[i].sliceSize;
    }

    auto taskArg = std::make_unique<CcuTaskArgAllGatherBatchMesh1D>(param.itemCount, batchItems);
    void *taskArgPtr = static_cast<void *>(taskArg.get());
    CHK_PRT_RET(ctx.resCtx.threads.empty(),
                HCCL_ERROR("[LaunchKernel] ctx thread list is empty"),
                HCCL_E_INTERNAL);
    CHK_PRT_RET(ctx.resCtx.ccuKernels.empty(),
                HCCL_ERROR("[LaunchKernel] ctx kernel list is empty"),
                HCCL_E_INTERNAL);
    CHK_RET(HcclCcuKernelLaunch(comm, ctx.resCtx.threads[0], ctx.resCtx.ccuKernels[0], taskArgPtr));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_allgather_batch
