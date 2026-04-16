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
#include "host_utils.h"
#include "hccl_res_dl.h"
#include <hccl_ccu_res.h>
#include <hccl_rank_graph.h>

namespace ops_hccl_allgather_batch {

namespace {

struct CcuKernelKeepAlive {
    std::shared_ptr<CcuKernelArgAllGatherBatchMesh1D> kernelArg;
    hcomm::KernelCreator kernelCreator;
};

std::mutex g_keepAliveMutex;
std::unordered_map<std::string, CcuKernelKeepAlive> g_keepAliveByTag;

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

HcclResult InitCcuContext(HcclComm comm, const char *engineCtxTag, const OpParam &param, CcuContextData *&ctx)
{
    HCCL_INFO("[InitCcuContext] begin, tag=%s rank=%u rankSize=%u itemCount=%u",
              engineCtxTag, param.rank, param.rankSize, param.itemCount);
    uint64_t ctxSize = sizeof(CcuContextData);
    void *ctxPtr = nullptr;
    HcclResult ret = HcclEngineCtxGet(comm, engineCtxTag, CommEngine::COMM_ENGINE_CCU, &ctxPtr, &ctxSize);
    HCCL_INFO("[InitCcuContext] HcclEngineCtxGet ret=%d ctxPtr=%p ctxSize=%llu", ret, ctxPtr, ctxSize);
    if (ret == HCCL_SUCCESS && ctxPtr != nullptr) {
        ctx = static_cast<CcuContextData *>(ctxPtr);
        if (ctx->initialized) {
            HCCL_INFO("[InitCcuContext] context already initialized, kernelHandle=%llu", ctx->kernelHandle);
            return HCCL_SUCCESS;
        }
    } else {
        HCCL_INFO("[InitCcuContext] creating engine ctx");
        ret = HcclEngineCtxCreate(comm, engineCtxTag, CommEngine::COMM_ENGINE_CCU, sizeof(CcuContextData), &ctxPtr);
        HCCL_INFO("[InitCcuContext] HcclEngineCtxCreate ret=%d ctxPtr=%p", ret, ctxPtr);
        CHK_RET(ret);
        ctx = static_cast<CcuContextData *>(ctxPtr);
        ctx->initialized = false;
        ctx->kernelHandle = 0;
    }

    if (ctx->initialized) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channelRequests;
    CHK_RET(BuildChannelRequests(comm, param.rank, param.rankSize, channelRequests));
    std::vector<ChannelHandle> channels(channelRequests.size());
    if (!channelRequests.empty()) {
        HCCL_INFO("[InitCcuContext] acquiring %zu CCU channels", channelRequests.size());
        ret = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
                                 channelRequests.data(), channelRequests.size(), channels.data());
        HCCL_INFO("[InitCcuContext] HcclChannelAcquire ret=%d", ret);
        CHK_RET(ret);
    }

    CcuKernelKeepAlive keepAlive;
    keepAlive.kernelArg = std::make_shared<CcuKernelArgAllGatherBatchMesh1D>(param.rankSize, param.rank, param.itemCount);
    keepAlive.kernelArg->channels = channels;
    keepAlive.kernelCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelAllGatherBatchMesh1D>(arg);
    };

    {
        std::lock_guard<std::mutex> guard(g_keepAliveMutex);
        g_keepAliveByTag[engineCtxTag] = keepAlive;
    }

    CcuKernelKeepAlive *keepAlivePtr = nullptr;
    {
        std::lock_guard<std::mutex> guard(g_keepAliveMutex);
        keepAlivePtr = &g_keepAliveByTag[engineCtxTag];
    }

    void *creatorPtr = static_cast<void *>(&keepAlivePtr->kernelCreator);
    void *kernelArgPtr = static_cast<void *>(keepAlivePtr->kernelArg.get());
    HCCL_INFO("[InitCcuContext] registering CCU kernel, creatorPtr=%p kernelArgPtr=%p", creatorPtr, kernelArgPtr);
    ret = HcclCcuKernelRegister(comm, &ctx->kernelHandle, creatorPtr, kernelArgPtr);
    HCCL_INFO("[InitCcuContext] HcclCcuKernelRegister ret=%d kernelHandle=%llu", ret, ctx->kernelHandle);
    CHK_RET(ret);
    ret = HcclCcuKernelRegisterFinish(comm);
    HCCL_INFO("[InitCcuContext] HcclCcuKernelRegisterFinish ret=%d", ret);
    CHK_RET(ret);
    ctx->initialized = true;
    HCCL_INFO("[InitCcuContext] success");
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(HcclComm comm, const OpParam &param, const CcuContextData &ctx, aclrtStream stream)
{
    ThreadHandle thread = 0;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &thread));

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
    CHK_RET(HcclCcuKernelLaunch(comm, thread, ctx.kernelHandle, taskArgPtr));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl_allgather_batch
