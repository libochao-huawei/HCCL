/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_custom_allgather.h"
#include "launch_kernel.h"
#include "common.h"
#include <vector>
#include <cstring>
#include <string>
#include <hccl_rank_graph.h>

using namespace ops_hccl_allgather;

constexpr uint32_t AIV_TAG_ADDR_OFFSET = 16 * 1024;

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream) {
    std::string aivTagStr = std::string(param.tag) + "_AIV";
    const char* aivTag = aivTagStr.c_str();
    
    void* aivCommInfoPtr = nullptr;
    uint64_t aivCommInfoSize = AIV_TAG_BUFF_LEN;
    HcclMemHandle memHandle;
    
    auto hcclRet = HcclEngineCtxGet(comm, aivTag, CommEngine::COMM_ENGINE_AIV, &aivCommInfoPtr, &aivCommInfoSize);
    HCCL_INFO("[PrepareResources] HcclEngineCtxGet ret=%d ptr=%p", hcclRet, aivCommInfoPtr);

    if (hcclRet != HCCL_SUCCESS || aivCommInfoPtr == nullptr) {
        hcclRet = HcclEngineCtxCreate(comm, aivTag, CommEngine::COMM_ENGINE_AIV, AIV_TAG_BUFF_LEN, &aivCommInfoPtr);
        if (hcclRet != HCCL_SUCCESS) {
            HCCL_ERROR("[PrepareResources] Failed to create AIV buffer. ret=%d", hcclRet);
            return hcclRet;
        }
        HCCL_INFO("[PrepareResources] Created AIV buffer %p", aivCommInfoPtr);
        ACLCHECK(aclrtMemset(aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        
        CommMem regMem{COMM_MEM_TYPE_DEVICE, aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        hcclRet = HcclCommMemReg(comm, aivTag, &regMem, &memHandle);
        if (hcclRet != HCCL_SUCCESS) {
             HCCL_ERROR("[PrepareResources] Failed to register memory. ret=%d", hcclRet);
             return hcclRet;
        }
        HCCL_INFO("[PrepareResources] Registered AIV memory handle");
    }
    
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[%s]local cclBufferAddr[%p] cclBufferSize[%llu]", __func__, cclBufferAddr, cclBufferSize);

    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[rank] = cclBufferAddr;
    buffersOut[rank] = aivCommInfoPtr;

    // 获取子通信域的建链请求

    std::vector<HcclChannelDesc> level0ChannelRequest;
    level0ChannelRequest.reserve(rankSize);

    for (size_t remoteRank = 0; remoteRank < rankSize; remoteRank++)
    {
        if (remoteRank == rank) continue;

        uint32_t netLayer = 0;
        CommLink *linkList = nullptr;
        uint32_t listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, rank, remoteRank, &linkList, &listSize));
        for (uint32_t idx = 0; idx < listSize; idx++) {
            HcclChannelDesc channelDesc;
            HcclChannelDescInit(&channelDesc, 1);
            channelDesc.memHandles = &memHandle;
            channelDesc.memHandleNum = 1;
            channelDesc.remoteRank = remoteRank;
            CommLink link = linkList[idx];
            channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
            channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            HCCL_DEBUG("[CalcChannelRequestMesh1D] local device phyId: %u, remote device phyId: %u.",
                        channelDesc.localEndpoint.loc.device.devPhyId,
                        channelDesc.remoteEndpoint.loc.device.devPhyId);
            HCCL_INFO("[CalcChannelRequestMesh1D] Add channel request between %zu and %zu, netLayerIdx %u, "
                        "linkListIdx %u, protocol %zu",
                        rank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
            channelDesc.channelProtocol = link.linkAttr.linkProtocol;
            constexpr uint32_t NORMAL_NOTIFY_NUM = 3;
            channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
            level0ChannelRequest.push_back(channelDesc);
        }
    }

    // 获取子通信域的建链数量
    uint32_t validChannelNum = level0ChannelRequest.size();
    std::vector<ChannelHandle> levelNChannels;
    levelNChannels.resize(validChannelNum);

    if (validChannelNum > 0) {
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, level0ChannelRequest.data(),
            validChannelNum, levelNChannels.data()));
    }

    for (uint32_t idx = 0; idx < validChannelNum; idx++) {
        HcclChannelDesc &channelDesc = level0ChannelRequest[idx];
        void* remoteBufferAddr;
        uint64_t remoteBufferSize;
        CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
        HCCL_INFO("[%s]remoteRank[%u] cclBufferAddr[%p] cclBufferSize[%llu]", __func__, channelDesc.remoteRank,
            remoteBufferAddr, remoteBufferSize);
        buffersIn[channelDesc.remoteRank] = remoteBufferAddr;

        uint32_t memNum;
        CommMem* remoteMems;
        char** memTags;
        CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
        CHK_PRT_RET(memNum != 1,
            HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum[%u] not equal to 1", __func__, memNum), HCCL_E_PARA);
        HCCL_INFO("[%s]remoteRank[%u] memNum[%u] regMemAddr[%p] regMemSize[%llu] memTag[%s]", __func__,
            channelDesc.remoteRank, memNum, remoteMems[0].addr, remoteMems[0].size, memTags[0]);
        buffersOut[channelDesc.remoteRank] = remoteMems[0].addr;
    }

    param.buffIn = (uint64_t)aivCommInfoPtr;
    ACLCHECK(aclrtMemcpy(aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, MAX_RANK_SIZE * sizeof(void*),
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(static_cast<uint8_t*>(aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
        buffersOut, MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);

    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclAllGatherCustom(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream) {
    HCCL_INFO("[HcclAllGatherCustom] Entry. sendCount=%lu sendBuf=%p recvBuf=%p", sendCount, sendBuf, recvBuf);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllGather_%s_Custom", commName);
    if (ret <= 0) return HCCL_E_INTERNAL;
    
    CHK_RET(PrepareResources(comm, param, stream));
        
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    
    param.buffIn = (uint64_t)param.buffIn;
    param.input = (uint64_t)sendBuf;
    param.output = (uint64_t)recvBuf;
    param.rank = rank;
    param.rankSize = rankSize;
    param.xRankSize = rankSize;
    param.yRankSize = 0;
    param.zRankSize = 0;
    param.len = sendCount; 
    param.dataType = dataType;
    param.reduceOp = 0; 
    param.root = 0; 
    param.tagId = 1; 
    
    param.inputSliceStride = sendCount;
    param.outputSliceStride = sendCount;
    
    param.repeatNum = 1;
    param.inputRepeatStride = 0;
    param.outputRepeatStride = 0;
    param.isOpBase = true;
    
    param.headCountMem = 0;
    param.tailCountMem = 0;
    param.addOneMem = 0;
    param.counterMemSize = 0;
    param.isEnableCounter = false;


    
    HCCL_INFO("[HcclAllGatherCustom] Launching kernel... rank=%u rankSize=%u", rank, rankSize);
    CHK_RET(LaunchKernel(param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Launch success");
    
    return HCCL_SUCCESS;
}

