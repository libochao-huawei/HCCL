/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "resource.h"

#include <atomic>
#include <securec.h>

#include "launch_kernel.h"

namespace ops_hccl_p2p_aiv {
namespace {

std::atomic<unsigned long long> g_tagCounter {0};

HcclResult FillTag(HcclComm comm, uint32_t peerRank, const char *opName, aclrtStream stream, P2pAivResource *resource)
{
    CHK_PTR_NULL(resource);
    CHK_RET(HcclGetCommName(comm, resource->commName));
    const unsigned long long tagId = ++g_tagCounter;
    int ret = sprintf_s(resource->tag, sizeof(resource->tag), "%s_%s_r%u_p%u_s%p_id%llu", opName,
        resource->commName, resource->rank, peerRank, stream, tagId);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("failed to build tag for peerRank=%u", peerRank), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TryFillChannelDescFromRankGraph(HcclComm comm, uint32_t rank, uint32_t peerRank, HcclChannelDesc *desc)
{
    CHK_PTR_NULL(desc);
    for (uint32_t netLayer = 0; netLayer < MAX_NET_LAYER; ++netLayer) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, rank, peerRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            continue;
        }
        desc->localEndpoint = links[0].srcEndpointDesc;
        desc->remoteEndpoint = links[0].dstEndpointDesc;
        desc->channelProtocol = links[0].linkAttr.linkProtocol;
        return HCCL_SUCCESS;
    }
    desc->channelProtocol = COMM_PROTOCOL_HCCS;
    return HCCL_SUCCESS;
}

HcclResult PrepareAivCommInfo(HcclComm comm, P2pAivResource *resource, HcclMemHandle *memHandle)
{
    CHK_PTR_NULL(resource);
    CHK_PTR_NULL(memHandle);

    void *aivCommInfo = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, resource->tag, COMM_ENGINE_AIV, resource->aivCommInfoSize, &aivCommInfo));
    ACLCHECK(aclrtMemset(aivCommInfo, resource->aivCommInfoSize, 0, resource->aivCommInfoSize));

    CommMem regMem {COMM_MEM_TYPE_DEVICE, aivCommInfo, resource->aivCommInfoSize};
    CHK_RET(HcclCommMemReg(comm, resource->tag, &regMem, memHandle));
    resource->aivCommInfo = aivCommInfo;
    return HCCL_SUCCESS;
}

HcclResult ValidateTransferLength(const P2pAivResource &resource, uint64_t lenBytes)
{
    CHK_PRT_RET(resource.localBuffer.addr == nullptr || resource.remoteBuffer.addr == nullptr,
        HCCL_ERROR("local/remote buffer not ready"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.aivCommInfo == nullptr || resource.remoteAivCommInfo == nullptr,
        HCCL_ERROR("local/remote aiv comm info not ready"), HCCL_E_INTERNAL);
    CHK_PRT_RET(lenBytes > resource.localBuffer.size,
        HCCL_ERROR("lenBytes=%llu exceeds local buffer size=%llu",
            static_cast<unsigned long long>(lenBytes),
            static_cast<unsigned long long>(resource.localBuffer.size)), HCCL_E_PARA);
    CHK_PRT_RET(lenBytes > resource.remoteBuffer.size,
        HCCL_ERROR("lenBytes=%llu exceeds remote buffer size=%llu",
            static_cast<unsigned long long>(lenBytes),
            static_cast<unsigned long long>(resource.remoteBuffer.size)), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

void BuildKernelParam(const P2pAivResource &resource, uint32_t taskType, HcclDataType dataType, uint64_t lenBytes,
    uint64_t inputAddr, uint64_t outputAddr, P2pAivKernelParam *param)
{
    *param = {};
    param->version = kP2pAivKernelCtxVersion;
    param->taskType = taskType;
    param->dataType = static_cast<uint32_t>(dataType);
    param->rank = resource.rank;
    param->peerRank = resource.peerRank;
    param->blockNum = kP2pAivKernelBlockNum;
    param->tag = kP2pAivTagValue;
    param->lenBytes = lenBytes;
    param->inputAddr = inputAddr;
    param->outputAddr = outputAddr;
    param->localBufferAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
    param->remoteBufferAddr = reinterpret_cast<uint64_t>(resource.remoteBuffer.addr);
    param->localCommInfoAddr = reinterpret_cast<uint64_t>(resource.aivCommInfo);
    param->remoteCommInfoAddr = reinterpret_cast<uint64_t>(resource.remoteAivCommInfo);
}

} // namespace

HcclResult PrepareP2pAivResource(
    HcclComm comm, uint32_t peerRank, const char *opName, aclrtStream stream, P2pAivResource *resource)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(resource);
    CHK_PTR_NULL(opName);
    CHK_PTR_NULL(stream);

    CHK_RET(HcclGetRankId(comm, &resource->rank));
    CHK_RET(HcclGetRankSize(comm, &resource->rankSize));
    CHK_PRT_RET(peerRank >= resource->rankSize, HCCL_ERROR("peerRank=%u invalid, rankSize=%u", peerRank,
        resource->rankSize), HCCL_E_PARA);
    CHK_PRT_RET(peerRank == resource->rank, HCCL_ERROR("self peer is not supported in this example"), HCCL_E_PARA);
    resource->peerRank = peerRank;
    CHK_RET(FillTag(comm, peerRank, opName, stream, resource));

    HcclMemHandle memHandle = nullptr;
    CHK_RET(PrepareAivCommInfo(comm, resource, &memHandle));

    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    channelDesc.remoteRank = peerRank;
    channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
    channelDesc.memHandles = &memHandle;
    channelDesc.memHandleNum = 1;
    CHK_RET(TryFillChannelDescFromRankGraph(comm, resource->rank, peerRank, &channelDesc));

    ChannelHandle channel = 0;
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AIV, &channelDesc, 1, &channel));
    resource->channelHandle = channel;

    CHK_RET(HcclGetHcclBuffer(comm, &resource->localBuffer.addr, &resource->localBuffer.size));
    CHK_RET(HcclChannelGetHcclBuffer(comm, resource->channelHandle, &resource->remoteBuffer.addr,
        &resource->remoteBuffer.size));

    uint32_t memNum = 0;
    CommMem *remoteMems = nullptr;
    char **memTags = nullptr;
    CHK_RET(HcclChannelGetRemoteMems(comm, resource->channelHandle, &memNum, &remoteMems, &memTags));
    CHK_PRT_RET(memNum != 1 || remoteMems == nullptr, HCCL_ERROR("unexpected remote mem num=%u", memNum),
        HCCL_E_INTERNAL);
    resource->remoteAivCommInfo = remoteMems[0].addr;
    return HCCL_SUCCESS;
}

HcclResult ExecuteSend(const P2pAivResource &resource, const void *sendBuf, uint64_t lenBytes, HcclDataType dataType,
    aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(stream);
    CHK_RET(ValidateTransferLength(resource, lenBytes));
    if (lenBytes == 0) {
        return HCCL_SUCCESS;
    }

    ACLCHECK(aclrtMemset(resource.aivCommInfo, kP2pAivFlagAreaSize, 0, kP2pAivFlagAreaSize));
    P2pAivKernelParam param;
    BuildKernelParam(resource, kP2pAivTaskSend, dataType, lenBytes, reinterpret_cast<uint64_t>(sendBuf), 0, &param);
    return LaunchKernel(param, stream);
}

HcclResult ExecuteRecv(const P2pAivResource &resource, void *recvBuf, uint64_t lenBytes, HcclDataType dataType,
    aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    CHK_RET(ValidateTransferLength(resource, lenBytes));
    if (lenBytes == 0) {
        return HCCL_SUCCESS;
    }

    ACLCHECK(aclrtMemset(resource.aivCommInfo, kP2pAivFlagAreaSize, 0, kP2pAivFlagAreaSize));
    P2pAivKernelParam param;
    BuildKernelParam(resource, kP2pAivTaskRecv, dataType, lenBytes, 0, reinterpret_cast<uint64_t>(recvBuf), &param);
    return LaunchKernel(param, stream);
}

} // namespace ops_hccl_p2p_aiv
