/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hccl_custom_alltoallv.h"
#include "common.h"
#include <vector>
#include <cstring>
#include <string>
#include <hccl_rank_graph.h>

using namespace ops_hccl_alltoallv;

static HcclResult LocalCopy(void* dst, void* src, uint64_t size) {
    if (size == 0) {
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

static HcclResult GetDataTypeSize(HcclDataType dataType, uint64_t& dataTypeSize) {
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
            dataTypeSize = sizeof(int8_t);
            break;
        case HCCL_DATA_TYPE_INT32:
            dataTypeSize = sizeof(int32_t);
            break;
        case HCCL_DATA_TYPE_FP16:
            dataTypeSize = sizeof(uint16_t);
            break;
        case HCCL_DATA_TYPE_FP32:
            dataTypeSize = sizeof(float);
            break;
        default:
            HCCL_ERROR("[GetDataTypeSize] Unsupported data type: %d", dataType);
            return HCCL_E_UNSUPPORTED;
    }
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclAllToAllVCustom(void *sendBuf, void *recvBuf, uint64_t *sendCounts, uint64_t *recvCounts,
                                          uint64_t *sdispls, uint64_t *rdispls,
                                          HcclDataType dataType, HcclComm comm, aclrtStream stream) {
    HCCL_INFO("[HcclAllToAllVCustom] Entry. sendBuf=%p recvBuf=%p", sendBuf, recvBuf);

    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(rdispls);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    uint32_t rank = 0;
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    uint64_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(dataType, dataTypeSize));

    void* cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[HcclAllToAllVCustom] CCL buffer: addr=%p, size=%lu", cclBufferAddr, cclBufferSize);

    uint64_t totalSendCount = 0;
    uint64_t totalRecvCount = 0;
    for (uint32_t i = 0; i < rankSize; i++) {
        totalSendCount += sendCounts[i];
        totalRecvCount += recvCounts[i];
    }
    HCCL_INFO("[HcclAllToAllVCustom] rank=%u, totalSend=%lu, totalRecv=%lu", rank, totalSendCount, totalRecvCount);

    std::vector<HcclChannelDesc> channelRequests;
    std::vector<ChannelHandle> channels;

    for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank) continue;

        uint32_t netLayer = 0;
        uint32_t listSize = 0;
        CommLink* linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, rank, remoteRank, &linkList, &listSize));

        for (uint32_t idx = 0; idx < listSize; idx++) {
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
            desc.notifyNum = 1;
            channelRequests.push_back(desc);
        }
    }

    if (!channelRequests.empty()) {
        uint32_t validNum = channelRequests.size();
        channels.resize(validNum);
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU,
                                    channelRequests.data(), validNum, channels.data()));
    }

    for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank) continue;

        uint64_t sendCount = sendCounts[remoteRank];
        uint64_t recvCount = recvCounts[remoteRank];
        uint64_t sendOffset = sdispls[remoteRank] * dataTypeSize;
        uint64_t recvOffset = rdispls[remoteRank] * dataTypeSize;
        uint64_t sendSize = sendCount * dataTypeSize;
        uint64_t recvSize = recvCount * dataTypeSize;

        if (sendSize > 0) {
            void* srcPtr = (uint8_t*)sendBuf + sendOffset;
            void* dstPtr = (uint8_t*)cclBufferAddr + remoteRank * (totalSendCount * dataTypeSize);

            CHK_RET(aclrtMemcpy(dstPtr, sendSize, srcPtr, sendSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
        }
    }

    CHK_RET(aclrtSynchronizeStream(stream));
    HCCL_INFO("[HcclAllToAllVCustom] Pre-copy to CCL buffer done");

    for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank) continue;

        uint64_t sendCount = sendCounts[remoteRank];
        uint64_t recvCount = recvCounts[remoteRank];
        uint64_t sendSize = sendCount * dataTypeSize;
        uint64_t recvSize = recvCount * dataTypeSize;

        if (sendSize == 0 && recvSize == 0) continue;

        void* remoteCclBuffAddr = nullptr;
        uint64_t remoteCclBuffSize = 0;

        for (size_t i = 0; i < channels.size(); i++) {
            if (channelRequests[i].remoteRank == remoteRank) {
                CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &remoteCclBuffAddr, &remoteCclBuffSize));
                break;
            }
        }

        if (remoteCclBuffAddr == nullptr) {
            HCCL_ERROR("[HcclAllToAllVCustom] Failed to get remote CCL buffer for rank %u", remoteRank);
            return HCCL_E_INTERNAL;
        }

        uint64_t srcOffset = rank * (totalSendCount * dataTypeSize);
        uint64_t dstOffset = remoteRank * (totalSendCount * dataTypeSize);

        void* sendSrc = (uint8_t*)cclBufferAddr + srcOffset;
        void* sendDst = remoteCclBuffAddr + dstOffset;
        void* recvSrc = remoteCclBuffAddr + srcOffset;
        void* recvDst = (uint8_t*)cclBufferAddr + dstOffset;

        if (sendSize > 0 && recvSize > 0) {
            CHK_RET(aclrtMemcpy(sendDst, sendSize, sendSrc, sendSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
        } else if (sendSize > 0) {
            CHK_RET(aclrtMemcpy(sendDst, sendSize, sendSrc, sendSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
        }
    }

    CHK_RET(aclrtSynchronizeStream(stream));
    HCCL_INFO("[HcclAllToAllVCustom] Network send done");

    for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank) continue;

        uint64_t recvCount = recvCounts[remoteRank];
        uint64_t recvOffset = rdispls[remoteRank] * dataTypeSize;
        uint64_t recvSize = recvCount * dataTypeSize;

        if (recvSize == 0) continue;

        uint64_t srcOffset = remoteRank * (totalSendCount * dataTypeSize);
        void* srcPtr = (uint8_t*)cclBufferAddr + srcOffset;
        void* dstPtr = (uint8_t*)recvBuf + recvOffset;

        CHK_RET(aclrtMemcpy(dstPtr, recvSize, srcPtr, recvSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }

    CHK_RET(aclrtSynchronizeStream(stream));
    HCCL_INFO("[HcclAllToAllVCustom] Post-copy to recv buffer done");

    if (sendCounts[rank] > 0 && recvCounts[rank] > 0) {
        uint64_t localSendOffset = sdispls[rank] * dataTypeSize;
        uint64_t localRecvOffset = rdispls[rank] * dataTypeSize;
        uint64_t localSize = sendCounts[rank] * dataTypeSize;

        void* srcPtr = (uint8_t*)sendBuf + localSendOffset;
        void* dstPtr = (uint8_t*)recvBuf + localRecvOffset;

        CHK_RET(aclrtMemcpy(dstPtr, localSize, srcPtr, localSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }

    CHK_RET(aclrtSynchronizeStream(stream));
    HCCL_INFO("[HcclAllToAllVCustom] Local copy done");

    return HCCL_SUCCESS;
}