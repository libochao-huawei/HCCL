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
#include <hccl_rank_graph.h>
#include <vector>
#include <cstring>
#include <algorithm>

#include "log.h"
#include "common.h"
#include "hccl_custom_alltoallv_aicpu.h"
#include "load_kernel.h"
#include "launch_kernel.h"

using namespace ops_hccl_alltoallv_aicpu;

static HcclResult GetDeviceType(DeviceType* devType)
{
    uint32_t deviceId = 0;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[GetDeviceType] aclrtGetDevice failed, ret[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }

    aclrtDeviceInfo info;
    aclRet = aclrtGetDeviceInfo(deviceId, &info);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[GetDeviceType] aclrtGetDeviceInfo failed, ret[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }

    switch (info.socName) {
        case ACL_RT_SOC_NAME_910B:
            *devType = DEVICE_TYPE_A2;
            break;
        case ACL_RT_SOC_NAME_910_95:
            *devType = DEVICE_TYPE_950;
            break;
        default:
            *devType = DEVICE_TYPE_A3;
            break;
    }
    return HCCL_SUCCESS;
}

static HcclResult AcquireChannels(HcclComm comm, CommEngine engine, uint32_t rank, uint32_t rankSize,
                                  std::vector<ChannelHandle>& channelHandles)
{
    std::vector<HcclChannelDesc> channelDescs;
    for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank) continue;

        uint32_t netLayer = 0, listSize = 0;
        CommLink *linkList = nullptr;
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
            desc.notifyNum = 2;
            channelDescs.push_back(desc);
        }
    }

    uint32_t validNum = channelDescs.size();
    channelHandles.resize(validNum);
    if (validNum > 0) {
        CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), validNum, channelHandles.data()));
    }
    
    return HCCL_SUCCESS;
}

static HcclResult PrepareResources(HcclComm comm, OpParam& param, uint32_t rank, uint32_t rankSize)
{
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;
    
    void *ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[PrepareResources] Engine context already exists");
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[PrepareResources] Creating engine context");
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, size, &ctx));
    param.resCtx = static_cast<AlgResourceCtx *>(ctx);
    
    AlgResourceCtx resCtxHost;
    
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, nullptr, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, COMM_ENGINE_AICPU_TS, &resCtxHost.cpuThreadOnAicpu));
    
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, rankSize - 1, &resCtxHost.mainThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.mainThread, COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));
    
    resCtxHost.slaveThreads.resize(rankSize - 1);
    for (uint32_t i = 0; i < rankSize - 1; i++) {
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &resCtxHost.slaveThreads[i]));
    }
    
    resCtxHost.notifyNumOnMainThread = rankSize - 1;
    resCtxHost.notifyNumPerThread.resize(rankSize - 1, 1);
    
    CHK_RET(AcquireChannels(comm, engine, rank, rankSize, resCtxHost.channelHandles));
    
    CHK_RET(HcclGetHcclBuffer(comm, &resCtxHost.cclBuffer.addr, &resCtxHost.cclBuffer.size));
    
    ACLCHECK(aclrtMemcpy(param.resCtx, size, &resCtxHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
    
    HCCL_INFO("[PrepareResources] Resources prepared successfully");
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclAlltoAllVCustomAicpu(void *sendBuf, void *sendCounts, void *sdispls, void *recvBuf, 
    void *recvCounts, void *rdispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("[HcclAlltoAllVCustomAicpu] Entry. sendBuf=%p recvBuf=%p", sendBuf, recvBuf);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AlltoAllV_Aicpu_%s", commName);
    if (ret <= 0) return HCCL_E_INTERNAL;
    
    uint32_t rank = 0, rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    CHK_RET(GetDeviceType(&param.devType));
    
    param.rank = rank;
    param.rankSize = rankSize;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    
    param.all2AllVDataDes.sendCounts = static_cast<uint64_t*>(sendCounts);
    param.all2AllVDataDes.recvCounts = static_cast<uint64_t*>(recvCounts);
    param.all2AllVDataDes.sdispls = static_cast<uint64_t*>(sdispls);
    param.all2AllVDataDes.rdispls = static_cast<uint64_t*>(rdispls);
    param.all2AllVDataDes.sendType = dataType;
    param.all2AllVDataDes.recvType = dataType;
    
    uint64_t totalSendCount = 0;
    uint64_t totalRecvCount = 0;
    uint64_t dataTypeSize = SIZE_TABLE[dataType];
    for (uint32_t i = 0; i < rankSize; i++) {
        totalSendCount += param.all2AllVDataDes.sendCounts[i];
        totalRecvCount += param.all2AllVDataDes.recvCounts[i];
    }
    param.inputSize = totalSendCount * dataTypeSize;
    param.outputSize = totalRecvCount * dataTypeSize;
    param.count = totalSendCount;
    
    CHK_RET(PrepareResources(comm, param, rank, rankSize));
    
    HCCL_INFO("[HcclAlltoAllVCustomAicpu] Launching kernel... rank=%u rankSize=%u", rank, rankSize);
    CHK_RET(LaunchKernel(param, stream));
    HCCL_INFO("[HcclAlltoAllVCustomAicpu] Launch success");
    
    return HCCL_SUCCESS;
}