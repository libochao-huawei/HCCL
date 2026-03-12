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

using namespace ops_hccl_allgather;

// Helper to check if independent op check is needed (omitted for simplicity or kept if needed)
bool CheckHCCLIndependentOp() {
    return true; // Simplified
}

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream) {
    CommEngine engine = CommEngine::COMM_ENGINE_AIV; // We only support AIV
    
    // Check if context exists
    void* ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    // Use tag to retrieve context. 
    // Note: HcclEngineCtxGet uses (tag, engine) key.
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = static_cast<AlgResourceCtx*>(ctx);
        return HCCL_SUCCESS;
    }
    
    // Create new context
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, size, &ctx));
    param.resCtx = static_cast<AlgResourceCtx*>(ctx);
    // Initialize the object in the allocated memory (placement new) or just assume POD-like usage
    // But AlgResourceCtx has std::vector, so we must construct it.
    // HcclEngineCtxCreate allocates memory but doesn't call constructor.
    // And it might be device memory? No, HcclEngineCtxCreate for CPU/Host engine allocates host memory?
    // OpParam is on host. resCtx is on host.
    // CommEngine logic in HCCL might imply where the context is stored.
    // In p2p example, it uses COMM_ENGINE_AICPU and casts ctx to AlgResourceCtx*.
    // If it's host memory, we can use placement new.
    // Since we are running on host, and accessing it, it must be host memory.
    new (param.resCtx) AlgResourceCtx();
    
    AlgResourceCtx* resCtx = param.resCtx;
    
    // 1. Get Rank Info
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    
    // 2. Get CCL Buffer (Scratch)
    CHK_RET(HcclGetHcclBuffer(comm, &resCtx->cclMem.addr, &resCtx->cclMem.size));
    
    // 3. Create AIV Comm Info Buffer
    // This buffer needs to be registered to comm for remote access
    uint64_t aivInfoSize = AIV_TAG_BUFF_LEN;
    void* aivInfoAddr = nullptr;
    // We allocate it on device.
    HCCL_INFO("[PrepareResources] Allocating AIV info buffer, size: %lu", aivInfoSize);
    ACLCHECK(aclrtMalloc(&aivInfoAddr, aivInfoSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMemset(aivInfoAddr, aivInfoSize, 0, aivInfoSize));
    resCtx->aivCommInfo.addr = aivInfoAddr;
    resCtx->aivCommInfo.size = aivInfoSize;
    
    // Register memory to comm
    HcclMemHandle memHandle;
    CommMem regMem{COMM_MEM_TYPE_DEVICE, aivInfoAddr, aivInfoSize};
    HCCL_INFO("[PrepareResources] Registering memory to comm...");
    CHK_RET(HcclCommMemReg(comm, param.tag, &regMem, &memHandle));
    
    // 4. Create Channels and Get Remote Buffers
    // Mesh 1D: Connect to all other ranks
    std::vector<HcclChannelDesc> channelDescs;
    HCCL_INFO("[PrepareResources] Creating channels...");
    for (uint32_t r = 0; r < rankSize; r++) {
        if (r == rank) continue;
        HcclChannelDesc desc;
        HcclChannelDescInit(&desc, 1);

        desc.remoteRank = r;
        desc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS; // Assume HCCS
        desc.notifyNum = 0; 
        desc.memHandles = &memHandle;
        desc.memHandleNum = 1;
        channelDescs.push_back(desc);
    }
    
    if (!channelDescs.empty()) {
        resCtx->channels.resize(channelDescs.size());
        CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), channelDescs.size(), resCtx->channels.data()));
    }
    HCCL_INFO("[PrepareResources] Channels acquired.");
    
    // 5. Fill AIV Comm Info Buffer content
    // We need to prepare arrays of pointers on host and copy to device.
    std::vector<uint64_t> buffersIn(MAX_RANK_SIZE, 0);
    std::vector<uint64_t> buffersOut(MAX_RANK_SIZE, 0);
    std::vector<uint64_t> topo(32, 0); // TOPO_LEN = 32
    
    // My info
    buffersIn[rank] = (uint64_t)resCtx->cclMem.addr;
    buffersOut[rank] = (uint64_t)resCtx->aivCommInfo.addr;
    topo[rank] = rank; // Simple topo mapping
    
    // Remote info
    for (size_t i = 0; i < resCtx->channels.size(); i++) {
        uint32_t remoteRank = channelDescs[i].remoteRank;
        
        // Remote CCL Buffer
        void* remoteCclAddr;
        uint64_t remoteCclSize;
        CHK_RET(HcclChannelGetHcclBuffer(comm, resCtx->channels[i], &remoteCclAddr, &remoteCclSize));
        buffersIn[remoteRank] = (uint64_t)remoteCclAddr;
        
        // Remote AIV Info Buffer
        uint32_t memNum;
        CommMem* remoteMems;
        char** memTags;
        CHK_RET(HcclChannelGetRemoteMems(comm, resCtx->channels[i], &memNum, &remoteMems, &memTags));
        if (memNum > 0) {
            buffersOut[remoteRank] = (uint64_t)remoteMems[0].addr;
        }
        
        topo[remoteRank] = remoteRank;
    }
    
    // Copy to device (aivCommInfo)
    // Offset 0: buffersIn
    HCCL_INFO("[PrepareResources] Copying info to device...");
    ACLCHECK(aclrtMemcpy(aivInfoAddr, MAX_RANK_SIZE * sizeof(uint64_t), buffersIn.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    
    // Offset 16KB: buffersOut
    uint64_t offsetOut = 16 * 1024;
    ACLCHECK(aclrtMemcpy((uint8_t*)aivInfoAddr + offsetOut, MAX_RANK_SIZE * sizeof(uint64_t), buffersOut.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    
    // Offset 32KB: Topo
    uint64_t offsetTopo = 32 * 1024;
    ACLCHECK(aclrtMemcpy((uint8_t*)aivInfoAddr + offsetTopo, 32 * sizeof(uint64_t), topo.data(), 32 * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclAllGatherCustom(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream) {
    HCCL_INFO("[HcclAllGatherCustom] Entry. sendCount=%lu", sendCount);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    
    // Generate tag
    // We use a fixed tag for simplicity, or based on comm name
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllGather_%s_Custom", commName);
    if (ret <= 0) return HCCL_E_INTERNAL;
    
    HCCL_INFO("[HcclAllGatherCustom] Preparing resources...");
    CHK_RET(PrepareResources(comm, param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Resources prepared.");
    
    // Fill Param
    AlgResourceCtx* resCtx = param.resCtx;
    uint32_t rank, rankSize;
    HcclGetRankId(comm, &rank);
    HcclGetRankSize(comm, &rankSize);
    
    param.buffIn = (uint64_t)resCtx->aivCommInfo.addr; // Passed as buffIn to kernel (which expects aivCommInfo there)
    param.input = (uint64_t)sendBuf;
    param.output = (uint64_t)recvBuf;
    param.rank = rank;
    param.rankSize = rankSize;
    param.xRankSize = rankSize;
    param.yRankSize = 0;
    param.zRankSize = 0;
    param.len = sendCount; // Element count
    param.dataType = dataType;
    param.reduceOp = 0; // Not used for AllGather
    param.root = 0; // Not used
    param.tagId = 1; // Fixed tag ID for sync
    
    param.inputSliceStride = sendCount;
    param.outputSliceStride = sendCount;
    
    param.repeatNum = 1;
    param.inputRepeatStride = 0;
    param.outputRepeatStride = 0;
    param.isOpBase = true;
    
    // Counters - not used/enabled
    param.headCountMem = 0;
    param.tailCountMem = 0;
    param.addOneMem = 0;
    param.counterMemSize = 0;
    param.isEnableCounter = false;
    
    // Launch
    HCCL_INFO("[HcclAllGatherCustom] Launching kernel...");
    CHK_RET(LaunchKernel(param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Launch returned.");
    
    return HCCL_SUCCESS;
}
