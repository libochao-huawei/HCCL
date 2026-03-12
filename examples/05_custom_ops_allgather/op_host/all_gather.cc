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
#include <map>

using namespace ops_hccl_allgather;

// Helper to check if independent op check is needed (omitted for simplicity or kept if needed)
bool CheckHCCLIndependentOp() {
    return true; // Simplified
}

constexpr uint32_t AIV_TAG_ADDR_OFFSET = 16 * 1024;
static std::map<std::string, HcclMemHandle> g_memHandleCache;

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream) {
    // 1. Get or Create Host Context (AlgResourceCtx)
    // We use COMM_ENGINE_CPU_TS for Host Context as per standard implementation
    CommEngine ctxEngine = CommEngine::COMM_ENGINE_CPU_TS;
    void* ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    bool isNewContext = false;
    
    HcclResult hcclRet = HcclEngineCtxGet(comm, param.tag, ctxEngine, &ctx, &size);
    if (hcclRet != HCCL_SUCCESS || ctx == nullptr) {
         HCCL_INFO("[PrepareResources] Context not found (ret=%d), creating new with COMM_ENGINE_CPU_TS...", hcclRet);
         hcclRet = HcclEngineCtxCreate(comm, param.tag, ctxEngine, size, &ctx);
         if (hcclRet != HCCL_SUCCESS) {
             HCCL_ERROR("[PrepareResources] Failed to allocate context memory via HcclEngineCtxCreate. ret=%d", hcclRet);
             return hcclRet;
         }
         isNewContext = true;
    }
    param.resCtx = static_cast<AlgResourceCtx*>(ctx);
    
    // Initialize the object in the allocated memory (placement new)
    if (isNewContext) {
        new (param.resCtx) AlgResourceCtx();
    }
    AlgResourceCtx* resCtx = param.resCtx;
    
    // 2. Get Rank Info
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    
    // 3. Get CCL Buffer (Scratch)
    CHK_RET(HcclGetHcclBuffer(comm, &resCtx->cclMem.addr, &resCtx->cclMem.size));

    // 4. Create/Get AIV Comm Info Buffer (Device Memory)
    // We use COMM_ENGINE_AIV
    // Use a separate tag for AIV buffer or just append suffix
    std::string aivTagStr = std::string(param.tag) + "_AIV";
    const char* aivTag = aivTagStr.c_str();
    
    void* aivCommInfoPtr = nullptr;
    uint64_t aivCommInfoSize = AIV_TAG_BUFF_LEN;
    HcclMemHandle memHandle;

    hcclRet = HcclEngineCtxGet(comm, aivTag, CommEngine::COMM_ENGINE_AIV, &aivCommInfoPtr, &aivCommInfoSize);
    if (hcclRet != HCCL_SUCCESS || aivCommInfoPtr == nullptr) {
        // Create new AIV buffer
        HCCL_INFO("[PrepareResources] AIV buffer not found, creating new...");
        hcclRet = HcclEngineCtxCreate(comm, aivTag, CommEngine::COMM_ENGINE_AIV, AIV_TAG_BUFF_LEN, &aivCommInfoPtr);
        if (hcclRet != HCCL_SUCCESS) {
            HCCL_ERROR("[PrepareResources] Failed to create AIV buffer. ret=%d", hcclRet);
            return hcclRet;
        }
        ACLCHECK(aclrtMemset(aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        
        // Register memory to comm
        CommMem regMem{COMM_MEM_TYPE_DEVICE, aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        HCCL_INFO("[PrepareResources] Registering AIV memory to comm...");
        hcclRet = HcclCommMemReg(comm, aivTag, &regMem, &memHandle);
        if (hcclRet != HCCL_SUCCESS) {
             HCCL_ERROR("[PrepareResources] Failed to register memory. ret=%d", hcclRet);
             return hcclRet;
        }
        g_memHandleCache[aivTagStr] = memHandle;
    } else {
        // Retrieve cached handle
        if (g_memHandleCache.find(aivTagStr) == g_memHandleCache.end()) {
             HCCL_WARNING("[PrepareResources] AIV buffer found but handle not in cache. Re-registering...");
             CommMem regMem{COMM_MEM_TYPE_DEVICE, aivCommInfoPtr, AIV_TAG_BUFF_LEN};
             hcclRet = HcclCommMemReg(comm, aivTag, &regMem, &memHandle);
             if (hcclRet != HCCL_SUCCESS) return hcclRet;
             g_memHandleCache[aivTagStr] = memHandle;
        } else {
             memHandle = g_memHandleCache[aivTagStr];
        }
    }
    
    resCtx->aivCommInfo.addr = aivCommInfoPtr;
    resCtx->aivCommInfo.size = AIV_TAG_BUFF_LEN;

    // 5. Create Channels
    // We need to pass the registered memory handle to HcclChannelAcquire
    std::vector<HcclChannelDesc> channelDescs;
    for (uint32_t r = 0; r < rankSize; r++) {
        if (r == rank) continue;
        HcclChannelDesc desc;
        HcclChannelDescInit(&desc, 1);
        desc.remoteRank = r;
        desc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
        desc.memHandles = &memHandle;
        desc.memHandleNum = 1;
        channelDescs.push_back(desc);
    }
    
    if (!channelDescs.empty()) {
        resCtx->channels.resize(channelDescs.size());
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, channelDescs.data(), channelDescs.size(), resCtx->channels.data()));
    }
    
    // 6. Fill AIV Comm Info Buffer content
    std::vector<uint64_t> buffersIn(MAX_RANK_SIZE, 0);
    std::vector<uint64_t> buffersOut(MAX_RANK_SIZE, 0);
    
    // My info
    if (rank < MAX_RANK_SIZE) {
        buffersIn[rank] = (uint64_t)resCtx->cclMem.addr;
        buffersOut[rank] = (uint64_t)resCtx->aivCommInfo.addr;
    }
    
    // Remote info
    for (size_t i = 0; i < resCtx->channels.size(); i++) {
        uint32_t remoteRank = channelDescs[i].remoteRank;
        
        // Remote CCL Buffer
        void* remoteCclAddr = nullptr;
        uint64_t remoteCclSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, resCtx->channels[i], &remoteCclAddr, &remoteCclSize));
        if (remoteRank < MAX_RANK_SIZE) {
            buffersIn[remoteRank] = (uint64_t)remoteCclAddr;
        }
        
        // Remote AIV Info Buffer
        uint32_t memNum = 0;
        CommMem* remoteMems = nullptr;
        char** memTags = nullptr;
        CHK_RET(HcclChannelGetRemoteMems(comm, resCtx->channels[i], &memNum, &remoteMems, &memTags));
        if (memNum > 0 && remoteMems != nullptr) {
            if (remoteRank < MAX_RANK_SIZE) {
                buffersOut[remoteRank] = (uint64_t)remoteMems[0].addr;
            }
        }
    }
    
    // Copy to device (aivCommInfo)
    // Offset 0: buffersIn
    ACLCHECK(aclrtMemcpy(aivCommInfoPtr, MAX_RANK_SIZE * sizeof(uint64_t), buffersIn.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    
    // Offset AIV_TAG_ADDR_OFFSET: buffersOut
    ACLCHECK(aclrtMemcpy((uint8_t*)aivCommInfoPtr + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(uint64_t), buffersOut.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    
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
