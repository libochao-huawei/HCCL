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
#include <new> // For placement new

using namespace ops_hccl_allgather;

// --- Simplified AlgResourceCtxSerializable Definitions ---
namespace {

// Forward declarations
struct HcclChannelDesc;

// Mock dependencies (simplified versions of internal HCCL structs)
enum class AlgType { ALG_TYPE_RING = 0, ALG_TYPE_MESH = 1 }; // Placeholder
struct AlgHierarchyInfoForAllLevel {
    std::vector<std::vector<std::vector<uint32_t>>> infos;
};
struct HcclMem {
    uint32_t type; // HCCL_MEM_TYPE_DEVICE
    void* addr;
    uint64_t size;
};
struct ChannelInfo {
    bool isValid;
    uint32_t remoteRank;
    CommProtocol protocol;
    EndpointLocType locationType;
    uint32_t notifyNum;
    ChannelHandle handle;
    HcclMem remoteCclMem;
    HcclMem remoteInput;
    HcclMem remoteOutput;
};
struct TopoInfoWithNetLayerDetails {
    // Simplified TopoInfo - just padding/placeholder if not used, 
    // or key fields if accessed. 
    // We only use userRank and userRankSize in our logic.
    uint32_t userRank;
    uint32_t userRankSize;
    // ... extensive fields omitted for brevity, assuming we don't need full layout for custom op
    // UNLESS HcclEngineCtxCreate validates size? Unlikely for custom op.
};
struct CcuKernelHandle { void* ptr; };

struct AlgResourceCtxSerializable {
    AlgType algType;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    HcclMem cclMem;
    uint32_t notifyNumOnMainThread;
    uint32_t slaveThreadNum;
    std::vector<uint32_t> notifyNumPerThread;
    void* aivCommInfoPtr = nullptr;
    std::vector<ThreadHandle> threads;
    std::vector<std::vector<ChannelInfo>> channels;
    void* commInfoPtr = nullptr;
    void *npu2DpuShmemPtr = nullptr;
    void *dpu2NpuShmemPtr = nullptr;
    std::vector<uint32_t> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;
    uint32_t topoInfoSeqSize = 0;
    TopoInfoWithNetLayerDetails topoInfo;

    AlgResourceCtxSerializable() {
        // Constructor to initialize vectors
    }
};

} // namespace

// Helper to check if independent op check is needed (omitted for simplicity or kept if needed)
bool CheckHCCLIndependentOp() {
    return true; // Simplified
}

constexpr uint32_t AIV_TAG_ADDR_OFFSET = 16 * 1024;
static std::map<std::string, HcclMemHandle> g_memHandleCache;

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream) {
    // 1. Get or Create Host Context (AlgResourceCtxSerializable)
    // We use COMM_ENGINE_CPU_TS (1) for Host Context as per standard implementation
    CommEngine ctxEngine = CommEngine::COMM_ENGINE_CPU_TS;
    void* ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtxSerializable);
    bool isNewContext = false;
    
    // Attempt to get existing context
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
    
    // Cast to our struct type
    AlgResourceCtxSerializable* resCtx = static_cast<AlgResourceCtxSerializable*>(ctx);
    
    // Initialize the object in the allocated memory (placement new) CRITICAL STEP
    if (isNewContext) {
        new (resCtx) AlgResourceCtxSerializable();
    }
    
    // Store in param for later use (casting to void* or keeping strict typing if possible)
    // OpParam in common.h has `AlgResourceCtx* resCtx`. We need to match types or reinterpret_cast.
    // For now, we just reinterpret_cast back when needed or change OpParam definition if we could.
    // Since we can't change common.h easily without risk, we reinterpret_cast.
    param.resCtx = reinterpret_cast<AlgResourceCtx*>(resCtx); 
    
    // 2. Get Rank Info
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    resCtx->topoInfo.userRank = rank;
    resCtx->topoInfo.userRankSize = rankSize;
    
    // 3. Get CCL Buffer (Scratch)
    CHK_RET(HcclGetHcclBuffer(comm, &resCtx->cclMem.addr, &resCtx->cclMem.size));

    // 4. Create/Get AIV Comm Info Buffer (Device Memory)
    // We use COMM_ENGINE_AIV
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
    
    resCtx->aivCommInfoPtr = aivCommInfoPtr;

    // 5. Create Channels
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
    
    // We store channels in resCtx->channels (vector of vector of ChannelInfo)
    // Flattened or just one level for simplicity
    if (!channelDescs.empty()) {
        if (resCtx->channels.empty()) {
            resCtx->channels.resize(1); // One level
        }
        // Need to acquire channels and convert to ChannelInfo
        std::vector<ChannelHandle> handles(channelDescs.size());
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, channelDescs.data(), channelDescs.size(), handles.data()));
        
        // Store info
        for (size_t i = 0; i < channelDescs.size(); i++) {
             ChannelInfo info;
             info.isValid = true;
             info.remoteRank = channelDescs[i].remoteRank;
             info.protocol = channelDescs[i].channelProtocol;
             info.handle = handles[i];
             // Fetch remote buffers
             void* remoteAddr = nullptr;
             uint64_t remoteSize = 0;
             HcclChannelGetHcclBuffer(comm, handles[i], &remoteAddr, &remoteSize);
             info.remoteCclMem = {0, remoteAddr, remoteSize}; // Type 0 is device? HCCL_MEM_TYPE_DEVICE
             
             uint32_t mNum = 0;
             CommMem* rMems = nullptr;
             char** tags = nullptr;
             HcclChannelGetRemoteMems(comm, handles[i], &mNum, &rMems, &tags);
             if (mNum > 0 && rMems) {
                 info.remoteInput = {0, rMems[0].addr, rMems[0].size};
             }
             
             resCtx->channels[0].push_back(info);
        }
    }
    
    // 6. Fill AIV Comm Info Buffer content
    std::vector<uint64_t> buffersIn(MAX_RANK_SIZE, 0);
    std::vector<uint64_t> buffersOut(MAX_RANK_SIZE, 0);
    
    // My info
    if (rank < MAX_RANK_SIZE) {
        buffersIn[rank] = (uint64_t)resCtx->cclMem.addr;
        buffersOut[rank] = (uint64_t)resCtx->aivCommInfoPtr;
    }
    
    // Remote info
    if (!resCtx->channels.empty()) {
        for (const auto& chan : resCtx->channels[0]) {
            uint32_t rRank = chan.remoteRank;
            if (rRank < MAX_RANK_SIZE) {
                buffersIn[rRank] = (uint64_t)chan.remoteCclMem.addr;
                buffersOut[rRank] = (uint64_t)chan.remoteInput.addr;
            }
        }
    }
    
    // Copy to device (aivCommInfo)
    ACLCHECK(aclrtMemcpy(aivCommInfoPtr, MAX_RANK_SIZE * sizeof(uint64_t), buffersIn.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
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
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllGather_%s_Custom", commName);
    if (ret <= 0) return HCCL_E_INTERNAL;
    
    HCCL_INFO("[HcclAllGatherCustom] Preparing resources...");
    CHK_RET(PrepareResources(comm, param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Resources prepared.");
    
    // Use reinterpret_cast to access our struct fields via the stored pointer
    // Note: param.resCtx was set to (AlgResourceCtx*)resCtx in PrepareResources
    // We cast it back to our local AlgResourceCtxSerializable* to access fields
    AlgResourceCtxSerializable* resCtx = reinterpret_cast<AlgResourceCtxSerializable*>(param.resCtx);
    
    uint32_t rank = resCtx->topoInfo.userRank;
    uint32_t rankSize = resCtx->topoInfo.userRankSize;
    
    param.buffIn = (uint64_t)resCtx->aivCommInfoPtr; // Passed as buffIn to kernel
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
    
    HCCL_INFO("[HcclAllGatherCustom] Launching kernel...");
    CHK_RET(LaunchKernel(param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Launch returned.");
    
    return HCCL_SUCCESS;
}

