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

enum class AlgType { ALG_TYPE_RING = 0, ALG_TYPE_MESH = 1 };
struct AlgHierarchyInfoForAllLevel {
    std::vector<std::vector<std::vector<uint32_t>>> infos;
};
struct HcclMem {
    uint32_t type;
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
    uint32_t userRank;
    uint32_t userRankSize;
};
struct CcuKernelHandle { void* ptr; };

struct AlgResourceCtxSerializable {
    AlgType algType; // 环境变量设置的算法类型
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 算法分层信息
    HcclMem cclMem; // 跨Rank缓存Buffer
    uint32_t notifyNumOnMainThread; // 主流上的notify数量
    uint32_t slaveThreadNum; // 需要的thread数量
    std::vector<uint32_t> notifyNumPerThread; // 每个thread需要的notify数量
    void* aivCommInfoPtr = nullptr;
    std::vector<ThreadHandle> threads;
    std::vector<std::vector<ChannelInfo>> channels;
    void* commInfoPtr = nullptr;
    // hostdpu
    void *npu2DpuShmemPtr = nullptr;
    void *dpu2NpuShmemPtr = nullptr;
    // ccu的
    std::vector<uint32_t> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;
    uint32_t topoInfoSeqSize = 0;
    TopoInfoWithNetLayerDetails topoInfo; // 提取的拓扑信息

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;

        binaryStream << algType;
        binaryStream << algHierarchyInfo.infos;
        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << slaveThreadNum;
        binaryStream << notifyNumPerThread;
        binaryStream << commInfoPtr;
        binaryStream << threads;
        binaryStream << channels;

        binaryStream << npu2DpuShmemPtr;
        binaryStream << dpu2NpuShmemPtr;

        binaryStream << ccuKernelNum;
        binaryStream << ccuKernels;
        std::vector<char> seq = topoInfo.Serialize();
        topoInfoSeqSize = seq.size();
        binaryStream << topoInfoSeqSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        result.insert(result.end(), seq.begin(), seq.end());

        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);

        binaryStream >> algType;
        binaryStream >> algHierarchyInfo.infos;
        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> slaveThreadNum;
        binaryStream >> notifyNumPerThread;
        binaryStream >> commInfoPtr;
        binaryStream >> threads;
        binaryStream >> channels;

        binaryStream >> npu2DpuShmemPtr;
        binaryStream >> dpu2NpuShmemPtr;

        binaryStream >> ccuKernelNum;
        binaryStream >> ccuKernels;
        binaryStream >> topoInfoSeqSize;
        size_t startPos = data.size() - topoInfoSeqSize;
        std::vector<char> tailData(data.begin() + startPos, data.end());
        TopoInfoWithNetLayerDetails topoTemp;
        topoTemp.DeSerialize(tailData);
        topoInfo = std::move(topoTemp);
    }
};

} // namespace

bool CheckHCCLIndependentOp() {
    return true;
}

constexpr uint32_t AIV_TAG_ADDR_OFFSET = 16 * 1024;
static std::map<std::string, HcclMemHandle> g_memHandleCache;

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream) {
    CommEngine ctxEngine = CommEngine::COMM_ENGINE_CPU_TS;
    void* ctx = nullptr;
    uint64_t ctxSize = 0;
    bool isNewContext = false;

    HcclResult hcclRet = HcclEngineCtxGet(comm, param.tag, ctxEngine, &ctx, &ctxSize);
    AlgResourceCtxSerializable* hostCtx = new AlgResourceCtxSerializable();
    if (hcclRet != HCCL_SUCCESS || ctx == nullptr) {
        HCCL_INFO("[PrepareResources] Context not found (ret=%d), creating new with COMM_ENGINE_CPU_TS...", hcclRet);
        isNewContext = true;
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);


        uint32_t rank, rankSize;
        CHK_RET(HcclGetRankId(comm, &rank));
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);
        CHK_RET(HcclGetRankSize(comm, &rankSize));
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);
        hostCtx->topoInfo.userRank = rank;
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);
        hostCtx->topoInfo.userRankSize = rankSize;
        HCCL_INFO("[PrepareResources] Rank %u, RankSize %u", rank, rankSize);

        CHK_RET(HcclGetHcclBuffer(comm, &hostCtx->cclMem.addr, &hostCtx->cclMem.size));
        HCCL_INFO("[PrepareResources] Got HCCL Buffer addr=%p size=%lu", hostCtx->cclMem.addr, hostCtx->cclMem.size);
        
    std::string aivTagStr = std::string(param.tag) + "_AIV";
    const char* aivTag = aivTagStr.c_str();
    
    void* aivCommInfoPtr = nullptr;
    uint64_t aivCommInfoSize = AIV_TAG_BUFF_LEN;
    HcclMemHandle memHandle;
    
    hcclRet = HcclEngineCtxGet(comm, aivTag, CommEngine::COMM_ENGINE_AIV, &aivCommInfoPtr, &aivCommInfoSize);
    HCCL_INFO("[PrepareResources] HcclEngineCtxGet ret=%d ptr=%p", hcclRet, aivCommInfoPtr);

    if (hcclRet != HCCL_SUCCESS || aivCommInfoPtr == nullptr) {
        hcclRet = HcclEngineCtxCreate(comm, aivTag, CommEngine::COMM_ENGINE_AIV, AIV_TAG_BUFF_LEN, &aivCommInfoPtr);
        if (hcclRet != HCCL_SUCCESS) {
            HCCL_ERROR("[PrepareResources] Failed to create AIV buffer. ret=%d", hcclRet);
            return hcclRet;
        }
        HCCL_INFO("[PrepareResources] Created AIV buffer %p", aivCommInfoPtr);
        ACLCHECK(aclrtMemset(aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);
        
        CommMem regMem{COMM_MEM_TYPE_DEVICE, aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);
        hcclRet = HcclCommMemReg(comm, aivTag, &regMem, &memHandle);
        HCCL_INFO("[PrepareResources] (line=%d)", __LINE__);
        if (hcclRet != HCCL_SUCCESS) {
             HCCL_ERROR("[PrepareResources] Failed to register memory. ret=%d", hcclRet);
             return hcclRet;
        }
        g_memHandleCache[aivTagStr] = memHandle;
        HCCL_INFO("[PrepareResources] Registered AIV memory handle");
    } else {
        if (g_memHandleCache.find(aivTagStr) == g_memHandleCache.end()) {
             CommMem regMem{COMM_MEM_TYPE_DEVICE, aivCommInfoPtr, AIV_TAG_BUFF_LEN};
             hcclRet = HcclCommMemReg(comm, aivTag, &regMem, &memHandle);
             if (hcclRet != HCCL_SUCCESS) return hcclRet;
             g_memHandleCache[aivTagStr] = memHandle;
             HCCL_INFO("[PrepareResources] Re-registered AIV memory handle (cache miss)");
        } else {
             memHandle = g_memHandleCache[aivTagStr];
        }
    }
    
    hostCtx->aivCommInfoPtr = aivCommInfoPtr;

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
            if (hostCtx->channels.empty()) {
                hostCtx->channels.resize(1);
            }
            std::vector<ChannelHandle> handles(channelDescs.size());
            CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, channelDescs.data(), channelDescs.size(), handles.data()));
            
            for (size_t i = 0; i < channelDescs.size(); i++) {
                 ChannelInfo info;
                 info.isValid = true;
                 info.remoteRank = channelDescs[i].remoteRank;
                 info.protocol = channelDescs[i].channelProtocol;
                 info.handle = handles[i];
                 void* remoteAddr = nullptr;
                 uint64_t remoteSize = 0;
                 HcclChannelGetHcclBuffer(comm, handles[i], &remoteAddr, &remoteSize);
                 info.remoteCclMem = {0, remoteAddr, remoteSize};
                 
                 uint32_t mNum = 0;
                 CommMem* rMems = nullptr;
                 char** tags = nullptr;
                 HcclChannelGetRemoteMems(comm, handles[i], &mNum, &rMems, &tags);
                 if (mNum > 0 && rMems) {
                     info.remoteInput = {0, rMems[0].addr, rMems[0].size};
                 }
                 hostCtx->channels[0].push_back(info);
            }
        }

        std::vector<char> serializedData = hostCtx->Serialize();
        ctxSize = serializedData.size();
        
        hcclRet = HcclEngineCtxCreate(comm, param.tag, ctxEngine, ctxSize, &ctx);
        if (hcclRet != HCCL_SUCCESS) {
            HCCL_ERROR("[PrepareResources] Failed to create ctx memory. ret=%d", hcclRet);
            delete hostCtx;
            return hcclRet;
        }
        
        std::memcpy(ctx, serializedData.data(), ctxSize);
        HCCL_INFO("[PrepareResources] New Context created and serialized. Size: %lu", ctxSize);

    } else {
        HCCL_INFO("[PrepareResources] Reuse Host Context %p", ctx);
        std::vector<char> serializedData(ctxSize);
        std::memcpy(serializedData.data(), ctx, ctxSize);
        hostCtx->DeSerialize(serializedData);
    }

    param.resCtx = reinterpret_cast<AlgResourceCtx*>(hostCtx);

    if (isNewContext) {
        uint32_t rank = hostCtx->topoInfo.userRank;
        std::vector<uint64_t> buffersIn(MAX_RANK_SIZE, 0);
        std::vector<uint64_t> buffersOut(MAX_RANK_SIZE, 0);
        
        if (rank < MAX_RANK_SIZE) {
            buffersIn[rank] = (uint64_t)hostCtx->cclMem.addr;
            buffersOut[rank] = (uint64_t)hostCtx->aivCommInfoPtr;
        }
        
        if (!hostCtx->channels.empty()) {
            for (const auto& chan : hostCtx->channels[0]) {
                uint32_t rRank = chan.remoteRank;
                if (rRank < MAX_RANK_SIZE) {
                    buffersIn[rRank] = (uint64_t)chan.remoteCclMem.addr;
                    buffersOut[rRank] = (uint64_t)chan.remoteInput.addr;
                }
            }
        }
        
        ACLCHECK(aclrtMemcpy(hostCtx->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(uint64_t), buffersIn.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
        ACLCHECK(aclrtMemcpy((uint8_t*)hostCtx->aivCommInfoPtr + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(uint64_t), buffersOut.data(), MAX_RANK_SIZE * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
        HCCL_INFO("[PrepareResources] AIV info copied to device");
    }
    
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
    
    AlgResourceCtxSerializable* resCtx = reinterpret_cast<AlgResourceCtxSerializable*>(param.resCtx);
    
    uint32_t rank = resCtx->topoInfo.userRank;
    uint32_t rankSize = resCtx->topoInfo.userRankSize;
    
    param.buffIn = (uint64_t)resCtx->aivCommInfoPtr;
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

    delete resCtx;
    param.resCtx = nullptr;
    
    return HCCL_SUCCESS;
}

