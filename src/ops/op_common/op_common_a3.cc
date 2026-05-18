/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hcomm_primitives.h"
#include "load_kernel.h"
#include "coll_alg_exec_registry.h"
#include "config_log.h"
#include "op_common_ops.h"
#include "topo.h"
#include "topo_host.h"
#include "hcomm_host_profiling_dl.h"
#include <algorithm>
#include <future>
#include <map>

namespace ops_hccl {
using NotifyArray = std::array<aclrtNotify, AICPU_CONTROL_NOTIFY_NUM>;
thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];
constexpr u32 DEVICE_EIGHT = 8;
constexpr u32 DEVICE_FOUR = 4;
constexpr u32 DEVICE_TWO = 2;
constexpr u32 DEVICE_ONE = 1;
constexpr u32 HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM = 8;

thread_local std::map<HcclComm, NotifyArray> g_notifiesMap;
bool IsStreamCapture(aclrtStream stream)
{
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    u64 modelId = 0xFFFFFFFF;
    CHK_PRT(haclrtGetCaptureInfo(stream, captureStatus, modelId, isCapture));
    return isCapture;
}

bool IsAiCpuMode(DevType deviceType, u32 rankSize)
{
    if (GetExternalInputHcclAicpuUnfold() == true && deviceType == DevType::DEV_TYPE_910_93 && (rankSize != 1)) {
        return true;
    }
    return false;
}

/* 算子级别基础拓扑解析，缓存在host上 */
HcclResult CalcBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo** topoInfo)
{
    uint64_t size = sizeof(TopoInfo);
    void *ctx = nullptr;
    // 若获取Context失败，表示对应Context尚未缓存
    HcclResult ret = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        // 创建新的Context
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        // 将Context内存地址强转为TopoInfo
        *topoInfo = static_cast<TopoInfo *>(ctx);
        // 将对应拓扑信息填入到Context内存中
        CHK_RET(InitRankInfo(comm, *topoInfo));
        return HCCL_SUCCESS;
    }

    *topoInfo = static_cast<TopoInfo *>(ctx);
    return HCCL_SUCCESS;
}

HcclResult GetDefaultAlgoLevel0Module(TopoInfo* topoInfo, AlgTypeLevel0 &algType)
{
    u32 deviceNumPerAggregation = topoInfo->deviceNumPerModule;
    if (deviceNumPerAggregation == DEVICE_EIGHT) {
        algType = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    } else if (deviceNumPerAggregation == DEVICE_FOUR) {
        algType = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    } else if (deviceNumPerAggregation == DEVICE_TWO) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (deviceNumPerAggregation == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        algType = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
    }

    if (!topoInfo->multiModuleDiffDeviceNumMode && topoInfo->deviceType == DevType::DEV_TYPE_910B) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to ALG_LEVEL0_NP_MESH (HCCS links is enabled).");
    }

    if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algType = topoInfo->isHCCSSWNumEqualToTwiceSIONum ? AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING :
                                                            AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to [%u].", algType);
    }
    return HCCL_SUCCESS;
}

HcclResult SetAlgoLevel0(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not supported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0Module(topoInfo, algType));
    return HCCL_SUCCESS;
}

HcclResult GetDefaultAlgoLevel1V1(TopoInfo* topoInfo, AlgTypeLevel1 &algType)
{
    u32 moduleNum = topoInfo->moduleNum;
    if (moduleNum >=  HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ?
            AlgTypeLevel1::ALG_LEVEL1_RING :
            AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    if (algType == AlgTypeLevel1::ALG_LEVEL1_HD && topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }
    HCCL_INFO("[AlgConfigurator][GetDefaultAlgoLevel1V1] algType[%u], moduleNum[%u]", algType, moduleNum);
    return HCCL_SUCCESS;
}

HcclResult SetAlgoLevel1(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel1 &algType,
    HcclCMDType opType)
{
    HcclAlgoType algoConfigShadow = algoConfig;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC:
            if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc algo is not support, set default.", topoInfo->moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE:
            if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is not support, set default.", topoInfo->moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", topoInfo->moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not supported. the config is ignored.");
        default:
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            break;
    }

    HCCL_DEBUG("[AlgConfigurator][SetAlgoLevel1] algType[%u], deviceType_[%u]", algType,
        topoInfo->deviceType);

    if (algoConfigShadow == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
        CHK_RET(GetDefaultAlgoLevel1V1(topoInfo, algType));
    }
    return HCCL_SUCCESS;
}

HcclResult SetAlgoLevel2(TopoInfo* topoInfo, HcclAlgoType algoConfig, AlgTypeLevel2 &algType)
{
    u32 superPodNum = topoInfo->superPodNum;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel2::ALG_LEVEL2_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel2::ALG_LEVEL2_RING;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel2::ALG_LEVEL2_NB;
            break;
        default: {
            // Level2默认选择NHR算法
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        }
    }
    HCCL_DEBUG("[AlgConfigurator][SetAlgoLevel2]algType[%u], deviceType_[%u], superPodNum_[%u]",
        algType, topoInfo->deviceType, superPodNum);
    return HCCL_SUCCESS;
}

HcclResult ReportProfilingThread(HcclComm comm, const OpParam &param, AlgResourceCtx *resCtxHost, TopoInfo* topoInfo)
{
    CHK_PTR_NULL(resCtxHost);
    CHK_PTR_NULL(topoInfo);
    HcomProInfoTmp profInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.tag, sizeof(profInfo.tag), param.tag));
    std::string algTypeStr = TransferAlgTypeStr(param.algType);
    CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()));
    CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.commName, sizeof(profInfo.commName), param.commName));
    profInfo.rankSize = topoInfo->userRankSize;
    profInfo.userRank = topoInfo->userRank;
    profInfo.dataCount = param.DataDes.count;
    profInfo.dataType = static_cast<uint8_t>(param.DataDes.dataType);
    
    profInfo.root = param.root;

    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        profInfo.slaveThreadNum = resCtxHost->slaveThreadNum;
        
        HCCL_DEBUG("[%s] algType[%s], commName[%s], rankSize[%u], userRank[%u], dataCount[%u], dataType[%u], slaveThreadNum[%u], root[%u]",
            __func__, profInfo.algType, profInfo.commName, profInfo.rankSize, 
            profInfo.userRank, profInfo.dataCount, profInfo.dataType, profInfo.slaveThreadNum, profInfo.root);
        
        char* curThreadPtr = reinterpret_cast<char*>(resCtxHost); // 拿到所有host下发的thread
        curThreadPtr += sizeof(AlgResourceCtx);// 偏移指针
        ThreadHandle* curThreads = reinterpret_cast<ThreadHandle *>(curThreadPtr);
        CHK_PRT(HcommProfilingRegThread(profInfo,curThreads));
    } else {
        profInfo.slaveThreadNum = 0;
        // aicpu下发 只需要上报主流
        ThreadHandle cpuTsThread;
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        CHK_PRT(HcommProfilingRegThread(profInfo, &cpuTsThread));
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgType(TopoInfo* topoInfo, HcclCMDType opType, AlgType& algType)
{
    std::vector<HcclAlgoType> ret;
    ret = GetExternalInputHcclAlgoConfig(opType);
    if (ret.size() != HCCL_ALGO_LEVEL_NUM) {
        HCCL_ERROR("alg type size is invalid");
        return HCCL_E_PARA;
    }

    AlgTypeLevel0 algType0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    AlgTypeLevel2 algType2 = AlgTypeLevel2::ALG_LEVEL2_RESERVED;

    CHK_RET(SetAlgoLevel0(topoInfo, ret[HCCL_ALGO_LEVEL_0], algType0));
    CHK_RET(SetAlgoLevel1(topoInfo, ret[HCCL_ALGO_LEVEL_1], algType1, opType));
    CHK_RET(SetAlgoLevel2(topoInfo, ret[HCCL_ALGO_LEVEL_2], algType2));

    algType.algoLevel0 = algType0;
    algType.algoLevel1 = algType1;
    algType.algoLevel2 = algType2;

    return HCCL_SUCCESS;
}

HcclResult InitCclBuffer(HcclComm comm, AlgResourceCtx* resCtxHost)
{
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    u64 sizePerCcl = cclBufferSize / 2;
    resCtxHost->cclInputMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, sizePerCcl};
    resCtxHost->cclOutputMem = HcclMem{HCCL_MEM_TYPE_DEVICE,
        static_cast<void*>(static_cast<char*>(cclBufferAddr) + sizePerCcl), sizePerCcl};
    return HCCL_SUCCESS;
}

HcclResult InitNotifyResources(HcclComm comm, AlgResourceCtx* resCtxHost)
{
    if (!HcommIsExportThreadSupported()) {
        if (g_notifiesMap[comm][0] == 0 &&
            aclrtCreateNotify(&(g_notifiesMap[comm][0]), 0x00000000U) != ACL_SUCCESS) {
            HCCL_ERROR("failed to alloc notify");
            return HCCL_E_INTERNAL;
        }
        if (g_notifiesMap[comm][1] == 0 &&
            aclrtCreateNotify(&(g_notifiesMap[comm][1]), 0x00000000U) != ACL_SUCCESS) {
            HCCL_ERROR("failed to alloc notify");
            return HCCL_E_INTERNAL;
        }
        for (u32 idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; idx++) {
            uint32_t notifyId;
            if (aclrtGetNotifyId(g_notifiesMap[comm][idx], &notifyId) != ACL_SUCCESS) {
                HCCL_ERROR("failed to get notify id");
                return HCCL_E_INTERNAL;
            }
            resCtxHost->notifyIds[idx] = notifyId;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireThreads(HcclComm comm, const OpParam& param,
    const AlgResourceRequest &resRequest, AlgResourceCtx* resCtxHost, ThreadHandle* threads)
{
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        u32 maxNotifyNum = resRequest.notifyNumOnMainThread + 1;
        for (u32 i = 0; i < resRequest.notifyNumPerThread.size(); i++) {
            if (resRequest.notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resRequest.notifyNumPerThread[i];
            }
        }
        u32 threadNum = resRequest.slaveThreadNum + 1;
        CHK_RET(HcclThreadAcquire(comm, param.engine, threadNum, maxNotifyNum, threads));
        resCtxHost->topoInfo.mainThread = *threads;
        HCCL_DEBUG("threads ptr is %p\n", *threads);
    } else {
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
            resRequest.notifyNumOnMainThread, threads));
        if (resRequest.slaveThreadNum > 0) {
            u32 maxNotifyNum = 0;
            for (u32 i = 0; i < resRequest.notifyNumPerThread.size(); i++) {
                if (resRequest.notifyNumPerThread[i] > maxNotifyNum) {
                    maxNotifyNum = resRequest.notifyNumPerThread[i];
                }
            }
            char* curPtr = reinterpret_cast<char *>(threads) + sizeof(ThreadHandle);
            ThreadHandle* slaveThreads = reinterpret_cast<ThreadHandle *>(curPtr);
            CHK_RET(HcclThreadAcquire(comm, param.engine, resRequest.slaveThreadNum, maxNotifyNum, slaveThreads));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam& param,
    const AlgResourceRequest &resRequest, AlgResourceCtx* resCtxHost, char* &curPtr)
{
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        const std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        u32 validChannelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(validChannelNum);

        if (validChannelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, levelNChannelRequest.data(),
                validChannelNum, levelNChannels.data()));
        }

        const SubCommInfo &subCommInfo = resCtxHost->algHierarchyInfo.infos[level];
        ChannelInfo* channels = reinterpret_cast<ChannelInfo *>(curPtr);
        for (u32 rank = 0; rank < subCommInfo.localRankSize; rank++) {
            channels[rank].isValid = false;
        }
        for (u32 idx = 0; idx < validChannelNum; idx++) {
            const HcclChannelDesc &channelDesc = levelNChannelRequest[idx];
            u32 levelRank;
            CHK_RET(GetSubCommRankByUserRank(channelDesc.remoteRank, level, resCtxHost->algHierarchyInfo, levelRank));
            channels[levelRank].isValid = true;
            channels[levelRank].remoteRank = channelDesc.remoteRank;
            channels[levelRank].protocol = channelDesc.channelProtocol;
            channels[levelRank].notifyNum = channelDesc.notifyNum;
            channels[levelRank].handle = levelNChannels[idx];
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            channels[levelRank].remoteInput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize};
            channels[levelRank].remoteOutput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize};
        }
        curPtr += sizeof(ChannelInfo) * subCommInfo.localRankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AllocAlgResource(HcclComm comm, const OpParam& param, AlgResourceRequest &resRequest,
    AlgResourceCtx* resCtxHost)
{
    CHK_RET(InitCclBuffer(comm, resCtxHost));

    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread.size() == 0 ? 0 :
        resRequest.notifyNumPerThread[0];

    CHK_RET(InitNotifyResources(comm, resCtxHost));

    char* curPtr = reinterpret_cast<char *>(resCtxHost) + sizeof(AlgResourceCtx);
    ThreadHandle* threads = reinterpret_cast<ThreadHandle *>(curPtr);
    CHK_RET(AcquireThreads(comm, param, resRequest, resCtxHost, threads));
    curPtr += sizeof(ThreadHandle) * (resRequest.slaveThreadNum + 1);

    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        HCCL_DEBUG("[AllocAlgResource] slaveThreadNum[%u]", resRequest.slaveThreadNum);
        for (u32 i = 0; i < resRequest.slaveThreadNum; i++) {
            HCCL_DEBUG("[AllocAlgResource] threads[%u]=[%llu]", i, threads[i]);
        }
    }

    if (HcommIsProfilingSupported()) {
        CHK_PRT(ReportProfilingThread(comm, param, resCtxHost, &(resCtxHost->topoInfo)));
    }

    CHK_RET(AcquireChannels(comm, param, resRequest, resCtxHost, curPtr));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

std::string SetLaunchMode(CommEngine engine)
{
    std::string launchMode = "UNKNOWN";
    if (engine == CommEngine::COMM_ENGINE_CPU) {
        launchMode = "HOST";
    } else if (engine == CommEngine::COMM_ENGINE_CPU_TS) {
        launchMode = "HOST_TS";
    } else if ((engine == CommEngine::COMM_ENGINE_AICPU) ||
               (engine == CommEngine::COMM_ENGINE_AICPU_TS)) {
        launchMode = "AI_CPU";
    } else if (engine == CommEngine::COMM_ENGINE_AIV) {
        launchMode = "AIV";
    }
    return launchMode;
}

HcclResult GetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<ExecutorBase> &executor,
    TopoInfo* topoInfo, AlgType& algType, AlgResourceCtx** resCtx)
{
    // 这种情况下资源已经有了
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size) == HCCL_SUCCESS) {
        *resCtx = static_cast<AlgResourceCtx *>(ctx);
        HCCL_INFO("[%s] Res Allready Exist", __func__);
        if (HcommIsProfilingSupported()) {
            CHK_PRT(ReportProfilingThread(comm, param, *resCtx, topoInfo));
        }
        return HCCL_SUCCESS;
    }

    // 资源计算
    AlgHierarchyInfo algHierarchyInfo;
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcResRequest(comm, param, topoInfo, algHierarchyInfo, resRequest, algType));

    // 开始计算资源Context的长度
    size = sizeof(AlgResourceCtx);
    // 计算变长数据区中threads占用的空间
    size += sizeof(ThreadHandle) * (resRequest.slaveThreadNum + 1);
    // 计算变长数据区中channels占用的空间
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        size += sizeof(ChannelInfo) * algHierarchyInfo.infos[level].localRankSize;
    }
    // 创建Context
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    // 将内存强转为AlgResourceCtx结构体
    *resCtx = static_cast<AlgResourceCtx *>(ctx);

    AlgResourceCtx* resCtxHost;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // AICPU模式下分配一块Host内存用于填充资源
        ACLCHECK(aclrtMallocHost(reinterpret_cast<void**>(&resCtxHost), size));
        topoInfo->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    } else {
        resCtxHost = *resCtx;
    }

    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algType = algType;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    // 创建资源，并填充到Host内存上
    HcclResult ret = AllocAlgResource(comm, param, resRequest, resCtxHost);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        if (param.engine == COMM_ENGINE_AICPU_TS) {
            ACLCHECK(aclrtFreeHost(resCtxHost));
        }
        return ret;
    }
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        topoInfo->mainThread = resCtxHost->topoInfo.mainThread;
    }
 
    CHK_RET(HcclEngineCtxCopy(comm, param.engine, param.algTag, resCtxHost, size, 0));
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // 从Host内存拷贝到Device Context内存上
        ACLCHECK(aclrtFreeHost(resCtxHost));
    }
    return HCCL_SUCCESS;
}

HcclResult FillAlgTagAndDebugInfo(OpParam &param, TopoInfo* topoInfo, AlgType& algType,
    const std::string &algName, const std::string &opNameForLog)
{
    bool isOpBase = true;
    if (isOpBase) {
        int ret = sprintf_s(param.algTag, sizeof(param.algTag), "%s_%s_%u", param.tag, algName.c_str(), param.root);
        if (ret <= 0) {
            HCCL_ERROR("failed to fill param.algTag");
            return HCCL_E_INTERNAL;
        }
    }

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        std::string opExpansionStr = SetLaunchMode(param.engine);

        const char* launchMode = (((param.engine == CommEngine::COMM_ENGINE_AICPU) ||
                                   (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ? "_device" : "_host");
        int ret = strcat_s(param.algTag, sizeof(param.algTag), launchMode);
        if (ret != 0) {
            HCCL_ERROR("failed to fill param.algTag");
            return HCCL_E_INTERNAL;
        }

        HCCL_INFO("[SelectAlg] %s algTag is [%s] algName is [%s]", opNameForLog.c_str(), param.algTag, algName.c_str());
        HCCL_CONFIG_INFO(HCCL_ALG,
                         "[%s] algTag[%s] algName[%s] userRank[%u] algType[%s] "\
                         "userRankSize[%u] level0Size[%u] moduleNum[%u] "\
                         "level2Size[%u] opExpansionMode[%s] isZeroCopy[%u] isOpBase[%u].",
                         __func__, param.algTag, algName.c_str(), topoInfo->userRank, AlgTypeToStr(algType).c_str(),
                         topoInfo->userRankSize, topoInfo->deviceNumPerModule, topoInfo->moduleNum,
                         topoInfo->superPodNum, opExpansionStr.c_str(), param.isZeroCopy, isOpBase);
    }
    return HCCL_SUCCESS;
}

HcclResult SelectAlgReduceScatter(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName)
{
    (void) comm;
    // 由于scatter只支持server间ring,nb和NHR，如果不是上述算法，需要重定向到ring；910_93仅支持server间ring
    if (!(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) &&
        !(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) &&
        !(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
        HCCL_INFO("[Reduce_Scatter] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType).c_str());
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    }

    if (topoInfo->userRankSize == 1) {
        return HCCL_E_INTERNAL;
    } else if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algName = "ReduceScatterBIRSExecutor";
    }

    CHK_RET(FillAlgTagAndDebugInfo(param, topoInfo, algType, algName, "Reduce_Scatter"));
    return HCCL_SUCCESS;
}

HcclResult SelectAlgScatter(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName)
{
    (void) comm;
    // 由于scatter只支持server间ring,nb和NHR，如果不是上述算法，需要重定向到ring；910_93仅支持server间ring
    if (!(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) &&
        !(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) &&
        !(algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)) {
        HCCL_INFO("[Scatter] algType[%s] is not supported, reset algType=ring",
            AlgTypeToStr(algType).c_str());
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    }

    if (topoInfo->userRankSize == 1) {
        algName = "ScatterSingleExecutor";
    } else if (topoInfo->multiModuleDiffDeviceNumMode || topoInfo->multiSuperPodDiffServerNumMode) {
        algName = "ScatterCommExecutor";
    } else if (topoInfo->deviceType == DevType::DEV_TYPE_910B) {
        algName = "ScatterMeshExecutor";
    } else if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        algName = "ScatterRingFor91093Executor";
    }

    CHK_RET(FillAlgTagAndDebugInfo(param, topoInfo, algType, algName, "Scatter"));
    return HCCL_SUCCESS;
}


struct ThreadResources {
    ThreadHandle cpuTsThread = 0;
    ThreadHandle exportedAicpuTsThread = 0;
    ThreadHandle exportedCpuTsThread = 0;
    AlgResourceCtx* resCtx = nullptr;
};

HcclResult PrepareThreadResources(HcclComm comm, OpParam &param, 
    std::unique_ptr<ExecutorBase> &executor, TopoInfo* topoInfo, AlgType& algType,
    ThreadResources &threadRes)
{
    if (g_notifiesMap.find(comm) == g_notifiesMap.end()) {
        g_notifiesMap[comm].fill(0);
    }

    if (HcommIsExportThreadSupported()) {
        if (param.engine == COMM_ENGINE_AICPU_TS) {
            CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &threadRes.cpuTsThread));
            CHK_RET(HcclThreadExportToCommEngine(comm, 1, &threadRes.cpuTsThread, COMM_ENGINE_AICPU_TS,
                &threadRes.exportedAicpuTsThread));
        }
        
        CHK_RET(GetAlgRes(comm, param, executor, topoInfo, algType, &threadRes.resCtx));

        if (param.engine == COMM_ENGINE_AICPU_TS) {
            ThreadHandle mainThread = topoInfo->mainThread;
            CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS,
                &threadRes.exportedCpuTsThread));
            char* curPtr = reinterpret_cast<char *>(threadRes.resCtx) + sizeof(AlgResourceCtx) -
                sizeof(TopoInfo) - sizeof(ThreadHandle) -
                sizeof(uint32_t) * AICPU_CONTROL_NOTIFY_NUM - sizeof(void*);
            ACLCHECK(aclrtMemcpy(curPtr, sizeof(ThreadHandle), &threadRes.exportedAicpuTsThread,
                sizeof(ThreadHandle), ACL_MEMCPY_HOST_TO_DEVICE));
        }
    } else {
        CHK_RET(GetAlgRes(comm, param, executor, topoInfo, algType, &threadRes.resCtx));
        char* curPtr = reinterpret_cast<char *>(threadRes.resCtx) + sizeof(AlgResourceCtx) -
            sizeof(TopoInfo) - sizeof(ThreadHandle) -
            sizeof(uint32_t) * AICPU_CONTROL_NOTIFY_NUM - sizeof(void*);
        CHK_RET(haclrtMemcpy(curPtr, sizeof(ThreadHandle), &threadRes.exportedAicpuTsThread,
            sizeof(ThreadHandle), ACL_MEMCPY_HOST_TO_DEVICE));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchAicpuKernel(HcclComm comm, OpParam &param, TopoInfo* topoInfo,
    const std::string &algName, ThreadResources &threadRes)
{
    param.resCtx = reinterpret_cast<void*>(threadRes.resCtx);
    int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
    if (result <= 0) {
        HCCL_ERROR("failed to fill param.algName");
        return HCCL_E_INTERNAL;
    }
    std::string algTypeStr = TransferAlgTypeStr(param.algType);
    CHK_SAFETY_FUNC_RET(strcpy_s(param.algTypeStr, sizeof(param.algTypeStr), algTypeStr.c_str()));
    int32_t retComm = HcommAcquireComm(param.commName);
    CHK_PRT_RET(retComm != HCCL_SUCCESS, HCCL_ERROR("[%s] [%s] HcommAcquireComm failed ",
        __func__, param.commName), static_cast<HcclResult>(retComm));

    if (HcommIsExportThreadSupported()) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threadRes.cpuTsThread,
            threadRes.exportedCpuTsThread, topoInfo->notifyNumOnMainThread)));
    } else {
        if (aclrtRecordNotify(g_notifiesMap[comm][0], param.stream) != ACL_SUCCESS) {
            HCCL_ERROR("failed to record aicpu stream");
            return HCCL_E_INTERNAL;
        }
    }

    uint64_t beginTime = 0;
    if (HcommIsProfilingSupported()) {
        beginTime = HcommGetProfilingSysCycleTime();
    }
    std::string kernelName = "HcclLaunchAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;

    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
                HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
                            ret, kernelName.c_str()), HCCL_E_RUNTIME);

    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
                HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s",
                            ret, kernelName.c_str()), HCCL_E_RUNTIME);

    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
                HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u, kernelName:%s",
                            ret, sizeof(OpParam), kernelName.c_str()), HCCL_E_RUNTIME);

    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS,
                HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s",
                            ret, kernelName.c_str()), HCCL_E_RUNTIME);

    constexpr u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(aclRet != ACL_SUCCESS,
                HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed",
                            ret), HCCL_E_OPEN_FILE_FAILURE);

    if (HcommIsProfilingSupported()) {
        std::string profName = "scatterAicpuKernel";
        HCCL_DEBUG("[%s] profName = [%s]", __func__, profName);
        HcommProfilingReportKernel(beginTime, profName.c_str());
    }

    if (HcommIsExportThreadSupported()) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threadRes.cpuTsThread, 0,
            NOTIFY_DEFAULT_WAIT_TIME)));
    } else {
        if (aclrtWaitAndResetNotify(g_notifiesMap[comm][1], param.stream, CUSTOM_TIMEOUT) != ACL_SUCCESS) {
            HCCL_ERROR("failed to wait from aicpu stream");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(HcclComm comm, OpParam &param)
{
    TopoInfo* topoInfo = nullptr;
    CHK_RET(CalcBaseTopoInfo(comm, param, &topoInfo));

    AlgType algType;
    CHK_RET(GetAlgType(topoInfo, param.opType, algType));

    std::string algName;
    if (param.opType == HCCL_CMD_SCATTER) {
        CHK_RET(SelectAlgScatter(comm, param, topoInfo, algType, algName));
    } else if (param.opType == HCCL_CMD_REDUCE_SCATTER) {
        CHK_RET(SelectAlgReduceScatter(comm, param, topoInfo, algType, algName));
    }

    std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
    CHK_PRT_RET(executor.get() == nullptr, HCCL_ERROR("[ExecOp]Fail to find executor for algName[%s]",
        algName.c_str()), HCCL_E_PARA);

    ThreadResources threadRes;
    CHK_RET(PrepareThreadResources(comm, param, executor, topoInfo, algType, threadRes));

    if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(LaunchAicpuKernel(comm, param, topoInfo, algName, threadRes));
    } else {
        CHK_RET(executor->Orchestrate(param, threadRes.resCtx));
        param.resCtx = threadRes.resCtx;
    }
    param.algType = algType;
    return HCCL_SUCCESS;
}

HcclResult ProcessA3(HcclComm comm, OpParam& param, uint64_t beginTime)
{
    CHK_RET(ExecOp(comm, param));  //ä¿ç•™åŽŸæœ‰A3æµç¨‹
    if (HcommIsProfilingSupported()) {
        // èŽ·å–profiling opä¸ŠæŠ¥çš„ä¿¡æ¯
        HcomProInfoTmp profInfo;
        std::string algTypeStr = TransferAlgTypeStr(param.algType);
        CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()));
        CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.commName, sizeof(profInfo.commName), param.commName));
        profInfo.beginTime = beginTime;
        profInfo.dataCount = param.DataDes.count;
        profInfo.dataType = static_cast<uint8_t>(param.DataDes.dataType);
        profInfo.cmdType = static_cast<uint8_t>(param.opType);
        CHK_PRT(HcommProfilingReportOp(profInfo));

        if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
            CHK_PTR_NULL(param.resCtx);
            AlgResourceCtx* tmpCtx = reinterpret_cast<AlgResourceCtx*>(param.resCtx);
            profInfo.slaveThreadNum = tmpCtx->slaveThreadNum;
            char* curThreadPtr = reinterpret_cast<char*>(param.resCtx);
            curThreadPtr += sizeof(AlgResourceCtx);
            ThreadHandle* curThreads = reinterpret_cast<ThreadHandle *>(curThreadPtr);
            CHK_PRT(HcommProfilingUnRegThread(profInfo,curThreads));
        }
    }
    return HCCL_SUCCESS;
}

}