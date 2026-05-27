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
#include "scatter_op.h"
#include "topo.h"
#include "topo_host.h"
#include "hcomm_host_profiling_dl.h"
#include <algorithm>
#include <future>
#include <map>

namespace ops_hccl_experimental {
using namespace ops_hccl;
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

HcclAlgoType MapAlgoLevel1Config(HcclAlgoType algoConfig, AlgTypeLevel1 &algType,
    HcclCMDType opType, u32 moduleNum)
{
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", moduleNum);
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", moduleNum);
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", moduleNum);
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC:
            if (opType < HcclCMDType::HCCL_CMD_ALL) {
                HCCL_INFO("server num[%u]: level1:ahc algo is not support, set default.", moduleNum);
                return HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            }
            algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE:
            if (opType < HcclCMDType::HCCL_CMD_ALL) {
                HCCL_INFO("server num[%u]: level1:ahc broke algo is not support, set default.", moduleNum);
                return HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            }
            algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", moduleNum);
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", moduleNum);
            return algoConfig;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not supported. the config is ignored.");
            return HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        default:
            return HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    }
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

static void ValidateAndResetAlgLevel1(AlgType& algType, const std::string &opName)
{
    if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR ||
        algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB ||
        algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        return;
    }
    HCCL_INFO("[%s] algType[%s] is not supported, reset algType=ring",
        opName.c_str(), AlgTypeToStr(algType).c_str());
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
}

HcclResult SelectAlgReduceScatter(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName)
{
    (void) comm;
    ValidateAndResetAlgLevel1(algType, "Reduce_Scatter");

    if (topoInfo->userRankSize == 1) {
        return HCCL_E_INTERNAL;
    } else if (topoInfo->deviceType == DevType::DEV_TYPE_910_93 && (topoInfo->userRankSize % 2 == 0)) {
        algName = "ReduceScatterBIRSExecutor";
    }

    CHK_RET(FillAlgTagAndDebugInfo(param, topoInfo, algType, algName, "Reduce_Scatter"));
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

HcclResult PrepareAicpuKernelParams(OpParam &param, AlgResourceCtx* resCtx, const std::string &algName)
{
    param.resCtx = reinterpret_cast<void*>(resCtx);
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
    return HCCL_SUCCESS;
}

HcclResult RecordNotifyBeforeLaunch(HcclComm comm, OpParam &param, TopoInfo* topoInfo,
    ThreadResources &threadRes)
{
    if (HcommIsExportThreadSupported()) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threadRes.cpuTsThread,
            threadRes.exportedCpuTsThread, topoInfo->notifyNumOnMainThread)));
    } else {
        if (aclrtRecordNotify(g_notifiesMap[comm][0], param.stream) != ACL_SUCCESS) {
            HCCL_ERROR("failed to record aicpu stream");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult DoAicpuKernelLaunch(OpParam &param, uint64_t &beginTime)
{
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
        std::string profName = "ReduceScatterAicpuKernel";
        HCCL_DEBUG("[%s] profName = [%s]", __func__, profName);
        HcommProfilingReportKernel(beginTime, profName.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult WaitNotifyAfterLaunch(HcclComm comm, OpParam &param, ThreadResources &threadRes,
    u16 waitTime)
{
    if (HcommIsExportThreadSupported()) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threadRes.cpuTsThread, 0, waitTime)));
    } else {
        if (aclrtWaitAndResetNotify(g_notifiesMap[comm][1], param.stream, CUSTOM_TIMEOUT) != ACL_SUCCESS) {
            HCCL_ERROR("failed to wait from aicpu stream");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchAicpuKernel(HcclComm comm, OpParam &param, TopoInfo* topoInfo,
    const std::string &algName, ThreadResources &threadRes)
{
    CHK_RET(PrepareAicpuKernelParams(param, threadRes.resCtx, algName));
    CHK_RET(RecordNotifyBeforeLaunch(comm, param, topoInfo, threadRes));

    uint64_t beginTime = 0;
    CHK_RET(DoAicpuKernelLaunch(param, beginTime));

    constexpr u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
    CHK_RET(WaitNotifyAfterLaunch(comm, param, threadRes, NOTIFY_DEFAULT_WAIT_TIME));
    return HCCL_SUCCESS;
}

HcclResult ExecOpBirs(HcclComm comm, OpParam &param)
{
    TopoInfo* topoInfo = nullptr;
    CHK_RET(CalcBaseTopoInfo(comm, param, &topoInfo));

    AlgType algType;
    CHK_RET(GetAlgType(topoInfo, param.opType, algType));

    std::string algName;
    if (param.opType == HCCL_CMD_REDUCE_SCATTER) {
        CHK_RET(SelectAlgReduceScatter(comm, param, topoInfo, algType, algName));
    }

    std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
    CHK_PRT_RET(executor.get() == nullptr, HCCL_ERROR("[ExecOpBirs]Fail to find executor for algName[%s]",
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
    CHK_RET(ExecOpBirs(comm, param)); //A3-specific execution method  
    if (HcommIsProfilingSupported()) {
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