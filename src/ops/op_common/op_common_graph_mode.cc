/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "op_common_graph_mode.h"

#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>  // 包含getenv函数
#include <cstring>  // 包含strcmp函数
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "mmpa_api.h"
#include "param_check.h"
#include "executor_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo_host.h"
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"
#include "alg_param.h"
#include "alg_type.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "dpu/kernel_launch.h"

namespace ops_hccl {
thread_local aclrtNotify g_notifies_host_with_device_graph_mode[AICPU_CONTROL_NOTIFY_NUM];
thread_local std::map<std::string, HcclMemHandle> g_memHandleCache_graph_mode; // 当前AIV存放注册内存的memHandle使用

HcclResult HcclExecOpGraphMode(HcclComm comm, OpParam &param)
{
    HCCL_INFO("Start to execute ExecOp.");
    param.hcclComm = comm;
    // 因为AICPU是保底的所以这里获取到是AICPU引擎就应该加载Kernel
    if (GetExternalInputHcclAicpuUnfold() == true) {
        HCCL_DEBUG("[HcclExecOp] is aicpu mode");
        param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    }
    else if (GetExternalInputHcclAivMode() == true) {
        HCCL_DEBUG("[HcclExecOp] is aiv mode");
        // 注册AIV kernel二进制
        CHK_RET(RegisterKernel(param.opType, g_aivKernelInfoMap[param.opType].first, g_aivKernelInfoMap[param.opType].second));
        param.opExecuteConfig = OpExecuteConfig::AIV;
        param.engine = CommEngine::COMM_ENGINE_AIV;
    }
    else if (GetExternalInputHcclCcuMSMode()) {
        param.opExecuteConfig = OpExecuteConfig::CCU_MS;
        param.engine = CommEngine::COMM_ENGINE_CCU;
    }
    else if (GetExternalInputHcclCcuSchedMode()) {
        param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
        param.engine = CommEngine::COMM_ENGINE_CCU;
    }
    // 注册图模式InputMem/OutputMem
    CHK_RET(HcclRegstryBuffGraphMode(comm, "InputBUffer", param.inputPtr, param.inputSize));
    CHK_RET(HcclRegstryBuffGraphMode(comm, "OutputBUffer", param.outputPtr, param.outputSize));
    // 获取基础拓扑
    TopoInfo *topoInfo = nullptr;
    CHK_RET(HcclCalcTopoInfo(comm, param, &topoInfo));

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    std::string algName = "";
    std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
    OpExecuteConfig opExecuteConfig;
    CHK_RET(collAlgSelector->Run(param, topoInfo, algName, opExecuteConfig));
    if (algName == "") {
        HCCL_ERROR("[HcclExecOp] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param, opExecuteConfig));
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aipcu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[HcclExecOp] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
    }
    SetOpParamAlgTag(param, algName);
    // 在原先的commName中添加执行模式，得到commModeTag
    bool isOpBase = false;
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_%s", param.commName, opModeStr);
    if (ret <= 0) {
        HCCL_ERROR("[%s] failed to fill param.commModeTag", __func__);
        return HCCL_E_INTERNAL;
    }

    std::shared_ptr<InsCollAlgBase> executor =
        CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 资源结构体
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    // 资源序列化结果
    void *resCtxSequence;
    bool isResourceReused = false;
    CHK_RET(HcclGetAlgResGraphMode(comm, param, executor, topoInfo, resCtxHost, &resCtxSequence, isResourceReused));

    // 算法执行
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
        param.resCtx = resCtxSequence;
        // 将算法名字放在param参数中
        int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
        if (result <= 0) {
            HCCL_ERROR("faled to fill param.algName");
            return HCCL_E_INTERNAL;
        }

        if (param.engine == COMM_ENGINE_CPU) {
            // 注册dpu回调函数
            CHK_RET(static_cast<HcclResult>(HcclTaskRegister(comm, param.algTag, HcclLaunchDPUKernel)));
        }

        // Host stream通知Device主thread，这里现在是直接用的acl的接口，是否有基础库的接口
        if (aclrtRecordNotify(g_notifies_host_with_device_graph_mode[0], param.stream) != ACL_SUCCESS) {
            HCCL_ERROR("failed to record aicpu stream");
            return HCCL_E_INTERNAL;
        }
        // 执行device测的算法编排
        std::string kernelName = "HcclLaunchAicpuKernel";
        aclrtFuncHandle funcHandle;
        aclrtArgsHandle argsHandle;
        // 注意，目前开源HCCL加载AICPU kernel使用的是从json文件加载
        // 详见load_kernel.cc中的LoadAICPUKernel函数，且只实现了scatter的，先共用scatter的
        aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName:%s",
                ret,
                kernelName.c_str()),
            HCCL_E_RUNTIME);
        ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR(
                "[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName:%s", ret, kernelName.c_str()),
            HCCL_E_RUNTIME);
        aclrtParamHandle paraHandle;
        size_t paramSize = sizeof(OpParam) + param.varMemSize;
        ret = aclrtKernelArgsAppend(argsHandle, &param, paramSize, &paraHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u,"
                       "kernelName:%s",
                ret,
                paramSize,
                kernelName.c_str()),
            HCCL_E_RUNTIME);
        ret = aclrtKernelArgsFinalize(argsHandle);
        CHK_PRT_RET(ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName:%s",
                ret,
                kernelName.c_str()),
            HCCL_E_RUNTIME);
        // notifywait默认1836等待时长
        u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;
        aclrtLaunchKernelCfg cfg;
        aclrtLaunchKernelAttr attr;
        attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
        attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
        cfg.numAttrs = 1;
        cfg.attrs = &attr;
        constexpr u32 numBlocks = 1;
        aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
            HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret),
            HCCL_E_OPEN_FILE_FAILURE);
        // Host stream等待Device的通知
        if (aclrtWaitAndResetNotify(g_notifies_host_with_device_graph_mode[1], param.stream, CUSTOM_TIMEOUT) != ACL_SUCCESS) {
            HCCL_ERROR("failed to wait from aicpu stream");
            return HCCL_E_INTERNAL;
        }
        if (aclrtSynchronizeStream(param.stream) != 0) {
            HCCL_ERROR("Stream Synchronize Failed");
            return HCCL_E_INTERNAL;
        }
    } else if (param.engine == COMM_ENGINE_AIV) {
        param.resCtx = resCtxSequence;
        AlgResourceCtxSerializable &resCtxHost = *static_cast<AlgResourceCtxSerializable *>(resCtxSequence);
        HCCL_INFO("[%s] algTag[%s] commModeTag[%s] resCtx(Host)[%p] aivCommInfoPtr(Device)[%p]", __func__,
            param.algTag, param.commModeTag, param.resCtx, resCtxHost.aivCommInfoPtr);
        CHK_RET(GetAivCountTag(param.commModeTag, topoInfo->userRank, param.aivCountTag)); // commTag需要拼接单算子或者图模式
        u32 numBlocksLimit = MAX_NUM_BLOCKS;
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
        CHK_PRT_RET(numBlocksLimit < 1,
            HCCL_ERROR("[%s] block num less than 1, block num[%d]", __func__, numBlocksLimit), HCCL_E_PARA);
        param.numBlocksLimit = numBlocksLimit;
        HCCL_INFO("[%s] Aiv core limit is [%d].", __func__, numBlocksLimit);
        bool isAivClearEnable = false; // 图模式首算子，暂不支持
        if (isAivClearEnable || param.aivCountTag == 1) {
            CHK_RET(ClearAivSyncBuf(param, resCtxHost));
        }
        CHK_RET(executor->Orchestrate(param, resCtxHost));
    } else {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char *ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    }
    HCCL_INFO("Execute ExecOp success.");
    return HCCL_SUCCESS;
}

HcclResult HcclRegstryBuffGraphMode(HcclComm comm, const char *memTag, void *bufferPtr, u64 buffSize)
{
    CommMem regMem{COMM_MEM_TYPE_DEVICE, bufferPtr, bufferSize};
    void *memHandle = nullptr;
    CHK_RET(HcclCommMemReg(comm, memTag, &regMem, &memHandle));
    return HCCL_SUCCESS;
}
HcclResult HcclGetRemoteBuffGraphMode(HcclComm comm, ChannelHandle channel, const char *memTag, void **bufferPtr, u64 *buffSize)
{
    CHK_PTR_NULL(bufferPtr);
    CHK_PTR_NULL(buffSize);

    u32 memNum;
    CommMem *remoteMemList;
    char **memTags;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMemList, &memTags));
    for (u32 i=0; i< memNum; i++) {
        HCCL_INFO("[%s] memNum[%u/%u] memTags[%s] size[%llu]", __func__, i, memNum, memTags[i], *size);
        if (strcmp(memTags[i], memTag) == 0) {
            *buffer = remoteMemList[i].addr;
            *size = remoteMemList[i].size;
            HCCL_INFO("[%s] Found %u memNum[%u/%u] is %u at index %u: addr=%p, size=%llu", __func__, *memTag, 
                memNum, i, remoteMemList[i].addr, remoteMemList[i].size);
            break;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult HcclGetAlgResGraphMode(HcclComm comm, OpParam& param, std::shared_ptr<InsCollAlgBase>& executor, TopoInfo* topoInfo,
                         std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void** resCtxSequence, bool &isResourceReused)
{
    HCCL_INFO("Start to execute HcclGetAlgRes.");
    bool increCreateChannelFlag = false;
    if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        // 增量建链模式
        HCCL_ERROR("[HcclGetAlgResGraphMode] grpah mode do not support HCCL_CMD_BATCH_SEND_RECV");
        return HCCL_E_PARA;
    }
    uint64_t size = 0;
    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo;  // 分级通信域信息{localRankId, localRankSize}
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
    // 资源计算
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));

    // host侧资源
    if (param.engine == COMM_ENGINE_RESERVED) {

    } else if (param.engine == COMM_ENGINE_CPU) {
        CHK_RET(GetAlgResDPUGraphMode(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
            size, increCreateChannelFlag));
    } else if (param.engine == COMM_ENGINE_CPU_TS) {

    } else if (param.engine == COMM_ENGINE_AICPU) {

    } else if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(GetAlgResAICPUGraphMode(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                               size, increCreateChannelFlag));
    } else if (param.engine == COMM_ENGINE_AIV) {
        CHK_RET(GetAlgResAivGraphMode(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence, size));
    } else if (param.engine == COMM_ENGINE_CCU) {
        CHK_RET(GetAlgResCcuGraphMode(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size));
    } else {
        HCCL_ERROR("fail to get engine.", HCCL_E_PARA);
    }
    param.ctxSize = size;
    return HCCL_SUCCESS;
}

HcclResult GetAlgResAICPUGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag)
{
    std::string tagStr = param.algTag;
    // 图模式不复用Ctx
    resCtxHost->commInfoPtr = static_cast<void *>(comm);
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;
    // 创建资源，并填充到Host内存上
    HcclResult ret = HcclAllocAlgResourceAICPUGraphMode(comm, param, resRequest, resCtxHost);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to alloc alg resource."), ret);
    // 在device侧创建Ctx，并将host资源拷贝到device侧
    ret = HcclMemcpyCtxHostToDevice(comm, param, resCtxHost, resCtxSequence, ctxSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to memcpy hostCtx to device."), ret);
    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}


HcclResult HcclAllocAlgResourceAICPUGraphMode(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    CHK_RET(HcclGetH2DNotify(resCtxHost));
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclGetChannelGraphMode(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    resCtxHost->channels.resize(resRequest.channels.size());
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        // 获取子通信域的建链数量
        u32 channelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(channelNum);

        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, levelNChannelRequest.data(),
                channelNum, levelNChannels.data()));
        }

        for (u32 idx = 0; idx < channelNum; idx++) {
            ChannelInfo channel;
            // 对于真实建链的链路进行填充
            HcclChannelDesc &channelDescNew = levelNChannelRequest[idx];
            channel.isValid = true;
            channel.remoteRank = channelDescNew.remoteRank;
            channel.protocol = channelDescNew.channelProtocol;
            channel.locationType = channelDescNew.remoteEndpoint.loc.locType;
            channel.notifyNum = channelDescNew.notifyNum;
            channel.handle = levelNChannels[idx];

            void* remoteCclBufferAddr;
            uint64_t remoteCclBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteCclBufferAddr, &remoteCclBufferSize));
            channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};

            void* remoteInputBufferAddr;
            uint64_t remoteInputBufferSize;
            CHK_RET(HcclGetRemoteBuffGraphMode(comm, levelNChannels[idx], &remoteInputBufferAddr, &remoteInputBufferSize, "InputBuffer"));
            channel.remoteInputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteInputBufferAddr, remoteInputBufferSize};

            void* remoteOutputBufferAddr;
            uint64_t remoteOutputBufferSize;
            CHK_RET(HcclGetRemoteBuffGraphMode(comm, levelNChannels[idx], &remoteOutputBufferAddr, &remoteOutputBufferSize, "OutputBuffer"));
            channel.remoteOutputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteOutputBufferAddr, remoteOutputBufferSize};

            resCtxHost->channels[level].push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgResCcuGraphMode(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo* topoInfo,
                        AlgHierarchyInfoForAllLevel& algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize)
{
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    // 创建资源，并填充到Host内存上
    HcclResult ret = HcclAllocAlgResourceCcuGraphMode(comm, param, resRequest, resCtxHost);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        return ret;
    }
    // todo : check resreq合法
    // 序列化
    std::vector<char> seq = resCtxHost->Serialize();
    uint64_t size = seq.size();

    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    memcpy_s(ctx, size, seq.data(), size);
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Execute GetAlgResCCU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceCcuGraphMode(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                                   std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    CHK_RET(HcclGetH2DNotify(resCtxHost));
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclGetChannelForCcu(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclGetCcuKernel(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelForCcuGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    // 以kernel为粒度申请channel
    for (CcuKernelInfo& kernelInfo: resRequest.ccuKernelInfos) {
        std::vector<HcclChannelDesc> &kernelChannelRequest = kernelInfo.channels;
        
        u32 channelNum = kernelChannelRequest.size();
        std::vector<ChannelHandle> kernelChannels;
        kernelChannels.resize(channelNum);
        
        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, kernelChannelRequest.data(),
                channelNum, kernelChannels.data()));
        }
        kernelInfo.kernelArg->channels = kernelChannels;
        HCCL_INFO("[HcclGetChannelForCcu] Get [%lu] channels", channelNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCcuKernelGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    
    u32 totalKernelNum = 0;
    for (auto t: resRequest.ccuKernelNum) {
        totalKernelNum += t;
    }
    CHK_PRT_RET(totalKernelNum != resRequest.ccuKernelInfos.size(),
        HCCL_ERROR("[HcclGetCcuKernel]ccuKernel num not match!"),
        HCCL_E_INTERNAL);

    for (CcuKernelInfo& kernelInfo: resRequest.ccuKernelInfos) {

        void* kernelArgPtr = static_cast<void*>(kernelInfo.kernelArg.get()); // 保证没有释放
        void* creatorPtr = static_cast<void*>(&kernelInfo.creator);
        
        HCCL_DEBUG("[AllocAlgResource] kernelArgPtr[%p], creator[%p]", kernelArgPtr, &(kernelInfo.creator));
        CcuKernelHandle handle;
        CHK_RET(HcclCcuKernelRegister(comm, &handle, creatorPtr, kernelArgPtr));
        resCtxHost->ccuKernels.push_back(handle);
    }
    CHK_RET(HcclCcuKernelRegisterFinish(comm));
    resCtxHost->ccuKernelNum = resRequest.ccuKernelNum;
    return HCCL_SUCCESS;
}

HcclResult GetAlgResAivGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize)
{
    uint64_t size = sizeof(AlgResourceCtxSerializable);
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, resCtxSequence));

    AlgResourceCtxSerializable* resCtxHost = static_cast<AlgResourceCtxSerializable *>(*resCtxSequence);
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    CHK_RET(HcclAllocAlgResourceAiv(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAivGraphMode(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, AlgResourceCtxSerializable* resCtxHost)
{
    HCCL_INFO("[%s]Start to execute.", __func__);
    HcclMemHandle memHandle; // 注册到通信域内存的handle，用于建链
    // 获取存放AIV对端信息和标记区的空间
    uint64_t commInfoSize = 0;
    if (HcclEngineCtxGet(comm, param.commModeTag, param.engine, &(resCtxHost->aivCommInfoPtr), &commInfoSize) != HCCL_SUCCESS) {
        CHK_RET(HcclEngineCtxCreate(comm, param.commModeTag, param.engine, AIV_TAG_BUFF_LEN, &(resCtxHost->aivCommInfoPtr)));
        // 清零
        ACLCHECK(aclrtMemset(resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        // 注册到通信域，支持建链时交换
        CommMem regMem{COMM_MEM_TYPE_DEVICE, resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        CHK_RET(HcclCommMemReg(comm, param.commModeTag, &regMem, &memHandle));
        g_memHandleCache_graph_mode[param.commModeTag] = memHandle;
    } else {
        if (g_memHandleCache_graph_mode.find(param.commModeTag) == g_memHandleCache_graph_mode.end()) {
            HCCL_ERROR("[%s]aiv memHandle not found in map", __func__);
            return HCCL_E_INTERNAL;
        }
        memHandle = g_memHandleCache_graph_mode[param.commModeTag];
    }
    HCCL_INFO("[%s]commModeTag[%s] regMemAddr[%p] memHandle[%p]", __func__, param.commModeTag, resCtxHost->aivCommInfoPtr,
        memHandle);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[%s]local cclBufferAddr[%p] cclBufferSize[%llu]", __func__, cclBufferAddr, cclBufferSize);
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[resCtxHost->topoInfo.userRank] = cclBufferAddr;
    buffersOut[resCtxHost->topoInfo.userRank] = resCtxHost->aivCommInfoPtr;

    // 迭代每个子通信域的建链请求，创建链路
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        for (auto &channelDesc : levelNChannelRequest) {
            channelDesc.memHandles = &memHandle;
            channelDesc.memHandleNum = 1;
        }
        // 获取子通信域的建链数量
        u32 validChannelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(validChannelNum);
        HCCL_INFO("[%s]level[%u] validChannelNum[%u]", __func__, level, validChannelNum);

        if (validChannelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, levelNChannelRequest.data(),
                validChannelNum, levelNChannels.data()));
        }

        for (u32 idx = 0; idx < validChannelNum; idx++) {
            HcclChannelDesc &channelDesc = levelNChannelRequest[idx];
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            HCCL_INFO("[%s]remoteRank[%u] cclBufferAddr[%p] cclBufferSize[%llu]", __func__, channelDesc.remoteRank,
                remoteBufferAddr, remoteBufferSize);
            buffersIn[channelDesc.remoteRank] = remoteBufferAddr;

            u32 memNum;
            CommMem* remoteMems;
            char** memTags;
            CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
            CHK_PRT_RET(memNum != 1,
                HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum[%u] not equal to 1", __func__, memNum), HCCL_E_PARA);
            HCCL_INFO("[%s]remoteRank[%u] memNum[%u] regMemAddr[%p] regMemSize[%llu] memTag[%s]", __func__,
                channelDesc.remoteRank, memNum, remoteMems[0].addr, remoteMems[0].size, memTags[0]);
            buffersOut[channelDesc.remoteRank] = remoteMems[0].addr;
        }
    }

    ACLCHECK(aclrtMemcpy(resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, MAX_RANK_SIZE * sizeof(void*),
        ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
        buffersOut, MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GetAlgResDPUGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfo *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag)
{
    // 申请共享内存
    uint64_t shmemSize = 100 * 1024 * 1024;
    void *shmemPtr = nullptr;
    bool newCreated;
    CHK_RET(HcclDevMemAcquire(comm, "DPUTAG", &shmemSize, &shmemPtr, &newCreated));
    resCtxHost->npu2DpuShmemPtr = shmemPtr;
    resCtxHost->dpu2NpuShmemPtr = static_cast<void*>(static_cast<uint8_t*>(shmemPtr) + shmemSize / 2);

    CHK_RET(GetAlgResAICPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                           ctxSize, increCreateChannelFlag));

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl