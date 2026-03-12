/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
#include "op_common_graph_mode.h"
#include "op_common.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "dpu/kernel_launch.h"

namespace ops_hccl {

HcclResult HcclExecOpGraphMode(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, std::string &algName, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute HcclExecOpGraphMode.");
    param.hcclComm = comm;
    // 在原先的commName中添加执行模式，得到commModeTag
    bool isOpBase = true;
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_%s", param.commName, opModeStr);
    if (ret <= 0) {
        HCCL_ERROR("[%s] failed to fill param.commModeTag", __func__);
        return HCCL_E_INTERNAL;
    }

    std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 资源结构体
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    // 资源序列化结果
    void *resCtxSequence;
    bool isResourceReused = false;

    ThreadHandle cpuTsThread;
    ThreadHandle exportedAicpuTsThread;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        // Export cpuTsThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    }

    CHK_RET(HcclGetAlgResGraphMode(comm, param, executor, topoInfo.get(), resCtxHost, &resCtxSequence, isResourceReused));

    ThreadHandle exportedCpuTsThread;
    ThreadHandle mainThread;
    u32 notifyNumOnMainThread;
    if (param.engine == COMM_ENGINE_AICPU_TS) {
        // 获取主流信息
        CHK_RET(GetMainThreadInfo(comm, param, mainThread, notifyNumOnMainThread));
        // Export mainThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
        // cpuTsThread 添加到param里
        param.opThread = exportedAicpuTsThread;
    }
    if (!resCtxHost.algHierarchyInfo.infos.empty()) {
        HCCL_INFO("[HcclExecOpGraphMode]resCtxHost.algHierarchyInfo.infos.size: %d", resCtxHost.algHierarchyInfo.infos.size());
        if (!resCtxHost.algHierarchyInfo.infos[0].empty()) {
            HCCL_INFO("[HcclExecOpGraphMode]resCtxHost.algHierarchyInfo.infos[0].size: %d", resCtxHost.algHierarchyInfo.infos[0].size());
            if (!resCtxHost.algHierarchyInfo.infos[0][0].empty()) {
                HCCL_INFO("[HcclExecOpGraphMode]resCtxHost.algHierarchyInfo.infos[0][0].size: %d", resCtxHost.algHierarchyInfo.infos[0][0].size());
            } else {
                HCCL_ERROR("HcclExecOpGraphMode] resCtxHost.algHierarchyInfo.infos[0][0] is empty.");
                return HCCL_E_INTERNAL;
            }
        }  else {
                HCCL_ERROR("HcclExecOpGraphMode] resCtxHost.algHierarchyInfo.infos[0] is empty.");
                return HCCL_E_INTERNAL;
        }
    } else {
        HCCL_ERROR("HcclExecOpGraphMode] resCtxHost.algHierarchyInfo.infos is empty.");
        return HCCL_E_INTERNAL;
    }
    // 算法执行
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclAicpuKernelEntranceLaunch(comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread,
            resCtxSequence, algName));
    } else if (param.engine == COMM_ENGINE_AIV) {
        param.resCtx = resCtxSequence;
        AlgResourceCtxSerializable &resCtxHost = *static_cast<AlgResourceCtxSerializable *>(resCtxSequence);
        CHK_RET(HcclAivKernelEntranceLaunch(comm, param, topoInfo, resCtxHost));
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
    HCCL_INFO("Execute HcclExecOp success.");
    return HCCL_SUCCESS;
}

HcclResult HcclRegstryBuffGraphMode(HcclComm comm, const char *memTag, void *bufferPtr, uint64_t bufferSize)
{
    CommMem regMem{COMM_MEM_TYPE_DEVICE, bufferPtr, bufferSize};
    void *memHandle = nullptr;
    CHK_RET(HcclCommMemReg(comm, memTag, &regMem, &memHandle));
    return HCCL_SUCCESS;
}

HcclResult HcclGetRemoteBuffGraphMode(HcclComm comm, ChannelHandle channel, const char *memTag, void **bufferPtr, uint64_t *bufferSize)
{
    CHK_PTR_NULL(bufferPtr);
    CHK_PTR_NULL(bufferSize);

    u32 memNum;
    CommMem *remoteMemList;
    char **memTags;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMemList, &memTags));
    for (u32 i=0; i< memNum; i++) {
        HCCL_INFO("[%s] memNum[%u/%u] memTags[%s]", __func__, i, memNum, memTags[i]);
        if (strcmp(memTags[i], memTag) == 0) {
            *bufferPtr = remoteMemList[i].addr;
            *bufferSize = remoteMemList[i].size;
            HCCL_INFO("[%s] Found %u memNum[%u/%u] is %u at index %u: addr=%p, size=%llu", __func__, *memTag, 
                memNum, i, remoteMemList[i].addr, remoteMemList[i].size);
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetAlgResGraphMode(HcclComm comm, OpParam& param, std::shared_ptr<InsCollAlgBase>& executor, TopoInfoWithNetLayerDetails* topoInfo,
                         std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void** resCtxSequence, bool &isResourceReused)
{
    HCCL_INFO("Start to execute HcclGetAlgResGraphMode.");

    bool increCreateChannelFlag = false;
    uint64_t size = 0;
    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo;  // 分级通信域信息{localRankId, localRankSize}
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO("[HcclGetAlgResGraphMode]algHierarchyInfo.infos.size: %d", algHierarchyInfo.infos.size());
    HCCL_INFO("[HcclGetAlgResGraphMode]algHierarchyInfo.infos[0].size: %d", algHierarchyInfo.infos[0].size());
    HCCL_INFO("[HcclGetAlgResGraphMode]algHierarchyInfo.infos[0][0].size: %d", algHierarchyInfo.infos[0][0].size());
    // 资源计算
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));

    // host侧资源
    if (param.engine == COMM_ENGINE_RESERVED) {

    } else if (param.engine == COMM_ENGINE_CPU) {
        CHK_RET(GetAlgResDPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
            size, increCreateChannelFlag));
    } else if (param.engine == COMM_ENGINE_CPU_TS) {

    } else if (param.engine == COMM_ENGINE_AICPU) {

    } else if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(GetAlgResAICPUGraphMode(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                               size, increCreateChannelFlag));
    } else if (param.engine == COMM_ENGINE_AIV) {
        CHK_RET(GetAlgResAiv(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence));
    } else if (param.engine == COMM_ENGINE_CCU) {
        CHK_RET(GetAlgResCcu(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size));
    } else {
        HCCL_ERROR("fail to get engine.", HCCL_E_PARA);
    }
    param.ctxSize = size;
    return HCCL_SUCCESS;
}

HcclResult GetAlgResAICPUGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag)
{
    std::string tagStr = param.algTag;
    // 直接创建host侧Ctx
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
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclGetChannelGraphMode(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}


HcclResult HcclGetChannelGraphMode(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    char inputBuffTag[MAX_MEM_TAG_LENGTH];
    char outputBuffTag[MAX_MEM_TAG_LENGTH];
    auto retIn = sprintf_s(inputBuffTag, sizeof(inputBuffTag), "%s_%s", param.algTag, "InputBUffer");
    auto retOut =  sprintf_s(outputBuffTag, sizeof(outputBuffTag), "%s_%s", param.algTag, "OutputBUffer");
    if (retIn <= 0 || retOut <= 0){
        HCCL_ERROR("[HcclGetChannelGraphMode]faled to fill BuffTag");
        return HcclResult::HCCL_E_INTERNAL;
    }
    CHK_RET(HcclRegstryBuffGraphMode(comm, inputBuffTag, param.inputPtr, param.inputSize));
    CHK_RET(HcclRegstryBuffGraphMode(comm, outputBuffTag, param.outputPtr, param.outputSize));
 
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
            CHK_RET(HcclGetRemoteBuffGraphMode(comm, levelNChannels[idx], inputBuffTag, &remoteInputBufferAddr, &remoteInputBufferSize));
            channel.remoteInputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteInputBufferAddr, remoteInputBufferSize};

            void* remoteOutputBufferAddr;
            uint64_t remoteOutputBufferSize;
            CHK_RET(HcclGetRemoteBuffGraphMode(comm, levelNChannels[idx], outputBuffTag, &remoteOutputBufferAddr, &remoteOutputBufferSize));
            channel.remoteOutputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteOutputBufferAddr, remoteOutputBufferSize};

            resCtxHost->channels[level].push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl