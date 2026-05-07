/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "reduce_scatter_birs_executor.h"

namespace ops_hccl {

ReduceScatterBIRSExecutor::ReduceScatterBIRSExecutor() : ReduceScatterExecutorBase()
{
    // 只有ring算法使能了DMA消减
    DMAReduceFlag_ = true; // 单算子模式下使能，当前executor只支持单算子模式
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
        AlgTypeLevel1::ALG_LEVEL1_NB,
        AlgTypeLevel1::ALG_LEVEL1_RING
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
        AlgTypeLevel2::ALG_LEVEL2_NB,
        AlgTypeLevel2::ALG_LEVEL2_RING
    };
}

HcclResult ReduceScatterBIRSExecutor::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    if (topoInfo->serverNum == 1) {
        CHK_RET(CalcGeneralTopoInfoForA3(comm, topoInfo, algHierarchyInfo));
    } else {
        CHK_RET(CalcGeneralTopoInfoForA3BirsInterServer(comm, topoInfo, algHierarchyInfo));
    }
    CHK_RET(RefreshAlgType(algType));
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    
    u32 threadNum = 3;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // level0 channel
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    
    HCCL_INFO("[ReduceScatterBIRSExecutor][CalcResRequest]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level0Channels[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread.size(), resourceRequest.notifyNumOnMainThread,
        level0Channels.size());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSExecutor::Log2HwPermutation(HcclComm comm,
    AlgResourceRequest& resourceRequest, AlgResourceCtx* ctx)
{
uint32_t RealRanks[ctx->topoInfo.userRankSize];
    HcclGetRankLists(comm, RealRanks);
    char* curPtr = reinterpret_cast<char *>(ctx);
    curPtr += sizeof(AlgResourceCtx);
    curPtr += sizeof(ThreadHandle) * (resourceRequest.slaveThreadNum + 1);
    ChannelInfo* channels = reinterpret_cast<ChannelInfo*>(curPtr);
    std::vector<ChannelInfo> tmpChannelList(ctx->topoInfo.userRankSize);
    for (u32 i = 0; i < ctx->topoInfo.userRankSize; i++)
        tmpChannelList[i] = channels[i];
    for (u32 i = 0; i < ctx->topoInfo.userRankSize; i++)
        channels[i] = tmpChannelList[RealRanks[i]];
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ReduceScatterBIRSExecutor][KernelRun] starts.");

    CHK_RET(KernelRunLevel0(param, execMem));

    HCCL_INFO("reduce_scatter birs run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSExecutor::KernelRunLevel0(const OpParam &param, ExecMem &execMem)
{
    SubCommInfo level0CommInfo;
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo));
    u32 level0LocalRank = level0CommInfo.localRank;
    for (u32 rnk = 0; rnk < topoInfo_->userRankSize; rnk++)
    {
        if (channels_[COMM_LEVEL0][rnk].isValid == 0)
            level0LocalRank = rnk;
        HCCL_INFO("local rank %u isValid %u remoteRank %u", level0CommInfo.localRank, channels_[COMM_LEVEL0][rnk].isValid, channels_[COMM_LEVEL0][rnk].remoteRank);
    }
    u32 level0LocalRankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0LocalRank;
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, unitSize_, sliceNum, dataSegsSlice));

    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    if (topoInfo_->serverNum == 1) {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCE_SCATTER_BIRS);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_SCATTER_BIRS in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(level0LocalRank, level0LocalRankSize));
    } else {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCE_SCATTER_BIRS_INTER);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_SCATTER_BIRS_INTER in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(topoInfo_->serverNum, topoInfo_->userRankSize));
    }
   
    HcclMem UsrInputMem{HCCL_MEM_TYPE_DEVICE, execMem.inputPtr, execMem.count * unitSize_};
    HcclMem UsrOutputMem{HCCL_MEM_TYPE_DEVICE, execMem.outputPtr, execMem.count * unitSize_};
    
    CHK_RET(level0TempAlg->Prepare(UsrInputMem, UsrOutputMem, execMem.inputMem, execMem.count,
    param.DataDes.dataType, thread_, slaveThreads_, param.reduceType, 0, dataSegsSlice));

    CHK_RET(level0TempAlg->RunAsync(level0LocalRank, level0LocalRankSize, channels_[COMM_LEVEL0]));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterBIRSExecutor", ReduceScatterBIRS, ReduceScatterBIRSExecutor);
}
