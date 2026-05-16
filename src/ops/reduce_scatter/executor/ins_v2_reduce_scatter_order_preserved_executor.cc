/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_order_preserved_executor.h"
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
#include <cmath>
#include <algorithm>

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceScatterOrderPreservedExecutor()
{
    deterministicStrict_ = true;
} 

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u]",
        myRank_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, myRank_, algHierarchyInfo.infos[0]);
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));

    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][Orchestrate] Start");

    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    threads_ = resCtx.threads;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    maxTmpMemSize_ = resCtx.cclMem.size;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceScatterOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d]",
        deterministicStrict_);

    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    tempAlgParams.count = dataCount_;
    tempAlgParams.allRankSliceSize = memInfo_.groupSize;
    tempAlgParams.sliceOffset.clear();
    for (u32 i = 0; i < rankSize_; i++) {
        tempAlgParams.sliceOffset.push_back(i * memInfo_.sizePerBlock);
    }
    tempAlgParams.sliceSize = memInfo_.groupSize[myRank_ % rankSize_];
    tempAlgParams.inputSliceStride = dataSize_;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.repeatNum = 1;
    tempAlgParams.inputRepeatStride = 0;
    tempAlgParams.outputRepeatStride = 0;
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;

    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);

    CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));

    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::InitExecutorInfo(const OpParam &param)
{
    deterministicStrict_ = IsNeedStrictMode(param);
    if (deterministicStrict_) {
        CHK_PRT_RET(!CheckStrictCondition(param),
            HCCL_ERROR("[InsV2ReduceScatterOrderPreservedExecutor] not support DETERMINISTIC_STRICT mode."),
            HCCL_E_NOT_SUPPORT);
    }
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
u64 InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::RoundUpWithDivisor(
    u64 value, u64 divisor) const
{
    if (value == 0 || divisor == 0) {
        return divisor;
    }
    return ((value + (divisor - 1)) / divisor) * divisor;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcSizePerBlock(const OpParam &param)
{
    u64 sizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
    memInfo_.sizePerBlock = RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
    memInfo_.all2allOffset = 0;
    memInfo_.scratchMemFlag = false;
    memInfo_.totalSize = 0;
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
        memInfo_.sizePerBlock, dataCount_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcGroupSlices(const OpParam &param)
{
    memInfo_.groupSize.clear();
    u64 sizeRemain = dataSize_;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
        memInfo_.groupSize.push_back(size);
        sizeRemain -= size;
    }
    memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
    HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu]",
        memInfo_.groupSize.size(), memInfo_.totalSize);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::IsNeedStrictMode(const OpParam &param) const
{
    bool isStrictMode = (param.deterministicConfig == DETERMINISTIC_STRICT)
        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
            param.DataDes.dataType == HCCL_DATA_TYPE_BFP16)
        && (param.reduceType == HCCL_REDUCE_SUM)
        && rankSize_ >= MIN_STRICT_RANK_NUM_ORDER_PRESERVED;
    return isStrictMode;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CheckStrictCondition(const OpParam &param) const
{
    CHK_PRT_RET(param.reduceType == HCCL_REDUCE_PROD,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support PROD."), false);
    CHK_PRT_RET(param.DataDes.dataType == HCCL_DATA_TYPE_FP64,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support FP64."), false);
    return true;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatterOrderPreserved,
    InsV2ReduceScatterOrderPreservedExecutor, TopoMatch1D, InsTempReduceScatterOrderPreservedLevel1);

}