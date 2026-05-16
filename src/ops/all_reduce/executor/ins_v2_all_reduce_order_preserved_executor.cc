/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_order_preserved_executor.h"
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
#include "ins_temp_reduce_scatter_order_preserved_level2.h"
#include "ins_temp_all_gather_order_preserved_level1.h"
#include "ins_temp_all_gather_order_preserved_level2.h"
#include <cmath>
#include <algorithm>

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::InsV2AllReduceOrderPreservedExecutor()
{
    deterministicStrict_ = true;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::InitCommInfo(const OpParam &param, 
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][InitCommInfo] myRank[%u], rankSize[%u], devType[%u], "
        "reduceOp[%u], dataType[%u], dataTypeSize[%u], aicpuUnfoldMode[%d]",
        myRank_, rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_, aicpuUnfoldMode_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::CalcAlgHierarchyInfo(HcclComm comm, 
    TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos.size() > 1 ? algHierarchyInfo.infos[1].size() : 0;
    rankSizeLevel2_ = algHierarchyInfo.infos.size() > 2 ? algHierarchyInfo.infos[2].size() : 0;

    std::shared_ptr<InsAlgTemplateRSLevel1> rsLevel1TempAlg =
        std::make_shared<InsAlgTemplateRSLevel1>(param, myRank_, algHierarchyInfo.infos[0]);
    
    std::shared_ptr<InsAlgTemplateRSLevel2> rsLevel2TempAlg = nullptr;
    if (algHierarchyInfo.infos.size() > 1) {
        rsLevel2TempAlg = std::make_shared<InsAlgTemplateRSLevel2>(param, myRank_, algHierarchyInfo.infos[1]);
    }

    std::shared_ptr<InsAlgTemplateAGLevel1> agLevel1TempAlg =
        std::make_shared<InsAlgTemplateAGLevel1>(param, myRank_, algHierarchyInfo.infos[0]);

    std::shared_ptr<InsAlgTemplateAGLevel2> agLevel2TempAlg = nullptr;
    if (algHierarchyInfo.infos.size() > 1) {
        agLevel2TempAlg = std::make_shared<InsAlgTemplateAGLevel2>(param, myRank_, algHierarchyInfo.infos[1]);
    }

    AlgResourceRequest resReqRSLevel1;
    AlgResourceRequest resReqRSLevel2;
    AlgResourceRequest resReqAGLevel1;
    AlgResourceRequest resReqAGLevel2;

    CHK_RET(rsLevel1TempAlg->CalcRes(comm, param, topoInfo, resReqRSLevel1));
    if (rsLevel2TempAlg != nullptr) {
        CHK_RET(rsLevel2TempAlg->CalcRes(comm, param, topoInfo, resReqRSLevel2));
    }
    CHK_RET(agLevel1TempAlg->CalcRes(comm, param, topoInfo, resReqAGLevel1));
    if (agLevel2TempAlg != nullptr) {
        CHK_RET(agLevel2TempAlg->CalcRes(comm, param, topoInfo, resReqAGLevel2));
    }

    resourceRequest.slaveThreadNum = std::max(resReqRSLevel1.slaveThreadNum,
        resReqRSLevel2.slaveThreadNum);
    resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, resReqAGLevel1.slaveThreadNum);
    if (agLevel2TempAlg != nullptr) {
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, resReqAGLevel2.slaveThreadNum);
    }

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqRSLevel1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqRSLevel1.notifyNumPerThread[i]);
        }
        if (i < resReqRSLevel2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqRSLevel2.notifyNumPerThread[i]);
        }
        if (i < resReqAGLevel1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqAGLevel1.notifyNumPerThread[i]);
        }
        if (i < resReqAGLevel2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqAGLevel2.notifyNumPerThread[i]);
        }
    }

    resourceRequest.notifyNumOnMainThread = std::max(resReqRSLevel1.notifyNumOnMainThread,
        resReqRSLevel2.notifyNumOnMainThread);
    resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread,
        resReqAGLevel1.notifyNumOnMainThread);
    if (agLevel2TempAlg != nullptr) {
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread,
            resReqAGLevel2.notifyNumOnMainThread);
    }

    resourceRequest.channels.clear();
    resourceRequest.channels.push_back(resReqRSLevel1.channels[0]);
    if (rsLevel2TempAlg != nullptr && resReqRSLevel2.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqRSLevel2.channels[0]);
    }
    if (resReqAGLevel1.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqAGLevel1.channels[0]);
    }
    if (agLevel2TempAlg != nullptr && resReqAGLevel2.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqAGLevel2.channels[0]);
    }

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::Orchestrate(const OpParam &param,
    const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] Start");

    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    if (algHierarchyInfo_.infos.size() > 1 && algHierarchyInfo_.infos[1].size() > 0) {
        rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();
    }

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    if (algHierarchyInfo_.infos.size() > 1 && algHierarchyInfo_.infos[1].size() > 0) {
        rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    }

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    InitExecutorInfo(param);
    CalcSizePerBlock(param);
    CalcGroupSlices(param);

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::OrchestrateLoop(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d]",
        deterministicStrict_);

    CHK_RET(RunReduceScatterLevel1(param, resCtx));

    if (algHierarchyInfo_.infos.size() > 1 && algHierarchyInfo_.infos[1].size() > 0) {
        CHK_RET(RunReduceScatterLevel2(param, resCtx));
    }

    CHK_RET(RunAllGatherLevel1(param, resCtx));

    if (algHierarchyInfo_.infos.size() > 1 && algHierarchyInfo_.infos[1].size() > 0) {
        CHK_RET(RunAllGatherLevel2(param, resCtx));
    }

    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::InitExecutorInfo(const OpParam& param)
{
    deterministicStrict_ = IsNeedStrictMode(param);
    if (deterministicStrict_) {
        CHK_PRT_RET(!CheckStrictCondition(param),
            HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor] not support DETERMINISTIC_STRICT mode."),
            HCCL_E_NOT_SUPPORT);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
u64 InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::RoundUpWithDivisor(u64 value, u64 divisor) const
{
    if (value == 0 || divisor == 0) {
        return divisor;
    }
    return ((value + (divisor - 1)) / divisor) * divisor;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
u32 InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::CalReduceStreamNum(const u32& localRankSize) const
{
    return (1 << static_cast<int>(std::floor(log2(localRankSize))));
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::CalcSizePerBlock(const OpParam& param)
{
    u64 sizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
    memInfo_.sizePerBlock = RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_A5);
    memInfo_.all2allOffset = 0;
    memInfo_.scratchMemFlag = false;
    memInfo_.totalSize = 0;
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
        memInfo_.sizePerBlock, dataCount_, rankSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::CalcGroupSlices(const OpParam& param)
{
    memInfo_.groupSize.clear();
    u64 sizeRemain = dataSize_;
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
        memInfo_.groupSize.push_back(size);
        sizeRemain -= size;
    }
    memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
    HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu]", memInfo_.groupSize.size(), memInfo_.totalSize);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
bool InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::IsNeedStrictMode(const OpParam& param) const
{
    bool isStrictMode = (param.deterministicConfig == DETERMINISTIC_STRICT)
        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
            param.DataDes.dataType == HCCL_DATA_TYPE_BFP16)
        && (param.reduceType == HCCL_REDUCE_SUM)
        && rankSize_ >= MIN_STRICT_RANK_NUM_A5;
    return isStrictMode;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
bool InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::CheckStrictCondition(const OpParam& param) const
{
    CHK_PRT_RET(param.reduceType == HCCL_REDUCE_PROD,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support PROD."), false);
    CHK_PRT_RET(param.DataDes.dataType == HCCL_DATA_TYPE_FP64,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support FP64."), false);
    return true;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::RunReduceScatterLevel1(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunReduceScatterLevel1] Start");

    if (rankSizeLevel0_ == 1) {
        CHK_RET(RunReduceScatterLevel1SingleRank(param, resCtx));
        return HCCL_SUCCESS;
    }

    TemplateDataParams tempAlgParams;
    tempAlgParams.count = dataCount_;
    tempAlgParams.allRankSliceSize = memInfo_.groupSize;
    tempAlgParams.sliceOffset.clear();
    for (u32 i = 0; i < rankSize_; i++) {
        tempAlgParams.sliceOffset.push_back(i * memInfo_.sizePerBlock);
    }

    TemplateResource templateResource;
    templateResource.threads = threads_;
    templateResource.channels = remoteRankToChannelInfo_[0];

    std::shared_ptr<InsAlgTemplateRSLevel1> rsLevel1TempAlg =
        std::make_shared<InsAlgTemplateRSLevel1>(param, myRank_, algHierarchyInfo_.infos[0]);

    CHK_RET(rsLevel1TempAlg->KernelRun(param, tempAlgParams, templateResource));

    HCCL_INFO("[RunReduceScatterLevel1] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::RunReduceScatterLevel1SingleRank(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunReduceScatterLevel1SingleRank] Skip for single rank");
    memInfo_.all2allOffset = algHierarchyInfo_.infos.size() > 1 ? 1 : 0;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::RunReduceScatterLevel2(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunReduceScatterLevel2] Start");

    TemplateDataParams tempAlgParams;
    tempAlgParams.count = dataCount_ / rankSizeLevel0_;
    tempAlgParams.allRankSliceSize.clear();
    for (u32 i = 0; i < rankSizeLevel1_; i++) {
        tempAlgParams.allRankSliceSize.push_back(memInfo_.groupSize[rankIdxLevel0_ + i * rankSizeLevel0_]);
    }
    tempAlgParams.sliceOffset.clear();
    for (u32 i = 0; i < rankSizeLevel1_; i++) {
        tempAlgParams.sliceOffset.push_back((rankIdxLevel0_ + i * rankSizeLevel0_) * memInfo_.sizePerBlock);
    }

    TemplateResource templateResource;
    templateResource.threads = threads_;
    if (remoteRankToChannelInfo_.size() > 1) {
        templateResource.channels = remoteRankToChannelInfo_[1];
    }

    std::shared_ptr<InsAlgTemplateRSLevel2> rsLevel2TempAlg =
        std::make_shared<InsAlgTemplateRSLevel2>(param, myRank_, algHierarchyInfo_.infos[1]);

    CHK_RET(rsLevel2TempAlg->KernelRun(param, tempAlgParams, templateResource));

    HCCL_INFO("[RunReduceScatterLevel2] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2,
          typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::RunAllGatherLevel1(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunAllGatherLevel1] Start");

    TemplateDataParams tempAlgParams;
    tempAlgParams.count = dataCount_ / rankSize_;
    tempAlgParams.allRankSliceSize.clear();
    for (u32 i = 0; i < rankSizeLevel0_; i++) {
        tempAlgParams.allRankSliceSize.push_back(memInfo_.groupSize[rankIdxLevel1_ * rankSizeLevel0_ + i]);
    }
    tempAlgParams.sliceOffset.clear();
    for (u32 i = 0; i < rankSizeLevel0_; i++) {
        tempAlgParams.sliceOffset.push_back((rankIdxLevel1_ * rankSizeLevel0_ + i) * memInfo_.sizePerBlock);
    }

    TemplateResource templateResource;
    templateResource.threads = threads_;
    templateResource.channels = remoteRankToChannelInfo_[0];

    std::shared_ptr<InsAlgTemplateAGLevel1> agLevel1TempAlg =
        std::make_shared<InsAlgTemplateAGLevel1>(param, myRank_, algHierarchyInfo_.infos[0]);

    CHK_RET(agLevel1TempAlg->KernelRun(param, tempAlgParams, templateResource));

    HCCL_INFO("[RunAllGatherLevel1] Success");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplateRSLevel1, typename InsAlgTemplateRSLevel2,
          typename InsAlgTemplateAGLevel1, typename InsAlgTemplateAGLevel2>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRSLevel1, InsAlgTemplateRSLevel2,
    InsAlgTemplateAGLevel1, InsAlgTemplateAGLevel2>::RunAllGatherLevel2(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[RunAllGatherLevel2] Start");

    TemplateDataParams tempAlgParams;
    tempAlgParams.count = dataCount_ / rankSizeLevel0_ / rankSizeLevel1_;

    std::vector<u64> level2GroupSize;
    for (u32 rank = 0; rank < rankSizeLevel1_; rank++) {
        u64 size = 0;
        for (u32 level1RankId = 0; level1RankId < rankSizeLevel0_; level1RankId++) {
            size += memInfo_.groupSize[rank * rankSizeLevel0_ + level1RankId];
        }
        level2GroupSize.push_back(size);
    }
    tempAlgParams.allRankSliceSize = level2GroupSize;

    tempAlgParams.sliceOffset.clear();
    for (u32 rank = 0; rank < rankSizeLevel1_; rank++) {
        tempAlgParams.sliceOffset.push_back(rank * rankSizeLevel0_ * memInfo_.sizePerBlock);
    }

    TemplateResource templateResource;
    templateResource.threads = threads_;
    if (remoteRankToChannelInfo_.size() > 1) {
        templateResource.channels = remoteRankToChannelInfo_[1];
    }

    std::shared_ptr<InsAlgTemplateAGLevel2> agLevel2TempAlg =
        std::make_shared<InsAlgTemplateAGLevel2>(param, myRank_, algHierarchyInfo_.infos[1]);

    CHK_RET(agLevel2TempAlg->KernelRun(param, tempAlgParams, templateResource));

    HCCL_INFO("[RunAllGatherLevel2] Success");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduceOrderPreservedExecutor,
    InsV2AllReduceOrderPreservedExecutor, AutoMatchMeshNhr,
    InsTempReduceScatterOrderPreservedLevel1, InsTempReduceScatterOrderPreservedLevel2,
    InsTempAllGatherOrderPreservedLevel1, InsTempAllGatherOrderPreservedLevel2);
}