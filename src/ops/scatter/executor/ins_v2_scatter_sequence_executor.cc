/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_scatter_sequence_executor.h"
#include "ins_temp_scatter_mesh_1D.h"
#include "ins_temp_scatter_nhr_dpu.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2ScatterSequenceExecutor() {}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2ScatterSequenceExecutor][InitCommInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    rankSizeLevel0_ = algHierarchyInfo.infos[0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1].size();

    std::shared_ptr<InsAlgTemplate0> intraScatterTempAlg =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> interScatterTempAlg =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqScatterIntra;
    AlgResourceRequest resReqScatterInter;

    CHK_RET(intraScatterTempAlg->CalcRes(comm, param, topoInfo, resReqScatterIntra));
    CHK_RET(interScatterTempAlg->CalcRes(comm, param, topoInfo, resReqScatterInter));

    // step1在完成后，完成后同步后展开step2，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max({resReqScatterIntra.slaveThreadNum, resReqScatterInter.slaveThreadNum});
    resourceRequest.notifyNumPerThread = std::max({resReqScatterIntra.notifyNumPerThread, resReqScatterInter.notifyNumPerThread});
    resourceRequest.notifyNumOnMainThread = std::max({resReqScatterIntra.notifyNumOnMainThread, resReqScatterInter.notifyNumOnMainThread});

    u64 channelsSize = 2;
    resourceRequest.channels.resize(channelsSize);
    resourceRequest.channels[0] = resReqScatterIntra.channels[0];
    resourceRequest.channels[1] = resReqScatterInter.channels[0];

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ScatterSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExecutorInfo(param, resCtx));
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ScatterSequenceExecutor][Orchestrate]errNo[0x%016llx] Scatter excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitExecutorInfo(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size(); // myrank在level0内的rankIdx
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size(); // myrank在level1内的rankIdx

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();

    // 计算框内的root同号卡
    intraLocalRoot_ = root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankSizeLevel0_;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO("[InsV2ScatterSequenceExecutor][InitExecutorInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        +myRank_,
        rankSize_,
        dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2ScatterSequenceExecutor][OrchestrateLoop] Start");

    // 不区分CCL-IN 与 CCL-OUT
    // 声明框内Scatter templateargs
    TemplateDataParams tempAlgParamsScatterIntra;
    tempAlgParamsScatterIntra.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterIntra.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterIntra.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框内Scatter template
    std::shared_ptr<InsAlgTemplate0> algTemplateScatterIntra =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间Scatter templateargs
    TemplateDataParams tempAlgParamsScatterInter;
    tempAlgParamsScatterInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterInter.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsScatterInter.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框间Scatter template
    std::shared_ptr<InsAlgTemplate1> algTemplateScatterInter =
        std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 构造框内template资源
    TemplateResource templateResourceIntra;
    templateResourceIntra.channels = remoteRankToChannelInfo_[0];
    templateResourceIntra.threads = resCtx.threads;
    templateResourceIntra.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceIntra.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    // 构造框间template资源
    TemplateResource templateResourceInter;
    templateResourceInter.channels = remoteRankToChannelInfo_[1];
    templateResourceInter.threads = resCtx.threads;
    templateResourceInter.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceInter.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = tempAlgParamsScatterIntra.buffInfo.hcclBuff.size / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // ----------- 框内Scatter数据搬运 -----------
        // 框内Scatter的数据偏移和搬运量计算
        tempAlgParamsScatterIntra.count = currDataCount;
        tempAlgParamsScatterIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsScatterIntra.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterIntra.buffInfo.hcclBuffBaseOff = 0;

        CHK_RET(SplitData(
            currDataCount, rankSizeLevel0_, tempAlgParamsScatterIntra));  // 计算每个卡对应位置的offset,count,size
        CHK_PRT_RET(tempAlgParamsScatterIntra.allRankSliceSize.size() != rankSizeLevel0_,
            HCCL_ERROR("[InsV2ScatterSequenceExecutor][tempAlgParamsScatterIntra] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterIntra.allRankSliceSize.size(),
                rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);

        tempAlgParamsScatterIntra.sliceSize = 0;
        tempAlgParamsScatterIntra.tailSize = 0;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterIntra.inputSliceStride = 0;
        tempAlgParamsScatterIntra.outputSliceStride = 0;

        // 不需要重复
        tempAlgParamsScatterIntra.repeatNum = 1;
        tempAlgParamsScatterIntra.inputRepeatStride = 0;
        tempAlgParamsScatterIntra.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        if(intraLocalRoot_ == root_) {
            CHK_RET(algTemplateScatterIntra->KernelRun(param, tempAlgParamsScatterIntra, templateResourceIntra));
        }

        // ----------- 框间Scatter数据搬运 -----------
        // 框间的数据偏移和搬运计算
        tempAlgParamsScatterInter.count = tempAlgParamsScatterIntra.allRankProcessedDataCount.at(rankIdxLevel0_);
        tempAlgParamsScatterInter.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.hcclBuffBaseOff = tempAlgParamsScatterIntra.allRankDispls.at(rankIdxLevel0_); // 将框内的切片偏移传到框间
        tempAlgParamsScatterInter.root = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsScatterInter.sliceSize = 0;
        tempAlgParamsScatterInter.tailSize = 0;

        CHK_RET(SplitData(tempAlgParamsScatterInter.count, rankSizeLevel1_, tempAlgParamsScatterInter));
        CHK_PRT_RET(tempAlgParamsScatterInter.allRankSliceSize.size() != rankSizeLevel1_,
            HCCL_ERROR("[InsV2ScatterSequenceExecutor][tempAlgParamsScatterInter] slice num[%u] is not equal to rank "
                       "size[%u].",
                tempAlgParamsScatterInter.allRankSliceSize.size(),
                rankSizeLevel1_),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO("[InsV2ScatterSequenceExecutor][SplitData][ScatterInter] count[%u] slicenum[%u]",
            tempAlgParamsScatterInter.count, rankSizeLevel1_);

        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterInter.inputSliceStride = 0;
        tempAlgParamsScatterInter.outputSliceStride = 0;

        HCCL_DEBUG("[InsV2ScatterSequenceExecutor] loop [%u] tempAlgParamsScatterInter.inputSliceStride [%u],"
                  "tempAlgParamsScatterInter.outputSliceStride [%u] tempAlgParamsScatterInter.sliceSize [%u]",
            loop,
            tempAlgParamsScatterInter.inputSliceStride,
            tempAlgParamsScatterInter.outputSliceStride,
            tempAlgParamsScatterInter.sliceSize);
        HCCL_DEBUG("[InsV2ScatterSequenceExecutor] loop [%u] tempAlgParamsScatterInter.buffInfo.inBuffBaseOff [%u],"
                  "tempAlgParamsScatterInter.buffInfo.outBuffBaseOff [%u]",
            loop,
            tempAlgParamsScatterInter.buffInfo.inBuffBaseOff,
            tempAlgParamsScatterInter.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsScatterInter.repeatNum = 1;
        tempAlgParamsScatterInter.inputRepeatStride = 0;
        tempAlgParamsScatterInter.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        if (tempAlgParamsScatterInter.count != 0) {  // 如果卡里没有数据，不需要参与框间
            CHK_RET(algTemplateScatterInter->KernelRun(param, tempAlgParamsScatterInter, templateResourceInter));
        }

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2ScatterSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SplitData(const u64 &dataCount, const uint64_t &rankSize, TemplateDataParams &tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    u64 offsetCount = 0;
    u64 offsetSize = 0;
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount >= sliceCount) {
            tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            tempAlgParams.allRankSliceSize.emplace_back(curSliceSize);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(curSliceCount);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG("[InsV2ScatterSequenceExecutor] SliceInfo: offset[%u] size[%u] count[%u]",
            tempAlgParams.allRankDispls.at(i),
            tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
u64 InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2ScatterSequenceExecutor][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_SCATTER, InsScatterSequenceMeshNhrDPU, InsV2ScatterSequenceExecutor,
    TopoMatchMultilevel, InsTempScatterMesh1D, InsTempScatterNHRDPU);
}  // namespace ops_hccl