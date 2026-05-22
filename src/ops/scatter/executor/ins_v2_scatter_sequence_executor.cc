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
        HCCL_ERROR("[InsV2ScatterSequenceExecutor][Orchestrate]errNo[0x%016llx] Broadcast excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ScatterSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitExecutorInfo(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();

    // 计算框内的root同号卡
    intraLocalRoot_ = root_ % rankSizeLevel0_ + rankIdxLevel1_ * rankIdxLevel0_;

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
    // 从input buf取要传输的数据
    tempAlgParamsScatterIntra.buffInfo.inputPtr = param.inputPtr;
    // 传输到对端的ccl buf的起始地址
    tempAlgParamsScatterIntra.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterIntra.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框内Scatter template
    std::shared_ptr<InsAlgTemplate0> algTemplateScatterIntra =
        std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间Scatter templateargs
    TemplateDataParams tempAlgParamsScatterInter;
    // 从ccl buffer取数据
    tempAlgParamsScatterInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    // 传输到对端的ccl buffer
    tempAlgParamsScatterInter.buffInfo.outputPtr = resCtx.cclMem.addr;
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
    u64 dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    u64 dataCount_ = param.DataDes.count;
    u64 maxCountPerLoop = tempAlgParamsScatterIntra.buffInfo.hcclBuff.size / rankSizeLevel1_ / HCCL_MIN_SLICE_ALIGN *
                          HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // ----------- 框内Scatter数据搬运 -----------
        // 框内Scatter的数据偏移和搬运量计算
        tempAlgParamsScatterIntra.count = currDataCount;
        tempAlgParamsScatterIntra.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsScatterIntra.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterIntra.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsScatterIntra.sliceSize = currDataCount * dataTypeSize_;
        tempAlgParamsScatterIntra.tailSize = tempAlgParamsScatterIntra.sliceSize;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterIntra.inputSliceStride = dataCount_;
        tempAlgParamsScatterIntra.outputSliceStride = 0;

        // m * n组网需要做m次重复
        tempAlgParamsScatterIntra.repeatNum = rankSizeLevel1_;
        tempAlgParamsScatterIntra.inputRepeatStride = rankSizeLevel0_ * dataSize_;
        // outputRepeatStride等于0是因为每次循环都会把数据无气泡发送给对端（属于m个rank的数据没有空隙）
        tempAlgParamsScatterIntra.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateScatterIntra->KernelRun(param, tempAlgParamsScatterIntra, templateResourceIntra));

        // ----------- 框间Scatter数据搬运 -----------
        // 框间的数据偏移和搬运计算
        tempAlgParamsScatterInter.count = currDataCount;
        // 只有第二段才需要dpu通信
        tempAlgParamsScatterInter.buffInfo.inBuffBaseOff = currDataCount;
        tempAlgParamsScatterInter.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.hcclBuffBaseOff = 0; // 上一步将数据都搬到了cclbuffer起始地址
        tempAlgParamsScatterInter.root = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsScatterInter.sliceSize = 0;
        tempAlgParamsScatterInter.tailSize = 0;

        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParamsScatterInter.inputSliceStride = 0;
        tempAlgParamsScatterInter.outputSliceStride = 0;

        // 
        tempAlgParamsScatterInter.repeatNum = rankSizeLevel1_;
        tempAlgParamsScatterInter.inputRepeatStride = currDataCount;
        tempAlgParamsScatterInter.outputRepeatStride = 0; //搬到对端cclbuffer的起始地址
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        if (tempAlgParamsScatterInter.count != 0) {  // 如果卡里没有数据，不需要参与框间
            CHK_RET(algTemplateScatterInter->KernelRun(param, tempAlgParamsScatterInter, templateResourceInter));
        }

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2ScatterSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_SCATTER, InsScatterSequenceMeshNhrDPU, InsV2ScatterSequenceExecutor,
    TopoMatchMultilevel, InsTempScatterMesh1D, InsTempScatterNHRDPU);
}  // namespace ops_hccl