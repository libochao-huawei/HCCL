/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "reduce_parallel_executor.h"
#include "coll_alg_v2_exec_registry.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#include "ccu_temp_reduce_scatter_nhr_1D_mem2mem.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::ReduceParallelExecutor()
{}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    myRank_ = topoInfo->userRank;
    HCCL_INFO("[ReduceParallelExecutor] CalcRes start, rank[%d]", myRank_);

    // 实例化算法模板类
    // 构建template
    // reduceScatter intra
    algTemplatePtrArr_.at(0).at(0) =
        std::make_shared<AlgTemplate0>(param, topoInfo->userRank, algHierarchyInfo.infos.at(0));
    // reduceScatter inter
    algTemplatePtrArr_.at(0).at(1) =
        std::make_shared<AlgTemplate1>(param, topoInfo->userRank, algHierarchyInfo.infos.at(1));
    // allGather intra
    algTemplatePtrArr_.at(1).at(0) =
        std::make_shared<AlgTemplate2>(param, topoInfo->userRank, algHierarchyInfo.infos.at(0));
    // allGather inter
    algTemplatePtrArr_.at(1).at(1) =
        std::make_shared<AlgTemplate3>(param, topoInfo->userRank, algHierarchyInfo.infos.at(1));

    // 计算资源
    AlgResourceRequest reduceScatterIntraTempRequest;
    AlgResourceRequest reduceScatterInterTempRequest;
    AlgResourceRequest allGatherIntraTempRequest;
    AlgResourceRequest allGatherInterTempRequest;
    AlgResourceRequest intraTempRequestFinal;
    AlgResourceRequest interTempRequestFinal;

    algTemplatePtrArr_.at(0).at(0)->CalcRes(comm, param, topoInfo, reduceScatterIntraTempRequest);
    algTemplatePtrArr_.at(0).at(1)->CalcRes(comm, param, topoInfo, reduceScatterInterTempRequest);
    algTemplatePtrArr_.at(1).at(0)->CalcRes(comm, param, topoInfo, allGatherIntraTempRequest);
    algTemplatePtrArr_.at(1).at(1)->CalcRes(comm, param, topoInfo, allGatherInterTempRequest);

    for (auto &KernelInfo : reduceScatterIntraTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 0;
    }
    for (auto &KernelInfo : reduceScatterInterTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 0;
    }
    for (auto &KernelInfo : allGatherIntraTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 1;
    }
    for (auto &KernelInfo : allGatherInterTempRequest.ccuKernelInfos) {
        KernelInfo.resGroup = 1;
    }

    u32 slaveThreadNumIntraMax = 0;
    if (reduceScatterIntraTempRequest.slaveThreadNum >= allGatherIntraTempRequest.slaveThreadNum) {
        slaveThreadNumIntraMax = reduceScatterIntraTempRequest.slaveThreadNum;
        intraTempRequestFinal.notifyNumPerThread = reduceScatterIntraTempRequest.notifyNumPerThread;
    } else {
        slaveThreadNumIntraMax = allGatherIntraTempRequest.slaveThreadNum;
        intraTempRequestFinal.notifyNumPerThread = allGatherIntraTempRequest.notifyNumPerThread;
    }
    u32 slaveThreadNumInterMax = 0;
    if (reduceScatterInterTempRequest.slaveThreadNum >= allGatherInterTempRequest.slaveThreadNum) {
        slaveThreadNumInterMax = reduceScatterInterTempRequest.slaveThreadNum;
        interTempRequestFinal.notifyNumPerThread = reduceScatterInterTempRequest.notifyNumPerThread;
    } else {
        slaveThreadNumInterMax = allGatherInterTempRequest.slaveThreadNum;
        interTempRequestFinal.notifyNumPerThread = allGatherInterTempRequest.notifyNumPerThread;
    }

    resourceRequest.notifyNumOnMainThread = 1 + 1;  // 用于intra和inter两个template间同步
    // intra主流 + intra从流 + inter主流 + inter从流
    resourceRequest.slaveThreadNum = stageSize_ + slaveThreadNumIntraMax + stageSize_ + slaveThreadNumInterMax;
    resourceRequest.notifyNumPerThread.emplace_back(reduceScatterIntraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.emplace_back(allGatherIntraTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        intraTempRequestFinal.notifyNumPerThread.begin(),
        intraTempRequestFinal.notifyNumPerThread.end());
    resourceRequest.notifyNumPerThread.emplace_back(reduceScatterInterTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.emplace_back(allGatherInterTempRequest.notifyNumOnMainThread + 1);
    resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
        interTempRequestFinal.notifyNumPerThread.begin(),
        interTempRequestFinal.notifyNumPerThread.end());

    if (param.engine != COMM_ENGINE_CCU) {
        resourceRequest.channels.emplace_back(reduceScatterIntraTempRequest.channels.at(0));
        resourceRequest.channels.emplace_back(reduceScatterInterTempRequest.channels.at(0));
    } else {
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] reduceScatterIntraTemp has [%d] kernels.", reduceScatterIntraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            reduceScatterIntraTempRequest.ccuKernelInfos.begin(),
                                            reduceScatterIntraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(reduceScatterIntraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] reduceScatterInterTemp has [%d] kernels.", reduceScatterInterTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            reduceScatterInterTempRequest.ccuKernelInfos.begin(),
                                            reduceScatterInterTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(reduceScatterInterTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] allGatherIntraTemp has [%d] kernels.", allGatherIntraTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            allGatherIntraTempRequest.ccuKernelInfos.begin(),
                                            allGatherIntraTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(allGatherIntraTempRequest.ccuKernelNum[0]);
        HCCL_INFO("[ReduceParallelExecutor][CalcRes] allGatherInterTemp has [%d] kernels.", allGatherInterTempRequest.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                            allGatherInterTempRequest.ccuKernelInfos.begin(),
                                            allGatherInterTempRequest.ccuKernelInfos.end());
        resourceRequest.ccuKernelNum.emplace_back(allGatherInterTempRequest.ccuKernelNum[0]);
    }

    myRank_ = topoInfo->userRank;

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::CalcLocalRoot()
{
    CHK_PRT_RET(root_ >= rankSize_,
        HCCL_ERROR("[ReduceParallelExecutor][CalcLocalRoot] root[%u] is out of rankSize[%u]", root_, rankSize_),
        HcclResult::HCCL_E_INTERNAL);
    rankIdxLevel0_ = myRank_ % intraLocalRankSize_;
    rankIdxLevel1_ = myRank_ / intraLocalRankSize_;
    intraLocalRoot_ = root_ / intraLocalRankSize_ * intraLocalRankSize_ + rankIdxLevel0_;
    interLocalRoot_ = root_ % intraLocalRankSize_ + rankIdxLevel1_ * intraLocalRankSize_;
    HCCL_INFO("[ReduceParallelExecutor][CalcLocalRoot] myRank[%d] intraLocalRoot[%u] interLocalRoot[%u]",
        myRank_,
        intraLocalRoot_,
        interLocalRoot_);
    return HcclResult::HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
uint64_t ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::GetRankSize(
    const std::vector<std::vector<u32>> &vTopo) const
{
    uint64_t count = 1;
    for (const auto &i : vTopo) {
        count *= i.size();
    }
    return count;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[ReduceParallelExecutor][Orchestrate] Orchestrate Start");

    maxTmpMemSize_ = resCtx.cclMem.size;  // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    myRank_ = resCtx.topoInfo.userRank;
    threads_ = resCtx.threads;
    param_ = param;
    resCtx_ = resCtx;
    HCCL_INFO("[ReduceParallelExecutor][Orchestrate] threads_ size[%d]", threads_.size());
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo;
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo));
        intraLinks_ = remoteRankToChannelInfo.at(0);
        interLinks_ = remoteRankToChannelInfo.at(1);
    }
    dataCount_ = param_.DataDes.count;
    dataType_ = param_.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param_.DataDes.dataType];

    root_ = param.root;
    vTopo_ = resCtx.algHierarchyInfo.infos;     // 本通信域内的通信平面

    intraLocalRankSize_ = GetRankSize(vTopo_.at(0));
    interLocalRankSize_ = GetRankSize(vTopo_.at(1));
    rankSize_ = intraLocalRankSize_ * interLocalRankSize_;
    HCCL_DEBUG("[ReduceParallelExecutor][Orchestrate] myRank[%u], intraLocalRankSize_[%u], interLocalRankSize_[%u]",
        myRank_,
        intraLocalRankSize_,
        interLocalRankSize_);

    CHK_RET(CalcLocalRoot());

    // 实例化算法模板类
    algTemplatePtrArr_.at(0).at(0) = std::make_shared<AlgTemplate0>(param, myRank_, vTopo_.at(0));
    algTemplatePtrArr_.at(0).at(1) = std::make_shared<AlgTemplate1>(param, myRank_, vTopo_.at(1));
    algTemplatePtrArr_.at(1).at(0) = std::make_shared<AlgTemplate2>(param, myRank_, vTopo_.at(0));
    algTemplatePtrArr_.at(1).at(1) = std::make_shared<AlgTemplate3>(param, myRank_, vTopo_.at(1));

    // 算法展开
    CHK_RET(OrchestrateImpl());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2,
    AlgTemplate3>::PrepareResForStage(u32 stage)
{
    std::array<std::array<AlgResourceRequest, stepSize_>, stageSize_> tempRequestArr;
    std::array<u32, stageSize_> intraThreadsNum;
    for (u32 stageIdx = 0; stageIdx < stageSize_; stageIdx++) {
        for (u32 stepIdx = 0; stepIdx < stepSize_; stepIdx++) {
            algTemplatePtrArr_.at(stageIdx).at(stepIdx)->GetRes(tempRequestArr.at(stageIdx).at(stepIdx));
        }
        intraThreadsNum.at(stageIdx) = tempRequestArr.at(stageIdx).at(0).slaveThreadNum + 1;
    }

    u32 intraThreadsNumMax = std::max(intraThreadsNum.at(0), intraThreadsNum.at(1));

    // 第0条流是全局主流
    intraThreads_ = {threads_.at(1 + stage)};
    intraThreads_.insert(intraThreads_.end(), threads_.begin() + stageSize_ + 1,
        threads_.begin() + stageSize_ + intraThreadsNum.at(stage));
    interThreads_ = {threads_.at(intraThreadsNumMax + stageSize_ + stage)};
    interThreads_.insert(interThreads_.end(), threads_.begin() + intraThreadsNumMax + stageSize_ + stageSize_,
        threads_.end());

    mainThread_ = threads_.at(0);
    templateMainThreads_ = {intraThreads_.at(0), interThreads_.at(0)};

    syncNotifyOnTemplates_ = {tempRequestArr.at(stage).at(0).notifyNumOnMainThread,
                              tempRequestArr.at(stage).at(1).notifyNumOnMainThread};
    syncNotifyOnMain_ = {0, 1};

    if (param_.engine == COMM_ENGINE_CCU) {
        intraTempAlgRes_.ccuKernels.clear();
        interTempAlgRes_.ccuKernels.clear();
        if (stage == 0) {
            intraTempAlgRes_.ccuKernels.insert(intraTempAlgRes_.ccuKernels.end(), resCtx_.ccuKernels.begin(),
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0]);
            interTempAlgRes_.ccuKernels.insert(interTempAlgRes_.ccuKernels.end(), resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0],
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1]);
        } else {
            intraTempAlgRes_.ccuKernels.insert(intraTempAlgRes_.ccuKernels.end(), resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1],
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1] + resCtx_.ccuKernelNum[2]);
            interTempAlgRes_.ccuKernels.insert(interTempAlgRes_.ccuKernels.end(),
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1] + resCtx_.ccuKernelNum[2],
                                               resCtx_.ccuKernels.begin() + resCtx_.ccuKernelNum[0] + resCtx_.ccuKernelNum[1] + resCtx_.ccuKernelNum[2] + resCtx_.ccuKernelNum[3]);
        }
    } else {
        intraTempAlgRes_.channels = intraLinks_;
        interTempAlgRes_.channels = interLinks_;
    }

    intraTempAlgRes_.threads = intraThreads_;
    intraTempAlgRes_.aivCommInfoPtr = resCtx_.aivCommInfoPtr;

    interTempAlgRes_.threads = interThreads_;
    interTempAlgRes_.aivCommInfoPtr = resCtx_.aivCommInfoPtr;

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
TemplateDataParams ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2,
    AlgTemplate3>:: GenDataParamsTempAlg(u32 dataSliceIdx, u32 stageIdx, u32 stepIdx, bool isInter)
{
    TemplateDataParams dataParams;
    bool isFirstStep = (stageIdx == 0 && stepIdx == 0);

    dataParams.buffInfo.inBuffType = isFirstStep ? BufferType::INPUT : BufferType::HCCL_BUFFER;
    dataParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    dataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 数据在inputBuffer上的起始偏移量，仅用于本次loop第一步firstStep时，要从inputBuffer上取数据
    const u64 inputBufferOffset = dataOffsetPerLoop_.at(dataSliceIdx);
    // totalCount是本次template处理，所有卡需要处理的总数据个数
    u64 totalCount = dataCountPerLoop_.at(dataSliceIdx);
    // dataSliceBaseOffset根据当前要处理的是哪片数据（dataSliceIdx），决定要输出到的ccl缓存起始位置
    const u64 dataSliceBaseOffset = dataSliceIdx == 0 ? 0 : dataCountPerLoop_.at(0) * dataTypeSize_;
    // dataOffset是在非firstStep时，数据输入和输出的起始地址
    u64 dataOffset = dataSliceBaseOffset;

    // 中间step需要额外处理，即stage0-step1（第二次reduceScatter）和stage1-step0（第一次allGather）需要
    // 重新计算dataOffset和totalCount
    if ((stageIdx ^ stepIdx) == 1) {
        const u32 othLocalRankSize = isInter ? intraLocalRankSize_ : interLocalRankSize_;
        const std::map<u32, u32> &othTempVirtRankMap = virtRankMap_.at(!isInter);
        const u32 othLocalRankIdx = othTempVirtRankMap.at(myRank_);
        const u64 trivialSize = dataCountPerLoop_.at(dataSliceIdx) / othLocalRankSize * dataTypeSize_;
        const u64 tailSize = dataCountPerLoop_.at(dataSliceIdx) * dataTypeSize_ - (othLocalRankSize - 1) * trivialSize;
        dataOffset = dataSliceBaseOffset + othLocalRankIdx * trivialSize;
        totalCount = ((othLocalRankIdx + 1 == othLocalRankSize) ? tailSize : trivialSize) / dataTypeSize_;
    }
    const u64 totalSize = totalCount * dataTypeSize_;   // totalSize是本次处理，所有卡需要处理的总数据量

    dataParams.buffInfo.inputPtr = isFirstStep ? param_.inputPtr : resCtx_.cclMem.addr;
    dataParams.buffInfo.inputSize = isFirstStep ? param_.inputSize : resCtx_.cclMem.size;
    dataParams.buffInfo.outputPtr = resCtx_.cclMem.addr;
    dataParams.buffInfo.outputSize = resCtx_.cclMem.size;
    dataParams.buffInfo.hcclBuff = resCtx_.cclMem;
    dataParams.buffInfo.hcclBuffSize = resCtx_.cclMem.size;

    // 前localRankSize - 1个rank的数据片为trivialSize，最后一个rank的数据片大小为tailSize
    const u32 localRankSize = isInter ? interLocalRankSize_ : intraLocalRankSize_;
    u64 trivialSize = totalCount / localRankSize * dataTypeSize_;
    dataParams.tailSize = totalSize - (localRankSize - 1) * trivialSize;
    dataParams.sliceSize = trivialSize;
    dataParams.count = dataParams.sliceSize / dataTypeSize_;

    if (stageIdx == 0 && !isInter) {
        // 框内reduceScatter，需要使用额外的scratchBuffer（不与输入输出相同位置）
        dataParams.buffInfo.hcclBuffBaseOff = (dataCountPerLoop_.at(0) + dataCountPerLoop_.at(1)) * dataTypeSize_;
    } else {
        // AllGather阶段，使用与输入输出相同位置的scratchBuffer，避免localCopy开销
        dataParams.buffInfo.hcclBuffBaseOff = dataOffset;
    }

    if (isFirstStep) {
        dataParams.buffInfo.inBuffBaseOff = inputBufferOffset;
        dataParams.buffInfo.outBuffBaseOff = dataSliceBaseOffset;
    } else {
        dataParams.buffInfo.inBuffBaseOff = dataOffset;
        dataParams.buffInfo.outBuffBaseOff = dataOffset;
    }
    dataParams.inputSliceStride = dataParams.sliceSize;
    dataParams.outputSliceStride = dataParams.sliceSize;

    dataParams.repeatNum = 1;
    dataParams.inputRepeatStride = 0;
    dataParams.outputRepeatStride = 0;

    return dataParams;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::OrchestrateImpl()
{
    for (u32 stage = 0; stage < stageSize_; stage++) {
        for (u32 isInter = 0; isInter < dataSplitPart_; isInter++) {
            HCCL_INFO("[ReduceParallelExecutor][OrchestrateImpl] stage[%u] isInter[%u] [%s]",
                stage,
                isInter,
                algTemplatePtrArr_.at(stage).at(isInter)->Describe().c_str());
        }
    }

    std::array<long double, dataSplitPart_> dataSplitSize{dataSplitSize0_, 1.0 - dataSplitSize0_};

    // inter模板不再需要额外的scratch，因为当input/output都在CCL BUFFER上是，NHR算法可以直接在原地进行
    const long double scratchMultipleIntra = std::max(dataSplitSize.at(0), dataSplitSize.at(1) / interLocalRankSize_);
    // + 1.0是因为要留一份scratch来临时存储中间数据
    const long double totalScratchMultiple = scratchMultipleIntra + 1.0;

    const u64 scratchMemBlockSize = maxTmpMemSize_ / totalScratchMultiple;
    const u64 maxCountPerLoop = std::min<u64>(scratchMemBlockSize, UB_MAX_DATA_SIZE) / dataTypeSize_;
    const u32 loopTimes = dataCount_ / maxCountPerLoop + ((dataCount_ % maxCountPerLoop == 0) ? 0 : 1);

    for (u32 isInter = 0; isInter < dataSplitPart_; isInter++) {
        for (u32 localRank = 0; localRank < vTopo_.at(isInter).at(0).size(); localRank++) {
            const u32 globalRank = vTopo_.at(isInter).at(0).at(localRank);
            virtRankMap_.at(isInter)[globalRank] = localRank;
        }
    }

    CHK_RET(OrchestrateLoop(loopTimes, maxCountPerLoop));

    HCCL_INFO("[ReduceParallelExecutor][OrchestrateImpl] myRank[%d] End.", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::OrchestrateLoop(
        u32 loopTimes, u64 maxCountPerLoop)
{
    u64 processedCount = 0;
    for (u32 loopIndex = 0; loopIndex < loopTimes; loopIndex++) {
        u64 currCount = (loopIndex + 1 == loopTimes) ? (dataCount_ - loopIndex * maxCountPerLoop) : maxCountPerLoop;
        dataCountPerLoop_.at(0) = static_cast<u64>(currCount * dataSplitSize0_);
        dataCountPerLoop_.at(1) = currCount - dataCountPerLoop_.at(0);
        dataOffsetPerLoop_.at(0) = loopIndex * maxCountPerLoop * dataTypeSize_;
        dataOffsetPerLoop_.at(1) = dataOffsetPerLoop_.at(0) + dataCountPerLoop_.at(0) * dataTypeSize_;

        for (u32 stageIdx = 0; stageIdx < 2; stageIdx++) {
            // 计算算法模板所需资源
            CHK_RET(PrepareResForStage(stageIdx));
            // 每个阶段分2步执行任务编排
            for (u32 stepIdx = 0; stepIdx < 2; stepIdx++) {
                CHK_RET(OrchestrateStep(stageIdx, stepIdx));
            }
        }
        if (myRank_ != root_) {
            continue;
        }
        const DataSlice srcSlice(resCtx_.cclMem.addr, 0, currCount * dataTypeSize_);
        const DataSlice dstSlice(param_.outputPtr, processedCount * dataTypeSize_, currCount * dataTypeSize_);
        CHK_RET(LocalCopy(threads_.at(0), srcSlice, dstSlice));

        processedCount += currCount;
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult
    ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::OrchestrateStep(
        u32 stageIdx, u32 stepIdx)
{
    CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
    for (u32 dataSliceIdx = 0; dataSliceIdx < dataSplitPart_; dataSliceIdx++) {
        // 第一个stage第一个step时，第一片数据跑intra，第二片数据跑inter；
        // 第一个stage第二个step时，第一片数据跑inter，第二片数据跑intra；
        // 第二个stage第一个step时，第一片数据跑inter，第二片数据跑intra；
        // 第二个stage第二个step时，第一片数据跑intra，第二片数据跑inter；
        bool isInter = (stageIdx == 1) ^ (stepIdx == 1) ^ (dataSliceIdx == 1);
        CHK_RET(RunTemplate(dataSliceIdx, stageIdx, stepIdx, isInter));
    }
    // 尾同步
    CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
HcclResult ReduceParallelExecutor<AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3>::RunTemplate(
    u32 dataSliceIdx, u32 stageIdx, u32 stepIdx, bool isInter)
{
    if (dataCountPerLoop_.at(dataSliceIdx) == 0) {
        return HCCL_SUCCESS;
    }
    const TemplateDataParams dataParams = GenDataParamsTempAlg(dataSliceIdx, stageIdx, stepIdx, isInter);
    TemplateResource tempAlgRes = isInter ? interTempAlgRes_ : intraTempAlgRes_;
    CHK_RET(algTemplatePtrArr_.at(stageIdx).at(isInter)->KernelRun(param_, dataParams, tempAlgRes));
    return HCCL_SUCCESS;
}

// 算法注册
REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, ReduceParallelMesh1DNHR, ReduceParallelExecutor,
    TopoMatchMultilevel, InsTempReduceScatterMesh1D, InsTempReduceScatterNHR, InsTempAllGatherMesh1D,
    InsTempAllGatherNHR);

#ifndef AICPU_COMPILE
    REGISTER_EXECUTOR_BY_FOUR_TEMPS(HcclCMDType::HCCL_CMD_REDUCE, CcuReduceParallelMesh1DNHR, ReduceParallelExecutor,
        TopoMatchMultilevel, CcuTempReduceScatterMesh1DMem2Mem, CcuTempReduceScatterNHR1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherNHR1DMem2Mem);
#endif
}  // namespace ops_hccl