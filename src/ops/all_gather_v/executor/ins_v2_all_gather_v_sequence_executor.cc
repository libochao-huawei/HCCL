/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_v_sequence_executor.h"
#include "topo_match_multilevel_mesh1d.h"
#include "ins_temp_all_gather_v_mesh_1D.h"
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSequenceExecutor<AlgTopoMatch, InsAlgTemplate>::InitCommInfo(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.vDataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    totalLevels_ = topoInfo->topoLevelNums;

    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (totalLevels_ >= COMM_LAYER_SIZE_2) {
        rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    }
    if (totalLevels_ >= COMM_LAYER_SIZE_3) {
        rankSizeLevel2_ = algHierarchyInfo.infos[2][0].size();
    }

    HCCL_INFO("[InsV2AllGatherVSequenceExecutor][InitCommInfo] myRank[%u], rankSize[%u], "
        "totalLevels[%u], rankSizeLevel0[%u], rankSizeLevel1[%u], rankSizeLevel2[%u]",
        myRank_, rankSize_, totalLevels_, rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSequenceExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;

    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSequenceExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    InitCommInfo(comm, param, topoInfo, algHierarchyInfo);

    // Layer 0 资源
    InsAlgTemplate tempAlgL0(param, myRank_, algHierarchyInfo.infos[0]);
    AlgResourceRequest resReqL0;
    CHK_RET(tempAlgL0.CalcRes(comm, param, topoInfo, resReqL0));

    resourceRequest.channels.push_back(resReqL0.channels[0]);
    resourceRequest.slaveThreadNum = resReqL0.slaveThreadNum;
    resourceRequest.notifyNumPerThread = resReqL0.notifyNumPerThread;
    resourceRequest.notifyNumOnMainThread = resReqL0.notifyNumOnMainThread;

    if (totalLevels_ >= COMM_LAYER_SIZE_2) {
        // Layer 1 资源
        InsAlgTemplate tempAlgL1(param, myRank_, algHierarchyInfo.infos[1]);
        AlgResourceRequest resReqL1;
        CHK_RET(tempAlgL1.CalcRes(comm, param, topoInfo, resReqL1));

        resourceRequest.channels.push_back(resReqL1.channels[0]);
        // 分级算法，slaveThread和notify可以复用（顺序执行）
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, resReqL1.slaveThreadNum);
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread,
            resReqL1.notifyNumOnMainThread);
    }

    if (totalLevels_ >= COMM_LAYER_SIZE_3) {
        // Layer 2 资源
        InsAlgTemplate tempAlgL2(param, myRank_, algHierarchyInfo.infos[2]);
        AlgResourceRequest resReqL2;
        CHK_RET(tempAlgL2.CalcRes(comm, param, topoInfo, resReqL2));

        resourceRequest.channels.push_back(resReqL2.channels[0]);
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, resReqL2.slaveThreadNum);
        resourceRequest.notifyNumOnMainThread = std::max(resourceRequest.notifyNumOnMainThread,
            resReqL2.notifyNumOnMainThread);
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSequenceExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherVSequenceExecutor][Orchestrate] Orchestrate Start");

    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.vDataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    maxTmpMemSize_ = resCtx.cclMem.size;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    totalLevels_ = algHierarchyInfo_.infos.size();
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    if (totalLevels_ >= COMM_LAYER_SIZE_2) {
        rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    }
    if (totalLevels_ >= COMM_LAYER_SIZE_3) {
        rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    }

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherVSequenceExecutor][Orchestrate]errNo[0x%016llx] Orchestrate failed",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[InsV2AllGatherVSequenceExecutor][Orchestrate] Orchestrate End");
    return HCCL_SUCCESS;
}

/**
 * OrchestrateLoop: 顺序执行每层的AllGatherV
 * 
 * 数据流:
 *   Layer 0 (pod内): inputPtr → outputPtr, 使用pod组原始counts
 *   Layer 1 (跨pod): outputPtr → outputPtr, 使用聚合后的podTotalCounts
 *   Layer 2 (跨超节点): outputPtr → outputPtr, 使用聚合后的supernodeTotalCounts
 *
 * counts聚合规则 (对称部署):
 *   Layer 1: podTotal[rank r] = sum(globalCounts[podStart(r)..podStart(r)+rankSizeLevel0_-1])
 *            其中 podStart(r) = (r / rankSizeLevel0_) * rankSizeLevel0_
 *   Layer 2: supernodeTotal[rank r] = sum of podTotals for all pods in rank r's supernode group
 */
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AllGatherVSequenceExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherVSequenceExecutor][OrchestrateLoop] Start, totalLevels[%u]", totalLevels_);

    // 从varData提取全局counts和displs
    const u64 *varData = reinterpret_cast<const u64 *>(param.varData);
    std::vector<u64> globalCounts(varData, varData + rankSize_);
    std::vector<u64> globalDispls(varData + rankSize_, varData + rankSize_ + rankSize_);

    // 传输上限 (所有层共用)
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;

    // ===== Layer 0: pod内 AllGatherV =====
    {
        const std::vector<u32>& layer0Ranks = algHierarchyInfo_.infos[0][0];
        u32 layer0RankSize = layer0Ranks.size();

        // 计算Layer 0的counts和displs (pod组的原始counts/displs)
        std::vector<u64> layer0Counts(layer0RankSize);
        std::vector<u64> layer0Displs(layer0RankSize);
        for (u32 i = 0; i < layer0RankSize; i++) {
            layer0Counts[i] = globalCounts[layer0Ranks[i]];
            layer0Displs[i] = globalDispls[layer0Ranks[i]];
        }

        // 找myAlgRank在layer0中的位置
        auto itL0 = std::find(layer0Ranks.begin(), layer0Ranks.end(), myRank_);
        CHK_PRT_RET(itL0 == layer0Ranks.end(),
            HCCL_ERROR("[InsV2AllGatherVSequenceExecutor] myRank[%u] not found in layer0Ranks", myRank_),
            HCCL_E_INTERNAL);
        u32 myAlgRankL0 = static_cast<u32>(itL0 - layer0Ranks.begin());

        // 创建Layer 0 template
        InsAlgTemplate tempAlgL0(param, myRank_, algHierarchyInfo_.infos[0]);

        // 计算scratch和loop参数
        u32 templateScratchMultiplierL0 = tempAlgL0.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT);
        u64 maxDataSizePerLoopL0 = 0;
        if (templateScratchMultiplierL0 != 0) {
            u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplierL0 /
                HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
            maxDataSizePerLoopL0 = std::min(transportBoundDataSize, scratchBoundDataSize);
        } else {
            maxDataSizePerLoopL0 = transportBoundDataSize;
        }
        u64 maxCountPerLoopL0 = maxDataSizePerLoopL0 / dataTypeSize_;

        u64 maxSendDataCountL0 = 0;
        for (u32 i = 0; i < layer0RankSize; i++) {
            maxSendDataCountL0 = std::max(maxSendDataCountL0, layer0Counts[i]);
        }
        u64 loopTimesL0 = 1 + ((maxSendDataCountL0 - 1) / maxCountPerLoopL0);

        // 构建Layer 0 template资源
        TemplateResource templateResL0;
        templateResL0.channels = remoteRankToChannelInfo_[0];
        templateResL0.threads = resCtx.threads;
        templateResL0.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        templateResL0.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

        // 逐loop执行Layer 0
        std::vector<u64> processedDataCountL0(layer0RankSize, 0);
        TemplateDataParams tempParamsL0;
        tempParamsL0.buffInfo.inputPtr = param.inputPtr;
        tempParamsL0.buffInfo.outputPtr = param.outputPtr;
        tempParamsL0.buffInfo.inputSize = param.inputSize;
        tempParamsL0.buffInfo.outputSize = param.outputSize;
        tempParamsL0.buffInfo.hcclBuff = resCtx.cclMem;
        tempParamsL0.buffInfo.inBuffType = BufferType::INPUT;
        tempParamsL0.buffInfo.outBuffType = BufferType::OUTPUT;
        tempParamsL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempParamsL0.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempParamsL0.sliceSize = 0;
        tempParamsL0.tailSize = 0;

        for (u64 loop = 0; loop < loopTimesL0; loop++) {
            tempParamsL0.allRankSliceSize = {};
            for (u32 i = 0; i < layer0RankSize; i++) {
                tempParamsL0.allRankSliceSize.push_back(
                    ((processedDataCountL0[i] < layer0Counts[i]) ?
                        std::min(maxCountPerLoopL0, layer0Counts[i] - processedDataCountL0[i]) : 0) * dataTypeSize_);
                if (loop == 0) {
                    tempParamsL0.sliceSize = std::max(tempParamsL0.sliceSize, tempParamsL0.allRankSliceSize[i]);
                }
            }

            u64 myProcessedCountL0 = processedDataCountL0[myAlgRankL0];
            tempParamsL0.buffInfo.inBuffBaseOff = myProcessedCountL0 * dataTypeSize_;
            tempParamsL0.buffInfo.outBuffBaseOff = myProcessedCountL0 * dataTypeSize_;
            tempParamsL0.buffInfo.hcclBuffBaseOff = 0;
            tempParamsL0.allRankDispls = layer0Displs;
            tempParamsL0.allRankProcessedDataCount = processedDataCountL0;
            tempParamsL0.inputSliceStride = dataSize_;
            tempParamsL0.outputSliceStride = maxCountPerLoopL0 * dataTypeSize_;
            tempParamsL0.repeatNum = 1;
            tempParamsL0.inputRepeatStride = 0;
            tempParamsL0.outputRepeatStride = 0;

            HCCL_INFO("[InsV2AllGatherVSequenceExecutor] L0 loop[%llu] myProcessedCount[%llu] "
                "inputSliceStride[%llu] outputSliceStride[%llu] sliceSize[%llu]",
                loop, myProcessedCountL0, tempParamsL0.inputSliceStride,
                tempParamsL0.outputSliceStride, tempParamsL0.sliceSize);

            CHK_RET(tempAlgL0.KernelRun(param, tempParamsL0, templateResL0));

            for (u32 i = 0; i < layer0RankSize; i++) {
                processedDataCountL0[i] += tempParamsL0.allRankSliceSize[i] / dataTypeSize_;
            }
            tempParamsL0.tailSize += maxCountPerLoopL0;
        }
        HCCL_INFO("[InsV2AllGatherVSequenceExecutor] Layer 0 AllGatherV completed.");
    }

    // ===== Layer 1: 跨pod AllGatherV =====
    if (totalLevels_ >= COMM_LAYER_SIZE_2) {
        const std::vector<u32>& layer1Ranks = algHierarchyInfo_.infos[1][0];
        u32 layer1RankSize = layer1Ranks.size();

        // 计算Layer 1的聚合counts (podTotal) 和 displs (pod起始displacement)
        std::vector<u64> layer1Counts(layer1RankSize);
        std::vector<u64> layer1Displs(layer1RankSize);
        for (u32 i = 0; i < layer1RankSize; i++) {
            u32 rank = layer1Ranks[i];
            // 对称部署: podStart = (rank / rankSizeLevel0_) * rankSizeLevel0_
            u32 podStartIdx = (rank / rankSizeLevel0_) * rankSizeLevel0_;
            u64 podTotal = 0;
            for (u32 j = 0; j < rankSizeLevel0_ && (podStartIdx + j) < rankSize_; j++) {
                podTotal += globalCounts[podStartIdx + j];
            }
            layer1Counts[i] = podTotal;
            layer1Displs[i] = globalDispls[podStartIdx];
        }

        // 找myAlgRank在layer1中的位置
        auto itL1 = std::find(layer1Ranks.begin(), layer1Ranks.end(), myRank_);
        CHK_PRT_RET(itL1 == layer1Ranks.end(),
            HCCL_ERROR("[InsV2AllGatherVSequenceExecutor] myRank[%u] not found in layer1Ranks", myRank_),
            HCCL_E_INTERNAL);
        u32 myAlgRankL1 = static_cast<u32>(itL1 - layer1Ranks.begin());

        // 创建Layer 1 template
        InsAlgTemplate tempAlgL1(param, myRank_, algHierarchyInfo_.infos[1]);

        // 计算scratch和loop参数
        u32 templateScratchMultiplierL1 = tempAlgL1.CalcScratchMultiple(BufferType::OUTPUT, BufferType::OUTPUT);
        u64 maxDataSizePerLoopL1 = 0;
        if (templateScratchMultiplierL1 != 0) {
            u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplierL1 /
                HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
            maxDataSizePerLoopL1 = std::min(transportBoundDataSize, scratchBoundDataSize);
        } else {
            maxDataSizePerLoopL1 = transportBoundDataSize;
        }
        u64 maxCountPerLoopL1 = maxDataSizePerLoopL1 / dataTypeSize_;

        u64 maxSendDataCountL1 = 0;
        for (u32 i = 0; i < layer1RankSize; i++) {
            maxSendDataCountL1 = std::max(maxSendDataCountL1, layer1Counts[i]);
        }
        u64 loopTimesL1 = 1 + ((maxSendDataCountL1 - 1) / maxCountPerLoopL1);

        // 构建Layer 1 template资源
        TemplateResource templateResL1;
        templateResL1.channels = remoteRankToChannelInfo_[1];
        templateResL1.threads = resCtx.threads;
        templateResL1.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        templateResL1.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

        // 逐loop执行Layer 1
        // 注意: Layer 1的输入来自Layer 0的输出 (outputPtr), inputPtr = outputPtr
        std::vector<u64> processedDataCountL1(layer1RankSize, 0);
        TemplateDataParams tempParamsL1;
        tempParamsL1.buffInfo.inputPtr = param.outputPtr;  // 读取Layer 0的结果
        tempParamsL1.buffInfo.outputPtr = param.outputPtr;
        tempParamsL1.buffInfo.inputSize = param.outputSize;
        tempParamsL1.buffInfo.outputSize = param.outputSize;
        tempParamsL1.buffInfo.hcclBuff = resCtx.cclMem;
        tempParamsL1.buffInfo.inBuffType = BufferType::OUTPUT;
        tempParamsL1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempParamsL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempParamsL1.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempParamsL1.sliceSize = 0;
        tempParamsL1.tailSize = 0;

        for (u64 loop = 0; loop < loopTimesL1; loop++) {
            tempParamsL1.allRankSliceSize = {};
            for (u32 i = 0; i < layer1RankSize; i++) {
                tempParamsL1.allRankSliceSize.push_back(
                    ((processedDataCountL1[i] < layer1Counts[i]) ?
                        std::min(maxCountPerLoopL1, layer1Counts[i] - processedDataCountL1[i]) : 0) * dataTypeSize_);
                if (loop == 0) {
                    tempParamsL1.sliceSize = std::max(tempParamsL1.sliceSize, tempParamsL1.allRankSliceSize[i]);
                }
            }

            u64 myProcessedCountL1 = processedDataCountL1[myAlgRankL1];
            tempParamsL1.buffInfo.inBuffBaseOff = myProcessedCountL1 * dataTypeSize_;
            tempParamsL1.buffInfo.outBuffBaseOff = myProcessedCountL1 * dataTypeSize_;
            tempParamsL1.buffInfo.hcclBuffBaseOff = 0;
            tempParamsL1.allRankDispls = layer1Displs;
            tempParamsL1.allRankProcessedDataCount = processedDataCountL1;
            tempParamsL1.inputSliceStride = layer1Counts[myAlgRankL1] * dataTypeSize_;
            tempParamsL1.outputSliceStride = maxCountPerLoopL1 * dataTypeSize_;
            tempParamsL1.repeatNum = 1;
            tempParamsL1.inputRepeatStride = 0;
            tempParamsL1.outputRepeatStride = 0;

            HCCL_INFO("[InsV2AllGatherVSequenceExecutor] L1 loop[%llu] myProcessedCount[%llu] "
                "inputSliceStride[%llu] outputSliceStride[%llu] sliceSize[%llu]",
                loop, myProcessedCountL1, tempParamsL1.inputSliceStride,
                tempParamsL1.outputSliceStride, tempParamsL1.sliceSize);

            CHK_RET(tempAlgL1.KernelRun(param, tempParamsL1, templateResL1));

            for (u32 i = 0; i < layer1RankSize; i++) {
                processedDataCountL1[i] += tempParamsL1.allRankSliceSize[i] / dataTypeSize_;
            }
            tempParamsL1.tailSize += maxCountPerLoopL1;
        }
        HCCL_INFO("[InsV2AllGatherVSequenceExecutor] Layer 1 AllGatherV completed.");
    }

    // ===== Layer 2: 跨超节点 AllGatherV =====
    if (totalLevels_ >= COMM_LAYER_SIZE_3) {
        const std::vector<u32>& layer2Ranks = algHierarchyInfo_.infos[2][0];
        u32 layer2RankSize = layer2Ranks.size();

        // 计算Layer 2的聚合counts (supernodeTotal) 和 displs
        // supernodeTotal = sum of podTotals for all pods in this rank's supernode group
        std::vector<u64> layer2Counts(layer2RankSize);
        std::vector<u64> layer2Displs(layer2RankSize);
        for (u32 i = 0; i < layer2RankSize; i++) {
            u32 rank = layer2Ranks[i];
            // 对称部署: 同一超节点组内的pod起止范围
            // 每个超节点包含 rankSizeLevel1_ 个pod, 每个pod包含 rankSizeLevel0_ 个rank
            // 超节点组的起始: (rank / (rankSizeLevel0_ * rankSizeLevel1_)) * (rankSizeLevel0_ * rankSizeLevel1_)
            u32 supernodeGroupStart = (rank / (rankSizeLevel0_ * rankSizeLevel1_)) *
                                       (rankSizeLevel0_ * rankSizeLevel1_);
            u64 supernodeTotal = 0;
            for (u32 j = 0; j < rankSizeLevel0_ * rankSizeLevel1_ && (supernodeGroupStart + j) < rankSize_; j++) {
                supernodeTotal += globalCounts[supernodeGroupStart + j];
            }
            layer2Counts[i] = supernodeTotal;
            layer2Displs[i] = globalDispls[supernodeGroupStart];
        }

        // 找myAlgRank在layer2中的位置
        auto itL2 = std::find(layer2Ranks.begin(), layer2Ranks.end(), myRank_);
        CHK_PRT_RET(itL2 == layer2Ranks.end(),
            HCCL_ERROR("[InsV2AllGatherVSequenceExecutor] myRank[%u] not found in layer2Ranks", myRank_),
            HCCL_E_INTERNAL);
        u32 myAlgRankL2 = static_cast<u32>(itL2 - layer2Ranks.begin());

        // 创建Layer 2 template
        InsAlgTemplate tempAlgL2(param, myRank_, algHierarchyInfo_.infos[2]);

        // 计算scratch和loop参数
        u32 templateScratchMultiplierL2 = tempAlgL2.CalcScratchMultiple(BufferType::OUTPUT, BufferType::OUTPUT);
        u64 maxDataSizePerLoopL2 = 0;
        if (templateScratchMultiplierL2 != 0) {
            u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplierL2 /
                HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
            maxDataSizePerLoopL2 = std::min(transportBoundDataSize, scratchBoundDataSize);
        } else {
            maxDataSizePerLoopL2 = transportBoundDataSize;
        }
        u64 maxCountPerLoopL2 = maxDataSizePerLoopL2 / dataTypeSize_;

        u64 maxSendDataCountL2 = 0;
        for (u32 i = 0; i < layer2RankSize; i++) {
            maxSendDataCountL2 = std::max(maxSendDataCountL2, layer2Counts[i]);
        }
        u64 loopTimesL2 = 1 + ((maxSendDataCountL2 - 1) / maxCountPerLoopL2);

        // 构建Layer 2 template资源
        TemplateResource templateResL2;
        templateResL2.channels = remoteRankToChannelInfo_[2];
        templateResL2.threads = resCtx.threads;
        templateResL2.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        templateResL2.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

        // 逐loop执行Layer 2
        std::vector<u64> processedDataCountL2(layer2RankSize, 0);
        TemplateDataParams tempParamsL2;
        tempParamsL2.buffInfo.inputPtr = param.outputPtr;  // 读取Layer 1的结果
        tempParamsL2.buffInfo.outputPtr = param.outputPtr;
        tempParamsL2.buffInfo.inputSize = param.outputSize;
        tempParamsL2.buffInfo.outputSize = param.outputSize;
        tempParamsL2.buffInfo.hcclBuff = resCtx.cclMem;
        tempParamsL2.buffInfo.inBuffType = BufferType::OUTPUT;
        tempParamsL2.buffInfo.outBuffType = BufferType::OUTPUT;
        tempParamsL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        tempParamsL2.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
        tempParamsL2.sliceSize = 0;
        tempParamsL2.tailSize = 0;

        for (u64 loop = 0; loop < loopTimesL2; loop++) {
            tempParamsL2.allRankSliceSize = {};
            for (u32 i = 0; i < layer2RankSize; i++) {
                tempParamsL2.allRankSliceSize.push_back(
                    ((processedDataCountL2[i] < layer2Counts[i]) ?
                        std::min(maxCountPerLoopL2, layer2Counts[i] - processedDataCountL2[i]) : 0) * dataTypeSize_);
                if (loop == 0) {
                    tempParamsL2.sliceSize = std::max(tempParamsL2.sliceSize, tempParamsL2.allRankSliceSize[i]);
                }
            }

            u64 myProcessedCountL2 = processedDataCountL2[myAlgRankL2];
            tempParamsL2.buffInfo.inBuffBaseOff = myProcessedCountL2 * dataTypeSize_;
            tempParamsL2.buffInfo.outBuffBaseOff = myProcessedCountL2 * dataTypeSize_;
            tempParamsL2.buffInfo.hcclBuffBaseOff = 0;
            tempParamsL2.allRankDispls = layer2Displs;
            tempParamsL2.allRankProcessedDataCount = processedDataCountL2;
            tempParamsL2.inputSliceStride = layer2Counts[myAlgRankL2] * dataTypeSize_;
            tempParamsL2.outputSliceStride = maxCountPerLoopL2 * dataTypeSize_;
            tempParamsL2.repeatNum = 1;
            tempParamsL2.inputRepeatStride = 0;
            tempParamsL2.outputRepeatStride = 0;

            HCCL_INFO("[InsV2AllGatherVSequenceExecutor] L2 loop[%llu] myProcessedCount[%llu] "
                "inputSliceStride[%llu] outputSliceStride[%llu] sliceSize[%llu]",
                loop, myProcessedCountL2, tempParamsL2.inputSliceStride,
                tempParamsL2.outputSliceStride, tempParamsL2.sliceSize);

            CHK_RET(tempAlgL2.KernelRun(param, tempParamsL2, templateResL2));

            for (u32 i = 0; i < layer2RankSize; i++) {
                processedDataCountL2[i] += tempParamsL2.allRankSliceSize[i] / dataTypeSize_;
            }
            tempParamsL2.tailSize += maxCountPerLoopL2;
        }
        HCCL_INFO("[InsV2AllGatherVSequenceExecutor] Layer 2 AllGatherV completed.");
    }

    HCCL_INFO("[InsV2AllGatherVSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

// 注册: AllGatherV多层级Mesh1D执行器（2~3层拓扑）
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER_V, InsAllGatherVMultilevelMesh1D,
    InsV2AllGatherVSequenceExecutor, TopoMatchMultilevelMesh1D, InsTempAllGatherVMesh1D);

}  // namespace ops_hccl