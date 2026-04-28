/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_ring_pipeline.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {

InsTempAllGatherRingPipeline::InsTempAllGatherRingPipeline(const OpParam &param, const u32 rankId,
                                                           const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAllGatherRingPipeline::~InsTempAllGatherRingPipeline() {}

HcclResult InsTempAllGatherRingPipeline::CalcRes(HcclComm comm, const OpParam &param,
                                                  const TopoInfoWithNetLayerDetails *topoInfo,
                                                  AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherRingPipeline][CalcRes] start");
    GetRes(resourceRequest);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    prevAlgRank_ = (myAlgRank + templateRankSize_ - 1) % templateRankSize_;
    nextAlgRank_ = (myAlgRank + 1) % templateRankSize_;
    prevRank_ = GetRankFromMap(prevAlgRank_);
    nextRank_ = GetRankFromMap(nextAlgRank_);

    HCCL_DEBUG("[InsTempAllGatherRingPipeline][CalcRes] myRank[%u], myAlgRank[%u], "
               "prevRank[%u], nextRank[%u]",
               myRank_, myAlgRank, prevRank_, nextRank_);

    // 优化: 只创建 Ring Pipeline 需要的 2 条通道（前驱+后继），而非 Mesh1D 的 N-1 条
    std::vector<HcclChannelDesc> ringChannels;
    CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, prevRank_, ringChannels));
    CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, nextRank_, ringChannels));

    resourceRequest.channels.push_back(ringChannels);
    HCCL_DEBUG("[InsTempAllGatherRingPipeline][CalcRes] ringChannels.size()=%zu (optimized from %u)",
               ringChannels.size(), templateRankSize_ - 1);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRingPipeline::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherRingPipeline::GetThreadNum() const
{
    return 1;
}

u64 InsTempAllGatherRingPipeline::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult InsTempAllGatherRingPipeline::KernelRun(const OpParam &param,
                                                    const TemplateDataParams &tempAlgParams,
                                                    TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAllGatherRingPipeline] Run start");

    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherRingPipeline] Rank [%d], sliceSize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = 1;
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;

    CHK_PRT_RET(threadNum_ != templateResource.threads.size(),
                HCCL_ERROR("[InsTempAllGatherRingPipeline] Rank [%d], required threadNum [%u] != actual [%zu].",
                           myRank_, threadNum_, templateResource.threads.size()),
                HcclResult::HCCL_E_INTERNAL);

    bool isPcieProtocal = IsPcieProtocol(templateResource.channels);
    isDmaRead_ = isPcieProtocal;
    HCCL_DEBUG("[InsTempAllGatherRingPipeline] Use Dma Read[%d]", isDmaRead_);

    CHK_RET(LocalDataCopy(templateResource.threads));
    CHK_RET(RunRingPipelineAllGather(templateResource.threads, templateResource.channels));
    CHK_RET(PostLocalCopy(templateResource.threads));

    HCCL_INFO("[InsTempAllGatherRingPipeline] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRingPipeline::RunRingPipelineAllGather(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherRingPipeline] RunRingPipelineAllGather start, rankSize[%u]", templateRankSize_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    HCCL_DEBUG("[InsTempAllGatherRingPipeline] myRank[%u], myAlgRank[%u], prevRank[%u], nextRank[%u]",
               myRank_, myAlgRank, prevRank_, nextRank_);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    u64 chunkSize = sliceSize / templateRankSize_;

    CHK_PRT_RET(channels.count(prevRank_) == 0 || channels.at(prevRank_).empty(),
                HCCL_ERROR("[InsTempAllGatherRingPipeline] prevRank[%u] channel not found", prevRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(channels.count(nextRank_) == 0 || channels.at(nextRank_).empty(),
                HCCL_ERROR("[InsTempAllGatherRingPipeline] nextRank[%u] channel not found", nextRank_),
                HcclResult::HCCL_E_INTERNAL);

    const ChannelInfo &channelFromPrev = channels.at(prevRank_)[0];
    const ChannelInfo &channelToNext = channels.at(nextRank_)[0];

    void *recvCclBuffAddr = channelFromPrev.remoteCclMem.addr;
    void *sendCclBuffAddr = channelToNext.remoteCclMem.addr;

    HCCL_DEBUG("[InsTempAllGatherRingPipeline] channelFromPrev ok, channelToNext ok");

    // ========================================================================
    // Ring AllGather 接力传递算法
    //
    // 以 4 rank 为例，初始 scratch 状态:
    //   R0: [0, _, _, _]  R1: [_, 1, _, _]  R2: [_, _, 2, _]  R3: [_, _, _, 3]
    //
    // step=1: 每个 rank 把自己的 chunk 发给后继
    //   R0→R1: [0]  R1→R2: [1]  R2→R3: [2]  R3→R0: [3]
    //   结果: R0:[0,_,_,3]  R1:[0,1,_,_]  R2:[_,1,2,_]  R3:[_,_,2,3]
    //
    // step=2: 每个 rank 把 step=1 收到的 chunk 发给后继
    //   R0→R1: [3]  R1→R2: [0]  R2→R3: [1]  R3→R0: [2]
    //   结果: R0:[0,_,2,3]  R1:[0,1,_,3]  R2:[0,1,2,_]  R3:[_,1,2,3]
    //
    // step=3: 每个 rank 把 step=2 收到的 chunk 发给后继
    //   R0→R1: [2]  R1→R2: [3]  R2→R3: [0]  R3→R0: [1]
    //   结果: R0:[0,1,2,3]  R1:[0,1,2,3]  R2:[0,1,2,3]  R3:[0,1,2,3] ✅
    //
    // 规律: step=s 时，rank=i 发送的 chunkIdx = (i - s + 1 + N) % N
    //                        接收的 chunkIdx = (i - s + N) % N
    // ========================================================================

    for (u32 step = 1; step < templateRankSize_; step++) {
        // 这一步我要发送的 chunkIdx
        u32 sendChunkRankIdx = (myAlgRank + templateRankSize_ - step + 1) % templateRankSize_;
        // 这一步我要接收的 chunkIdx（即前驱这一步发送的 chunkIdx）
        u32 recvChunkRankIdx = (myAlgRank + templateRankSize_ - step) % templateRankSize_;

        u64 sendScratchOffset = sendChunkRankIdx * chunkSize;
        u64 recvScratchOffset = recvChunkRankIdx * chunkSize;

        bool isTailSend = (sendChunkRankIdx == templateRankSize_ - 1);
        bool isTailRecv = (recvChunkRankIdx == templateRankSize_ - 1);
        u64 sendChunkSize = (isTailSend && tempAlgParams_.tailSize != 0) ? tempAlgParams_.tailSize : chunkSize;
        u64 recvChunkSize = (isTailRecv && tempAlgParams_.tailSize != 0) ? tempAlgParams_.tailSize : chunkSize;

        u64 sendCount = sendChunkSize / dataTypeSize;
        u64 recvCount = recvChunkSize / dataTypeSize;

        HCCL_DEBUG("[InsTempAllGatherRingPipeline] step[%u], sendChunk[%u], recvChunk[%u]",
                   step, sendChunkRankIdx, recvChunkRankIdx);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        txSrcSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, sendScratchOffset,
                                  sendChunkSize, sendCount);
        txDstSlices.emplace_back(sendCclBuffAddr, sendScratchOffset, sendChunkSize, sendCount);

        rxSrcSlices.emplace_back(recvCclBuffAddr, recvScratchOffset, recvChunkSize, recvCount);
        rxDstSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, recvScratchOffset,
                                  recvChunkSize, recvCount);

        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channelToNext, channelFromPrev);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (isDmaRead_) {
            CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[0]),
                        HCCL_ERROR("[InsTempAllGatherRingPipeline] SendRecvRead failed at step[%u]", step),
                        HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]),
                        HCCL_ERROR("[InsTempAllGatherRingPipeline] SendRecvWrite failed at step[%u]", step),
                        HcclResult::HCCL_E_INTERNAL);
        }
    }

    HCCL_INFO("[InsTempAllGatherRingPipeline] RunRingPipelineAllGather end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRingPipeline::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        sliceSize = tempAlgParams_.tailSize;
    }

    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchBaseOff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.sliceSize * templateRankSize_;

        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 scOff = myAlgRank * tempAlgParams_.sliceSize + scratchBaseOff;

        if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr && inOff == scOff) {
            continue;
        }

        u64 sliceCount = sliceSize / dataTypeSize;
        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, sliceSize, sliceCount);

        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherRingPipeline::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (tempAlgParams_.buffInfo.outputPtr == tempAlgParams_.buffInfo.hcclBuff.addr) {
        HCCL_INFO("[InsTempAllGatherRingPipeline] PostLocalCopy skip, output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.sliceSize * templateRankSize_;

        for (u32 algRank = 0; algRank < templateRankSize_; algRank++) {
            u64 curSliceSize = sliceSize;
            if (tempAlgParams_.tailSize != 0 && algRank == templateRankSize_ - 1) {
                curSliceSize = tempAlgParams_.tailSize;
            }

            u64 scratchOffset = algRank * tempAlgParams_.sliceSize + scratchBase;
            u64 outOffset = algRank * tempAlgParams_.outputSliceStride + outBaseOff;
            u64 sliceCount = curSliceSize / dataTypeSize;

            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, curSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, curSliceSize, sliceCount);

            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherRingPipeline::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
}

void InsTempAllGatherRingPipeline::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
}

u32 InsTempAllGatherRingPipeline::GetRankFromMap(const u32 algRankIdx) const
{
    return subCommRanks_[0].at(algRankIdx);
}

} // namespace ops_hccl
