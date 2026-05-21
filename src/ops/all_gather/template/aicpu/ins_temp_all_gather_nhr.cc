/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
InsTempAllGatherNHR::InsTempAllGatherNHR(const OpParam &param, const u32 rankId,
                                         const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAllGatherNHR::~InsTempAllGatherNHR() {}

HcclResult InsTempAllGatherNHR::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                        AlgResourceRequest &resourceRequest)
{
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    channelsPerRank_ = CalcChannelsPerRank(level1Channels);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherNHR::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}
u64 InsTempAllGatherNHR::GetThreadNum() const
{
    return channelsPerRank_;
}

u64 InsTempAllGatherNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempAllGatherNHR::PreprareDataSplitForMultiChannel(const TemplateResource &templateResource) {
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalDataCount = tempAlgParams_.sliceSize / dataTypeSize;
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(totalDataCount, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplit_, dataOffset_));
    if (tempAlgParams_.tailSize > 0) {
        u64 totalDataCountTail = tempAlgParams_.tailSize / dataTypeSize;
        CHK_RET(CalcDataSplitByPortGroup(totalDataCountTail, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplitTail_, dataOffsetTail_));
    }
    
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHR::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                          TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHR] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherNHR] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = GetThreadNum();
    if (templateResource.threads.size() < threadNum_)
    {
        HCCL_ERROR("[InsTempAllGatherNHR] Rank [%d], thread num[%u] is not as expected[%u].", myRank_, templateResource.threads.size(), threadNum_);
        return HcclResult::HCCL_E_INTERNAL;
    }
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;

    bool isPcieProtocal = IsPcieProtocol(templateResource.channels);  // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocal;  // 是否使用Read模式
    HCCL_DEBUG("[InsTempAllGatherNHR] Use Dma Read[%d]", isDmaRead_);
    CHK_RET(PreprareDataSplitForMultiChannel(templateResource));
    sendLocalSliceFromOutput_ = CanSendLocalSliceFromOutput();
    HCCL_DEBUG("[InsTempAllGatherNHR] Send local slice from output[%d]", sendLocalSliceFromOutput_);

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(LocalDataCopy(templateResource.threads, channelIdx));   // input buffer拷贝到scratch buffer上
        CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels, channelIdx));
        CHK_RET(PostLocalCopy(templateResource.threads, channelIdx));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAllGatherNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

bool InsTempAllGatherNHR::CanSendLocalSliceFromOutput() const
{
    return !isDmaRead_ && tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr &&
           tempAlgParams_.buffInfo.inBuffType == BufferType::OUTPUT &&
           tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT &&
           tempAlgParams_.buffInfo.inBuffBaseOff == tempAlgParams_.buffInfo.outBuffBaseOff &&
           tempAlgParams_.inputSliceStride == tempAlgParams_.outputSliceStride &&
           tempAlgParams_.inputRepeatStride == tempAlgParams_.outputRepeatStride;
}

bool InsTempAllGatherNHR::IsLastStepReadSlice(u32 algRank) const
{
    for (u32 readSliceIdx : lastStepReadSliceIdxs_) {
        if (readSliceIdx == algRank) {
            return true;
        }
    }
    return false;
}

HcclResult InsTempAllGatherNHR::RunLastStepWriteThenRead(const std::vector<ThreadHandle> &threads,
                                                         const ChannelInfo &channelSend,
                                                         const ChannelInfo &channelRecv,
                                                         const AicpuNHRStepInfo &stepInfo,
                                                         const u32 &channelIdx, u32 step)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    void *sendCclBuffAddr = channelSend.remoteCclMem.addr;
    void *recvCclBuffAddr = channelRecv.remoteCclMem.addr;

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            const u32 txIdx = stepInfo.txSliceIdxs[i];
            const u32 rxIdx = stepInfo.rxSliceIdxs[i];
            const u64 txPartialOffset = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
            const u64 rxPartialOffset = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
            const u64 txScratchOff = scratchBase + tempAlgParams_.sliceSize * txIdx + txPartialOffset;
            const u64 rxScratchOff = scratchBase + tempAlgParams_.sliceSize * rxIdx + rxPartialOffset;

            const u64 txSliceSize = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
            const u64 rxSliceSize = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ?
                dataSplitTail_[channelIdx]: dataSplit_[channelIdx];

            if (i == 0) {
                const u64 txOutputOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                        rpt * tempAlgParams_.outputRepeatStride +
                                        tempAlgParams_.outputSliceStride * txIdx + txPartialOffset;
                const bool sendFromOutput = txIdx == stepInfo.myRank;
                void *txSrcPtr = sendFromOutput ? tempAlgParams_.buffInfo.outputPtr : tempAlgParams_.buffInfo.hcclBuff.addr;
                const u64 txSrcOff = sendFromOutput ? txOutputOff : txScratchOff;
                txSrcSlices.emplace_back(txSrcPtr, txSrcOff, txSliceSize, txSliceSize / dataTypeSize);
                txDstSlices.emplace_back(sendCclBuffAddr, txScratchOff, txSliceSize, txSliceSize / dataTypeSize);
                continue;
            }

            const u64 rxOutputOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride +
                                    tempAlgParams_.outputSliceStride * rxIdx + rxPartialOffset;
            rxSrcSlices.emplace_back(recvCclBuffAddr, rxScratchOff, rxSliceSize, rxSliceSize / dataTypeSize);
            rxDstSlices.emplace_back(tempAlgParams_.buffInfo.outputPtr, rxOutputOff, rxSliceSize,
                rxSliceSize / dataTypeSize);
            if (rpt == 0) {
                lastStepReadSliceIdxs_.push_back(rxIdx);
            }
        }
    }

    TxRxChannels sendRecvChannels(channelSend, channelRecv);
    const std::vector<DataSlice> emptySlices;
    TxRxSlicesList writeSlicesList({txSrcSlices, txDstSlices}, {emptySlices, emptySlices});
    SendRecvInfo writeInfo(sendRecvChannels, writeSlicesList);
    CHK_PRT_RET(SendRecvWrite(writeInfo, threads[channelIdx]),
        HCCL_ERROR("[InsTempAllGatherNHR] last step write failed (step=%u)", step),
        HcclResult::HCCL_E_INTERNAL);

    if (rxSrcSlices.empty()) {
        return HcclResult::HCCL_SUCCESS;
    }

    TxRxSlicesList readSlicesList({emptySlices, emptySlices}, {rxSrcSlices, rxDstSlices});
    SendRecvInfo readInfo(sendRecvChannels, readSlicesList);
    CHK_PRT_RET(SendRecvRead(readInfo, threads[channelIdx]),
        HCCL_ERROR("[InsTempAllGatherNHR] last step read failed (step=%u)", step),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHR::RunAllGatherNHR(const std::vector<ThreadHandle> &threads,
                                                const std::map<u32, std::vector<ChannelInfo>> &channels, const u32 &channelIdx)
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_);  // NHR 通信步数， celi(log2(rankSize))
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    lastStepReadSliceIdxs_.clear();

    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));  // 计算当前step要通信的卡，数据

        const ChannelInfo &channelRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[channelIdx];
        const ChannelInfo &channelSend = channels.at(GetRankFromMap(stepInfo.toRank))[channelIdx];
        // 构造SendRecv， 都是Scratch到Scratch的传输，没有DMA消减
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        void *sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void *recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        HCCL_DEBUG(
            "[InsTempAllGatherNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        const bool readLastStepToOutput = sendLocalSliceFromOutput_ && step == nSteps - 1 && stepInfo.nSlices > 1;
        if (readLastStepToOutput) {
            CHK_RET(RunLastStepWriteThenRead(threads, channelSend, channelRecv, stepInfo, channelIdx, step));
            continue;
        }

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];
                const u64 txPartialOffset = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ? dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
                const u64 rxPartialOffset = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ? dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
                const u64 txScratchOff = scratchBase + tempAlgParams_.sliceSize * txIdx + txPartialOffset;
                const u64 rxScratchOff = scratchBase + tempAlgParams_.sliceSize * rxIdx + rxPartialOffset;

                const u64 txSliceSize = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ? dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
                const u64 rxSliceSize = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ? dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
                const bool sendFromOutput = sendLocalSliceFromOutput_ && txIdx == stepInfo.myRank;
                const u64 txOutputOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride +
                                        tempAlgParams_.outputSliceStride * txIdx + txPartialOffset;
                void *txSrcPtr = sendFromOutput ? tempAlgParams_.buffInfo.outputPtr : tempAlgParams_.buffInfo.hcclBuff.addr;
                const u64 txSrcOff = sendFromOutput ? txOutputOff : txScratchOff;

                txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOff, txSliceSize, txSliceSize / dataTypeSize);
                txDstSlicesAll.emplace_back(sendCclBuffAddr, txScratchOff, txSliceSize, txSliceSize / dataTypeSize);
                rxSrcSlicesAll.emplace_back(recvCclBuffAddr, rxScratchOff, rxSliceSize, rxSliceSize / dataTypeSize);
                rxDstSlicesAll.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff, rxSliceSize, rxSliceSize / dataTypeSize);
            }
        }
        // write模式使用tx, rx地址不生效，仅使用对端link做Post/Wait
        // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (isDmaRead_) {
            CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR("[InsTempAllGatherNHR] sendrecv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR("[InsTempAllGatherNHR] sendrecv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
u32 InsTempAllGatherNHR::GetRankFromMap(const u32 algRankIdx) const
{
    return subCommRanks_[0].at(algRankIdx);
}
HcclResult InsTempAllGatherNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;

    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (myAlgRank + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (myAlgRank + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量， NHR是一个传输数据变化的
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = myAlgRank;
    u32 rxSliceIdx = (myAlgRank - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[AllGatherNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHR::LocalDataCopy(const std::vector<ThreadHandle> &threads, const u32 &channelIdx)

{
    if (sendLocalSliceFromOutput_) {
        HCCL_DEBUG("[InsTempAllGatherNHR][LocalDataCopy] skip local output to ccl copy.");
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 partialSliceSize = dataSplit_[channelIdx];
    u64 partialOffset = dataOffset_[channelIdx];
    // 尾块模式
    if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
        partialSliceSize = dataSplitTail_[channelIdx];
        partialOffset = dataOffsetTail_[channelIdx];
    }
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff + partialOffset;
        const u64 scOff = tempAlgParams_.sliceSize * myAlgRank + scratchBaseoff + partialOffset;
        if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr && inOff == scOff) {
            continue;
        }
        u64 sliceCount = partialSliceSize / dataTypeSize;
        DataSlice srcSlices(tempAlgParams_.buffInfo.inputPtr, inOff, partialSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, partialSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[channelIdx], srcSlices, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHR::PostLocalCopy(const std::vector<ThreadHandle> &threads, const u32 &channelIdx)
{
    if (tempAlgParams_.buffInfo.outputPtr == tempAlgParams_.buffInfo.hcclBuff.addr) {
        HCCL_INFO("[InsTempAllGatherNHR] PostLocalCopy skip because output is scratch" );
        return HcclResult::HCCL_SUCCESS;
    }
    u64 partialSliceSize = dataSplit_[channelIdx];
    u64 partialOffset = dataOffset_[channelIdx];
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            if (sendLocalSliceFromOutput_ && algRank == myAlgRank) {
                continue;
            }
            if (sendLocalSliceFromOutput_ && IsLastStepReadSlice(algRank)) {
                continue;
            }
            // 尾块模式
            if (tempAlgParams_.tailSize !=0 && algRank == templateRankSize_ -1) {
                partialSliceSize = dataSplitTail_[channelIdx];
                partialOffset = dataOffsetTail_[channelIdx];
            }
            u64 sliceCount = partialSliceSize / dataTypeSize;
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase + partialOffset;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff + partialOffset;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, partialSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, partialSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[channelIdx], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
void InsTempAllGatherNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl
