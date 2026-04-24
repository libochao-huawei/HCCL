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
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    channelsPerRank_ = (dieNum == 1) ? 1 : CalcChannelsPerRank(level1Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherNHR::GetRes(AlgResourceRequest &resourceRequest) const
{
    // NHR算法主需要一条主流
    resourceRequest.slaveThreadNum = channelsPerRank_ - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;
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

HcclResult InsTempAllGatherNHR::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                          TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHR] Run start");
    channelsPerRank_ = templateResource.channels.begin()->second.size();
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherNHR] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    CHK_PRT_RET(threadNum_ != templateResource.threads.size(),
                HCCL_ERROR("[InsTempAllGatherNHR] Rank [%d], requiredQueNum [%u] not equals templateQueNum [%zu].",
                           myRank_, threadNum_, templateResource.threads.size()),
                HcclResult::HCCL_E_INTERNAL);

    CHK_RET(LocalDataCopy(templateResource.threads));  // input buffer拷贝到scratch buffer上
    PreSync(templateResource.threads);

    CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels));

    PostSync(templateResource.threads);
    CHK_RET(PostLocalCopy(templateResource.threads));  // 从scratch buffer拷贝到output buffer上

    HCCL_INFO("[InsTempAllGatherNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}
HcclResult InsTempAllGatherNHR::RunAllGatherNHR(const std::vector<ThreadHandle> &threads,
                                                const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_);  // NHR 通信步数， celi(log2(rankSize))
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));  // 计算当前step要通信的卡，数据

            std::vector<ChannelInfo> channelRecvs = channels.at(GetRankFromMap(stepInfo.fromRank));
            std::vector<ChannelInfo> channelSends = channels.at(GetRankFromMap(stepInfo.toRank));
            if (recvChannels.size()!=sendChannels.size()) {
                HCCL_ERROR("[InsTempAllGatherNHR][RunAllGatherNHR] channelRecvs.size()!=channelSends.size()");
                return HcclResult::HCCL_E_INTERNAL;
            }
            // const ChannelInfo &channelRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[0];
            // const ChannelInfo &channelSend = channels.at(GetRankFromMap(stepInfo.toRank))[0];


            HCCL_DEBUG(
                "[InsTempAllGatherNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
                myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

            // 根据channel数据切分数据
            std::vector<float> dataSplitRate(channelRecvs.size());
            CalcDataSplitRateForLinks(channelRecvs, dataSplitRate);
            for (u32 channelId = 0; channelId < channelRecvs.size(); channelId++) {
                // 构造SendRecv， 都是Scratch到Scratch的传输，没有DMA消减
                std::vector<DataSlice> txSrcSlices;
                std::vector<DataSlice> txDstSlices;
                std::vector<DataSlice> rxSrcSlices;
                std::vector<DataSlice> rxDstSlices;
                void *sendCclBuffAddr = channelSends[channelId].remoteCclMem.addr;
                void *recvCclBuffAddr = channelRecvs[channelId].remoteCclMem.addr;

                for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                    const u32 txIdx = stepInfo.txSliceIdxs[i];
                    const u32 rxIdx = stepInfo.rxSliceIdxs[i];
                    const u64 txScratchOff = scratchBase + tempAlgParams_.sliceSize * txIdx;
                    const u64 rxScratchOff = scratchBase + tempAlgParams_.sliceSize * rxIdx;

                    const u64 txSliceSize = (txIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ? tempAlgParams_.tailSize: tempAlgParams_.sliceSize;
                    const u64 rxSliceSize = (rxIdx == templateRankSize_ - 1 && tempAlgParams_.tailSize != 0) ? tempAlgParams_.tailSize: tempAlgParams_.sliceSize;


                    DataSlice txSrcSliceTmp = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, txScratchOff, txSliceSize, txSliceSize / dataTypeSize);
                    DataSlice txDstSliceTmp = DataSlice(sendCclBuffAddr, txScratchOff, txSliceSize, txSliceSize / dataTypeSize);
                    DataSlice rxSrcSliceTmp = DataSlice(recvCclBuffAddr, rxScratchOff, rxSliceSize, rxSliceSize / dataTypeSize);
                    DataSlice rxDstSliceTmp = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff, rxSliceSize, rxSliceSize / dataTypeSize);

                    txSrcSlices.emplace_back(CalcDataSliceForLinks(txSrcSliceTmp, dataSplitRate, channelId, dataType_));
                    txDstSlices.emplace_back(CalcDataSliceForLinks(txDstSliceTmp, dataSplitRate, channelId, dataType_));
                    rxSrcSlices.emplace_back(CalcDataSliceForLinks(rxSrcSliceTmp, dataSplitRate, channelId, dataType_));
                    rxDstSlices.emplace_back(CalcDataSliceForLinks(rxDstSliceTmp, dataSplitRate, channelId, dataType_));
                }
                // write模式使用tx, rx地址不生效，仅使用对端link做Post/Wait
                // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
                TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
                TxRxChannels sendRecvChannels(channelRecvs[channelId], channelRecvs[channelId]);
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

                CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]),
                            HCCL_ERROR("[InsTempAllGatherNHR] sendrecv failed (step=%u, rpt=%u)", step, rpt),
                            HcclResult::HCCL_E_INTERNAL);
            }
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

HcclResult InsTempAllGatherNHR::LocalDataCopy(const std::vector<ThreadHandle> &threads)

{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 尾块模式
    if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
        sliceSize = tempAlgParams_.tailSize;
    }
    for (u64 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 scOff = tempAlgParams_.sliceSize * myAlgRank + scratchBaseoff;
        if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr && inOff == scOff) {
            continue;
        }
        u64 sliceCount = sliceSize / dataTypeSize;
        DataSlice srcSlices(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, sliceSize, sliceCount);
        LocalCopy(threads[0], srcSlices, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHR::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (tempAlgParams_.buffInfo.outputPtr == tempAlgParams_.buffInfo.hcclBuff.addr) {
        HCCL_INFO("[InsTempAllGatherNHR] PostLocalCopy skip because output is scratch" );
        return HcclResult::HCCL_SUCCESS;
    }
    u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
                        // 尾块模式
            if (tempAlgParams_.tailSize !=0 && algRank == templateRankSize_ -1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 sliceCount = sliceSize / dataTypeSize;
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
 
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    notifyIdxMainToSub.assign(slaveThreadNum, 0);  // 从线程全部用第0个Notify等待主线程的同步信号
}
 
void InsTempAllGatherNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
 
    notifyIdxSubToMain.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < slaveThreadNum; ++notifyIdx) {
        notifyIdxSubToMain.emplace_back(notifyIdx);
    }
}

HcclResult InsTempAllGatherNHR::PreSync(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slaveThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads.at(0), slaveThreads, notifyIdxMainToSub_));
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherNHR::PostSync(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slaveThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads.at(0), slaveThreads, notifyIdxSubToMain_));
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace Hccl