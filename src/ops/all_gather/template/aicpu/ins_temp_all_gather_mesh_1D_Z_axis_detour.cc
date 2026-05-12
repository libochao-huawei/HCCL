/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
bool InsTempAllGatherMesh1D1DZAxisDetour::isNew;
InsTempAllGatherMesh1D1DZAxisDetour::InsTempAllGatherMesh1D1DZAxisDetour(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1D1DZAxisDetour::~InsTempAllGatherMesh1D1DZAxisDetour() {}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] start");
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    level0ChannelNumPerRank_ = CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = CalcChannelsPerRank(level1Channels);
    std::vector<HcclChannelDesc> mergedChannels;
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    HCCL_INFO("level0Channels[%d]level1Channels[%d]\n", level0Channels.size(), level1Channels.size());
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    HCCL_INFO("mergedChannels[%d]\n", mergedChannels.size());
    channelsSize = mergedChannels.size();

    if(subCommRanks_.size() <= COMM_LEVEL0) {
        return HCCL_E_PARA;
    }
    GetRes(resourceRequest);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D1DZAxisDetour::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][GetRes] threadNum[%u]", threadNum);
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]", templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

u64 InsTempAllGatherMesh1D1DZAxisDetour::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize ==0) {
        HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (opMode_ == OpMode::OPBASE) {
        CHK_RET(PostLocalCopy(templateResource.threads));
    }
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize,
    const std::vector<ChannelInfo> &channels,
    std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
    std::vector<u64> &elemOffset)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcDataSplitByPortGroup] Run Start[%u][%u][%f]\n", level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return CalcDataSplitByPortGroupZAxisDetour(totalDataCount, dataTypeSize, channels,
        elemCountOut, sizeOut, elemOffset,
        level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::SetchannelsPerRank(
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    if (channelsPerRank_ > 1) {
        level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
        level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
        level0DataRatio_ = 0.5f;
    }
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
              "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
              channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 当输入为hcclbuffer时，可以直接用read模式+dma消减，跳过后拷贝
    bool dmaRead = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER && tempAlgParams_.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
    auto& ranks = subCommRanks_[COMM_LEVEL0];

    std::vector<ChannelInfo> mergedChannels;
    for (u32 i = 0; i < ranks.size(); i++) {
        if (i == myAlgRank) {
            continue;
        }
        mergedChannels.insert(mergedChannels.end(), channels.at(i).begin(), channels.at(i).end());
    }
    
    u32 threadNum = mergedChannels.size();
    HCCL_DEBUG("threadNum %u\n", threadNum);

    for (u32 threadIdx = 0; threadIdx < threadNum; threadIdx++) {
            u32 connectedRank;
            connectedRank = mergedChannels[threadIdx].remoteRank;
            const std::vector<ChannelInfo> &curChannels = channels.at(connectedRank);
            channelsPerRank_ = curChannels.size();
            HCCL_INFO("[RunAllGatherMesh]channelsPerRank_[%u]\n", channelsPerRank_);
            u32 idx = (channelsPerRank_ == 0) ? 0 : (threadIdx % channelsPerRank_);

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                      myRank_, connectedRank, connectedAlgRank);
            
            u64 sliceSize = tempAlgParams_.sliceSize;
            HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour11] sliceSize[%u]\n", sliceSize);
            if (dmaRead) {
                if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                    sliceSize = tempAlgParams_.tailSize;
                    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour22] sliceSize[%u]\n", sliceSize);
                }
            } else if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
                sliceSize = tempAlgParams_.tailSize;
                HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour33] sliceSize[%u]\n", sliceSize);
            }
            u64 sliceCount = sliceSize / dataTypeSize;
            std::vector<u64> elemCountOut;
            std::vector<u64> sizeOut;
            std::vector<u64> elemOffset;
            
            HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour44] silceCount[%u]\n", sliceCount);
            CHK_RET(CalcDataSplitByPortGroup(sliceCount, dataTypeSize, curChannels, elemCountOut, sizeOut, elemOffset));

            CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 || 
                        channels.at(connectedRank).empty(),
                        HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][RankID]=%u threadIdx=%u, threads.size=%u, "
                                   "connectedRank=%d, channels.size=%u",
                                   myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                        HcclResult::HCCL_E_INTERNAL);

            const ChannelInfo &linkRemote = channels.at(connectedRank)[idx];
            void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

            std::vector<DataSlice> txSrcSlicesAll;
            std::vector<DataSlice> txDstSlicesAll;
            std::vector<DataSlice> rxDstSlicesAll;
            std::vector<DataSlice> rxSrcSlicesAll;

            for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
                const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
                const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
                const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

                u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff + elemOffset[idx];
                u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank + elemOffset[idx];
                u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

                u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff + elemOffset[idx];
                u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank + elemOffset[idx];
                u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

                void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
                void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

                txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, sizeOut[idx], elemCountOut[idx]);
                txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sizeOut[idx], elemCountOut[idx]);
                rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sizeOut[idx], elemCountOut[idx]);
                rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sizeOut[idx], elemCountOut[idx]);

                HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txSrcSlices: "
                           "offset[%d] sliceSize[%d] count[%d].",
                           myRank_, connectedRank, rpt, txOutOffset, sizeOut[idx], elemCountOut[idx]);

                HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txDstSlices: "
                           "offset[%d] sliceSize[%d] count[%d].",
                           myRank_, connectedRank, rpt, txDstOffset, sizeOut[idx], elemCountOut[idx]);

                HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxSrcSlices: "
                           "offset[%d] sliceSize[%d] count[%d].",
                           myRank_, connectedRank, rpt, rxOutOffset, sizeOut[idx], elemCountOut[idx]);

                HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxDrcSlices: "
                           "offset[%d] sliceSize[%d] count[%d].",
                           myRank_, connectedRank, rpt, rxSrcOffset, sizeOut[idx], elemCountOut[idx]);
            }

            TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            if (dmaRead){
                CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                            HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
            } else {
                CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                            HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
            }

        }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] LocalDataCopy.");

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    // 尾块模式
    if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
        sliceSize = tempAlgParams_.tailSize;
    }
    u64 sliceCount = sliceSize / dataTypeSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
        if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff) {
            continue;
        }
        HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d] "
                   "sliceSize[%d] count[%d].",
                   myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
        HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
                   "outOff[%d] sliceSize[%d] count[%d].",
                   myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);

        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] PostLocalCopy.");
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] PostLocalCopy skip because output is scratch" );
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] PostLocalCopy skip because input is scratch and should be read to output" );
        return HcclResult::HCCL_SUCCESS;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            // 尾块模式
            if (tempAlgParams_.tailSize !=0 && algRank == templateRankSize_ -1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
            u64 sliceCount = sliceSize / dataTypeSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                       myRank_, rank, algRank, outBaseOff, outBaseOff, scratchOffset, outOffset);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1D1DZAxisDetour::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1D1DZAxisDetour::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl