/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_clos_opt_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {
namespace {
constexpr u32 COPY_TO_COMM_NOTIFY_IDX = 1;

HcclResult CheckRemoteOutputRange(const char *tag, u32 myRank, u32 peer, const ChannelInfo &linkRemote,
                                  u64 offset, u64 size)
{
    CHK_PRT_RET(linkRemote.remoteOutputGraphMode.addr == nullptr,
                HCCL_ERROR("[%s] Rank[%u] peer[%u] remote output unavailable.", tag, myRank, peer),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(offset + size > linkRemote.remoteOutputGraphMode.size,
                HCCL_ERROR("[%s] Rank[%u] peer[%u] remote output range overflow. off[%llu] size[%llu] "
                           "remoteOutputSize[%llu]",
                           tag, myRank, peer, offset, size, linkRemote.remoteOutputGraphMode.size),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
}

InsTempAllGatherMeshClosOptNoMemcpy::InsTempAllGatherMeshClosOptNoMemcpy(const OpParam &param, const u32 rankId,
                                                       const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1DOptNoMemcpy(param, rankId, subCommRanks)
{
}

InsTempAllGatherMeshClosOptNoMemcpy::~InsTempAllGatherMeshClosOptNoMemcpy() {}

u64 InsTempAllGatherMeshClosOptNoMemcpy::GetThreadNum() const
{
    return channelsPerRank_;
}

HcclResult InsTempAllGatherMeshClosOptNoMemcpy::GetRes(AlgResourceRequest &resourceRequest) const
{   
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, COPY_TO_COMM_NOTIFY_IDX + 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOptNoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestMesh1DByTopo(comm, param, topoInfo, subCommRanks_, levelChannels,
                                           CommTopo::COMM_TOPO_CLOS));
    CHK_PRT_RET(levelChannels.empty(),
                HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy][CalcRes] Rank[%u] CLOS channel request is empty.",
                           topoInfo->userRank),
                HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());
    std::map<u32, u32> peerChannelCount;
    for (const auto &channel : levelChannels) {
        peerChannelCount[channel.remoteRank]++;
    }
    for (const auto &peerInfo : peerChannelCount) {
        HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][CalcRes] Rank[%u] peer[%u] channelCount[%u]",
                     topoInfo->userRank, peerInfo.first, peerInfo.second);
    }
    // if (channelsPerRank_ != 4) {
    //     HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy][CalcRes] Rank[%u] channelsPerRank[%u] is not expected clos link num 4.",
    //                topoInfo->userRank, channelsPerRank_);
    //     return HcclResult::HCCL_E_INTERNAL;
    // }

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOptNoMemcpy::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                                  TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherMeshClosOptNoMemcpy] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    if (templateRankSize_ == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(templateResource.threads.size() < GetThreadNum(),
                HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy] Rank[%u] threads[%zu] < required[%llu].",
                           myRank_, templateResource.threads.size(), GetThreadNum()),
                HcclResult::HCCL_E_INTERNAL);

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
    HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOptNoMemcpy::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    CHK_PRT_RET(threads.size() < channelsPerRank_,
                HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherMesh] Rank[%d] threads[%zu] < required[%u].",
                           myRank_, threads.size(), channelsPerRank_),
                HcclResult::HCCL_E_INTERNAL);
    std::vector<ThreadHandle> commThreads(threads.begin(), threads.begin() + channelsPerRank_);
    HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherMesh] Rank[%d] templateRankSize[%u] totalLinks[%u].",
              myRank_, templateRankSize_, channelsPerRank_);
    // ========== 新增日志 ==========
    HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherMesh] Rank[%d] remoteWrite[%d] threads.size[%zu] "
                 "channels.size[%zu] repeatNum0[%u] repeatNum1[%u] allGatherToAllRanksCounts[%llu] "
                 "inputSize0[%llu] inputSize1[%llu] outStride0[%llu] outStride1[%llu]",
                 myRank_, remoteWrite, commThreads.size(), channels.size(), tempAlgParams_.repeatNum,
                 tempAlgParams1_.repeatNum, allGatherToAllRanksCounts,
                 tempAlgParams_.buffInfo.inputSize, tempAlgParams1_.buffInfo.inputSize,
                 tempAlgParams_.outputSliceStride, tempAlgParams1_.outputSliceStride);
    if (commThreads.size() != channelsPerRank_) {
        HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherMesh] Rank[%d] commThreads.size[%zu] != "
                     "channelsPerRank[%u].",
                     myRank_, commThreads.size(), channelsPerRank_);
    }
    for (const auto &peerInfo : channels) {
        HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherMesh] Rank[%d] channelMap peer[%u] channels[%zu]",
                     myRank_, peerInfo.first, peerInfo.second.size());
    }
    // ============================
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

    if (remoteWrite) {
        allGatherToAllRanksCounts = 0;
    }

    if (allGatherToAllRanksCounts == 0 && tempAlgParams1_.repeatNum > 0) {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        u64 sliceSize = tempAlgParams1_.buffInfo.inputSize;
        u64 sliceCount = sliceSize / dataTypeSize;
        u64 inputOffset = tempAlgParams1_.buffInfo.inBuffBaseOff;
        u64 outputOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + myRank_ * tempAlgParams1_.outputSliceStride;
        DataSlice srcSlice(tempAlgParams1_.buffInfo.inputPtr, inputOffset, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams1_.buffInfo.outputPtr, outputOffset, sliceSize, sliceCount);
        CHK_RET(LocalCopy(commThreads[0], srcSlice, dstSlice));
    }

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        if (remoteWrite) {
            u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
            u64 sliceCount = sliceSize / dataTypeSize;
            u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
            u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * tempAlgParams_.outputSliceStride;
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(commThreads[0], srcSlice, dstSlice));
            break;
        } else {
            HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy][LOCAL_READ] Rank[%u] skip HCCL->output copy, rpt[%u].",
                      myRank_, rpt);
        }
    }
    if (commThreads.size() > 1) {
        std::vector<ThreadHandle> subCommThreads(commThreads.begin() + 1, commThreads.end());
        CHK_RET(PreSyncInterThreads(commThreads[0], subCommThreads, std::vector<u32>(subCommThreads.size(), COPY_TO_COMM_NOTIFY_IDX)));
    }
    
    for (u32 linkIdx = 0; linkIdx < commThreads.size() - 1; linkIdx++) {
        CHK_RET(RunAllGatherOnLink(commThreads, channels, linkIdx));
    }
    CHK_RET(RunAllGatherToAllRanks(commThreads, channels, allGatherToAllRanksCounts,
                                   allGatherToAllRanksCounts + tempAlgParams1_.repeatNum));
     allGatherToAllRanksCounts += tempAlgParams1_.repeatNum;

    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOptNoMemcpy::RunAllGatherToAllRanks(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels, u32 startStep, u32 endStep)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myRank_ + 1 + neighborIdx) % subCommRanks_[0].size()];

        u32 step = (connectedRank ^ myRank_) % rankSize_;
        if (step < startStep || step >= endStep) {
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 linkIdx = totalLinksToNeighbor - 1;
        CHK_PRT_RET(linkIdx >= it->second.size() || threads.empty(),
                    HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherToAllRanks] Rank[%d] invalid link/thread. "
                               "connectedRank[%u] linkIdx[%u] peerChannels[%zu] threads[%zu]",
                               myRank_, connectedRank, linkIdx, it->second.size(), threads.size()),
                    HcclResult::HCCL_E_INTERNAL);


        HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy] Rank[%d] linkIdx[%u] matched connectedRank[%u] "
                  "totalLinks[%u] totalThreads[%u] enableRemoteMemAccess[%d]",
                  myRank_, linkIdx, connectedRank, totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_);

        const ChannelInfo &linkRemote = it->second[linkIdx];

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        
        u64 sliceSize = tempAlgParams1_.buffInfo.inputSize;
        u64 sliceCount = sliceSize / dataTypeSize;
        u64 outputSliceStride = tempAlgParams1_.outputSliceStride;
        
        // 远端写 不应该启动
        void *txSrcPtr = tempAlgParams1_.buffInfo.inputPtr;
        u64 txSrcOffset = tempAlgParams1_.buffInfo.inBuffBaseOff;
        txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

        void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 txDstOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + myRank_ * outputSliceStride;
        CHK_RET(CheckRemoteOutputRange("InsTempAllGatherMeshClosOptNoMemcpy[C]", myRank_, connectedRank,
                                       linkRemote, txDstOffset, sliceSize));
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);

        // no-memcpy mode下C直接写到对端output；rx slice仅用于SendRecvInfo占位。
        void *rxSrcPtr = linkRemote.remoteOutputGraphMode.addr;
        u64 rxSrcOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
        
        void *rxDstPtr = tempAlgParams1_.buffInfo.outputPtr;
        u64 rxOutOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
        u32 threadIdx = threads.size() - 1;
        HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][C_PRE] Rank[%d] step[%u] peer[%u] linkIdx[%u] threadIdx[%u] "
                     "peerChannels[%zu] txDstOff[%llu] rxSrcOff[%llu] rxDstOff[%llu] size[%llu]",
                     myRank_, step, connectedRank, linkIdx, threadIdx, it->second.size(), txDstOffset,
                     rxSrcOffset, rxOutOffset, sliceSize);
        HcclResult ret = SendRecvWrite(sendRecvInfo, threads[threadIdx]);
        HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][C_POST] Rank[%d] step[%u] peer[%u] linkIdx[%u] threadIdx[%u] ret[%d]",
                     myRank_, step, connectedRank, linkIdx, threadIdx, ret);
        CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAllGatherMesh1DOptNoMemcpy] RunAllGather SendRecvWrite failed"),
                    HcclResult::HCCL_E_INTERNAL);
                    
    }
    return HCCL_SUCCESS;

}
HcclResult InsTempAllGatherMeshClosOptNoMemcpy::RunAllGatherOnLink(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    u32 linkIdx)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myRank_ + 1 + neighborIdx) % subCommRanks_[0].size()];

        if ((connectedRank % meshSize_) != (myRank_ % meshSize_)) {
            continue; // 只处理同一clos内的通信
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myRank_ ^ connectedRank) % (threads.size() - 1);
        CHK_PRT_RET(linkIdx >= it->second.size(),
                    HCCL_ERROR("[InsTempAllGatherMeshClosOptNoMemcpy][RunAllGatherOnLink] Rank[%d] invalid link. "
                               "connectedRank[%u] linkIdx[%u] peerChannels[%zu] threads[%zu]",
                               myRank_, connectedRank, linkIdx, it->second.size(), threads.size()),
                    HcclResult::HCCL_E_INTERNAL);

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_INFO("[InsTempAllGatherMeshClosOptNoMemcpy] Rank[%d] linkIdx[%u] matched connectedRank[%u] "
                  "selectedLinkIdx[%u] totalLinks[%u] totalThreads[%u] enableRemoteMemAccess[%d]",
                  myRank_, linkIdx, connectedRank, selectedLinkIdx, totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_);

        const ChannelInfo &linkRemote = it->second[linkIdx];
        
        u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
        u64 sliceCount = sliceSize / dataTypeSize;
        u64 outputSliceStride = tempAlgParams_.outputSliceStride;

        if (remoteWrite) {
            std::vector<DataSlice> txSrcSlicesAll;
            std::vector<DataSlice> txDstSlicesAll;
            std::vector<DataSlice> rxDstSlicesAll;
            std::vector<DataSlice> rxSrcSlicesAll;
            
            // 阶段1 远端写， 从本地的 input ptr 到 远端的 hccl buffer
            // 需要确保 当前场景下配置的repeatNum 为 1

            // tx 远端写
            void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
            u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

            void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
            u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * outputSliceStride;
            CHK_RET(CheckRemoteOutputRange("InsTempAllGatherMeshClosOptNoMemcpy[B]", myRank_, connectedRank,
                                           linkRemote, txDstOffset, sliceSize));
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);

            // rx 远端读，不应该启动
            void *rxSrcPtr = linkRemote.remoteOutputGraphMode.addr;
            u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
            
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

            TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
            HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][B_WRITE_PRE] Rank[%d] peer[%u] linkIdx[%u] "
                         "peerChannels[%zu] txSrcOff[%llu] txDstOff[%llu] size[%llu]",
                         myRank_, connectedRank, linkIdx, it->second.size(), txSrcOffset, txDstOffset, sliceSize);
            HcclResult ret = SendRecvWrite(sendRecvInfo, threads[linkIdx]);
            HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][B_WRITE_POST] Rank[%d] peer[%u] linkIdx[%u] ret[%d]",
                         myRank_, connectedRank, linkIdx, ret);
            CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAllGatherMesh1DOptNoMemcpy] RunAllGather SendRecvWrite failed"),
                        HcclResult::HCCL_E_INTERNAL);

        } else {
            // 阶段 2 远端写， 将本地 output 中对端需要的数据直写到对端 output
            std::vector<DataSlice> txSrcSlicesAll;
            std::vector<DataSlice> txDstSlicesAll;
            std::vector<DataSlice> rxDstSlicesAll;
            std::vector<DataSlice> rxSrcSlicesAll;

            for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
                u32 localMeshRank = (myRank_ + rpt) % meshSize_;
                u32 txRank = localMeshRank + myRank_ / meshSize_ * meshSize_;
                void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
                u64 txSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + txRank * outputSliceStride;
                txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

                void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
                u64 txDstOffset = tempAlgParams_.buffInfo.outBuffBaseOff + txRank * outputSliceStride;
                CHK_RET(CheckRemoteOutputRange("InsTempAllGatherMeshClosOptNoMemcpy[B]", myRank_, connectedRank,
                                               linkRemote, txDstOffset, sliceSize));
                txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);


                // rx slice用于描述对端会写回本端的数据槽。
                u32 remoteMeshRank = (connectedRank + rpt) % meshSize_;
                u32 remoteRank = remoteMeshRank + connectedRank / meshSize_ * meshSize_;
                void *rxSrcPtr = linkRemote.remoteOutputGraphMode.addr;
                u64 rxSrcOffset = tempAlgParams_.buffInfo.outBuffBaseOff + remoteRank * outputSliceStride;
                rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
                
                void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
                u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + remoteRank * outputSliceStride;
                rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

                HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][B_WRITE_PRE] Rank[%d] peer[%u] rpt[%u] remoteRank[%u] "
                             "txRank[%u] linkIdx[%u] peerChannels[%zu] txSrcOff[%llu] txDstOff[%llu] "
                             "rxSrcOff[%llu] rxDstOff[%llu] size[%llu]",
                             myRank_, connectedRank, rpt, remoteRank, txRank, linkIdx, it->second.size(),
                             txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset, sliceSize);
            }
            TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
            HcclResult ret = SendRecvWrite(sendRecvInfo, threads[linkIdx]);
            HCCL_WARNING("[InsTempAllGatherMeshClosOptNoMemcpy][B_WRITE_POST] Rank[%d] peer[%u] repeatNum[%u] "
                         "linkIdx[%u] ret[%d]",
                         myRank_, connectedRank, tempAlgParams_.repeatNum, linkIdx, ret);
            CHK_PRT_RET(ret, HCCL_ERROR("[InsTempAllGatherMesh1DOptNoMemcpy] RunAllGather SendRecvWrite failed"),
                        HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
