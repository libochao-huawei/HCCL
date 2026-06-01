/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_clos_opt.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"

namespace ops_hccl {

InsTempAllGatherMeshClosOpt::InsTempAllGatherMeshClosOpt(const OpParam &param, const u32 rankId,
                                                       const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1DOpt(param, rankId, subCommRanks)
{
}

InsTempAllGatherMeshClosOpt::~InsTempAllGatherMeshClosOpt() {}

u64 InsTempAllGatherMeshClosOpt::GetThreadNum() const
{
    return channelsPerRank_;
    // 最多使用 min(链路数, 邻居数) 个线程
    // u32 numNeighbors = std::max(1u, templateRankSize_ - 1);
    // return std::min(channelsPerRank_, numNeighbors);
}

HcclResult InsTempAllGatherMeshClosOpt::GetRes(AlgResourceRequest &resourceRequest) const
{   
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOpt::CalcRes(HcclComm comm, const OpParam &param,
                                               const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMeshClosOpt][CalcRes] start");
    std::vector<HcclChannelDesc> levelChannels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, levelChannels));
    resourceRequest.channels.push_back(levelChannels);

    channelsPerRank_ = levelChannels.empty() ? 1 : CalcChannelsPerRank(levelChannels);
    HCCL_INFO("[InsTempAllGatherMeshClosOpt][CalcRes] totalLinks[%u], channelCount[%zu]",
              channelsPerRank_, levelChannels.size());

    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOpt::RunAllGatherMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMeshClosOpt][RunAllGatherMesh] Rank[%d] templateRankSize[%u] totalLinks[%u].",
              myRank_, templateRankSize_, channelsPerRank_);
    // ========== 新增日志 ==========
    HCCL_INFO("[InsTempAllGatherMeshClosOpt][RunAllGatherMesh] threads.size=%zu, channels.size=%zu",
              threads.size(), channels.size());
    // ============================
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }

     for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        if (remoteWrite) {
            u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
            u64 sliceCount = sliceSize / dataTypeSize;
            u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * tempAlgParams_.outputSliceStride;
            DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[threads.size() - 1], srcSlice, dstSlice));
            break;
        } else {
            u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
            u64 sliceCount = sliceSize / dataTypeSize;

            u32 localMeshRank = (myRank_ + rpt) % meshSize_;
            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + (localMeshRank + myRank_ / meshSize_ * meshSize_) * tempAlgParams_.outputSliceStride;;
            u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + (localMeshRank + myRank_ / meshSize_ * meshSize_) * tempAlgParams_.outputSliceStride;

            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[threads.size() - 1], srcSlice, dstSlice));
        }
    }
    
    for (u32 linkIdx = 0; linkIdx < threads.size() - 1; linkIdx++) {
        CHK_RET(RunAllGatherOnLink(threads, channels, linkIdx));
    }
    for (u32 step = allGatherToAllRanksCounts; step < allGatherToAllRanksCounts + tempAlgParams1_.repeatNum; step++) {
        if (step == 0) {
            // 本地数据需要拷贝到 HCCL BUFFER内
            const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
            u64 sliceSize = tempAlgParams1_.buffInfo.inputSize;
            u64 sliceCount = sliceSize / dataTypeSize;
            u64 inputOffset = tempAlgParams1_.buffInfo.inBuffBaseOff;
            u64 scratchOffset = tempAlgParams1_.buffInfo.hcclBuffBaseOff + myRank_ * tempAlgParams1_.outputSliceStride;
            DataSlice srcSlice(tempAlgParams1_.buffInfo.inputPtr, inputOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams1_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[threads.size() - 1], srcSlice, dstSlice));

            u64 outputOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + myRank_ * tempAlgParams1_.outputSliceStride;
            DataSlice dstSlice1(tempAlgParams1_.buffInfo.outputPtr, outputOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[threads.size() - 1], srcSlice, dstSlice1));
        }
        CHK_RET(RunAllGatherToAllRanks(threads, channels, step));
    }
     allGatherToAllRanksCounts += tempAlgParams1_.repeatNum;

    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMeshClosOpt::RunAllGatherToAllRanks(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels, u32 step)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myRank_ + 1 + neighborIdx) % subCommRanks_[0].size()];

        if (((connectedRank ^ myRank_) % rankSize_) != step) {
            continue;
        }

        auto it = channels.find(connectedRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAllGatherMeshClosOpt] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 linkIdx = totalLinksToNeighbor - 1;


        HCCL_INFO("[InsTempAllGatherMeshClosOpt] Rank[%d] linkIdx[%u] matched connectedRank[%u] "
                  "totalLinks[%u] totalThreads[%u] enableRemoteMemAccess[%d]",
                  myRank_, linkIdx, connectedRank, totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_);

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        
        u64 sliceSize = tempAlgParams1_.buffInfo.inputSize;
        u64 sliceCount = sliceSize / dataTypeSize;
        u64 outputSliceStride = tempAlgParams1_.outputSliceStride;
        u64 inputSliceStride = tempAlgParams1_.inputSliceStride;
        
        // 远端写 不应该启动
        void *txSrcPtr = tempAlgParams1_.buffInfo.inputPtr;
        u64 txSrcOffset = tempAlgParams1_.buffInfo.inBuffBaseOff;
        txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

        void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        u64 txDstOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + myRank_ * outputSliceStride;
        txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);

        // rx 远端读
        void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        u64 rxSrcOffset = tempAlgParams1_.buffInfo.hcclBuffBaseOff + connectedRank * outputSliceStride;
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
        
        void *rxDstPtr = tempAlgParams1_.buffInfo.outputPtr;
        u64 rxOutOffset = tempAlgParams1_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threads.size() - 1]),
                     HCCL_ERROR("[InsTempAllGatherMesh1DOpt] RunAllGather SendRecvRead failed"), HcclResult::HCCL_E_INTERNAL);
                    
    }
    return HCCL_SUCCESS;

}
HcclResult InsTempAllGatherMeshClosOpt::RunAllGatherOnLink(
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
            HCCL_ERROR("[InsTempAllGatherMeshClosOpt] Rank[%d] connectedRank[%u] has no channels.",
                       myRank_, connectedRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        u32 totalLinksToNeighbor = it->second.size();
        u32 selectedLinkIdx = (myRank_ ^ connectedRank) % (threads.size() - 1);

        if (selectedLinkIdx != linkIdx) {
            continue;
        }

        HCCL_INFO("[InsTempAllGatherMeshClosOpt] Rank[%d] linkIdx[%u] matched connectedRank[%u] "
                  "selectedLinkIdx[%u] totalLinks[%u] totalThreads[%u] enableRemoteMemAccess[%d]",
                  myRank_, linkIdx, connectedRank, selectedLinkIdx, totalLinksToNeighbor, threads.size(),
                  enableRemoteMemAccess_);

        const ChannelInfo &linkRemote = it->second[linkIdx];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        
        u64 sliceSize = tempAlgParams_.buffInfo.inputSize;
        u64 sliceCount = sliceSize / dataTypeSize;
        u64 outputSliceStride = tempAlgParams_.outputSliceStride;
        u64 inputSliceStride = tempAlgParams_.inputSliceStride;

        if (remoteWrite) {
            // 阶段1 远端写， 从本地的 input ptr 到 远端的 hccl buffer
            // 需要确保 当前场景下配置的repeatNum 为 1

            // tx 远端写
            void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
            u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            u64 txDstOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * outputSliceStride;
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);

            // rx 远端读，不应该启动
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            u64 rxSrcOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + connectedRank * outputSliceStride;
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
            
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + connectedRank * outputSliceStride;
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

            TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[linkIdx]),
                        HCCL_ERROR("[InsTempAllGatherMesh1DOpt] RunAllGather SendRecvWrite failed"), HcclResult::HCCL_E_INTERNAL);

        } else {
            // 阶段 2 远端读， 从远端的 hccl buffer 到 本地的 output buffer
            for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
                
                // tx 远端写, 不应该启动
                void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
                u64 txSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff;
                txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, sliceSize, sliceCount);

                void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                u64 txDstOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + myRank_ * outputSliceStride;
                txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);


                // rx 远端读
                u32 remoteMeshRank = (connectedRank + rpt) % meshSize_;
                u32 remoteRank = remoteMeshRank + connectedRank / meshSize_ * meshSize_;
                void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                u64 rxSrcOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + remoteRank * outputSliceStride;
                rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);
                
                void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
                u64 rxOutOffset = tempAlgParams_.buffInfo.outBuffBaseOff + remoteRank * outputSliceStride;
                rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);

                TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
                TxRxChannels sendRecvChannels(linkRemote, linkRemote);
                SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
                CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[linkIdx]),
                            HCCL_ERROR("[InsTempAllGatherMesh1DOpt] RunAllGather SendRecvRead failed"), HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
