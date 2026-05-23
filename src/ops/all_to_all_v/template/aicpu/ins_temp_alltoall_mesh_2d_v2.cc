/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_2d_v2.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "channel.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {

InsTempAlltoAllMesh2DV2::InsTempAlltoAllMesh2DV2(const OpParam &param, const u32 rankId,
                                                   const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    xRankSize_ = subCommRanks_[0].size();
    for (u64 i = 0; i < xRankSize_; i++) {
        if (subCommRanks_[0][i] == rankId) {
            myXRingRank_ = i;
            break;
        }
    }
}

InsTempAlltoAllMesh2DV2::~InsTempAlltoAllMesh2DV2() {}

HcclResult InsTempAlltoAllMesh2DV2::CalcChannelRequestMesh2D(
    HcclComm comm,
    const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(::ops_hccl::CalcChannelRequestMesh2D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::CalcRes(
    HcclComm comm,
    const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][CalcRes] start");
    CHK_RET(CalcChannelRequestMesh2D(comm, param, topoInfo, resourceRequest));
    return GetRes(resourceRequest);
}

HcclResult InsTempAlltoAllMesh2DV2::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.slaveThreadNum = xRankSize_ - 1;
    resourceRequest.notifyNumOnMainThread = xRankSize_ - 1;
    resourceRequest.notifyNumPerThread.assign(xRankSize_ - 1, 1);
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllMesh2DV2::GetThreadNum() const
{
    return xRankSize_ > 1 ? xRankSize_ : 1;
}

u64 InsTempAlltoAllMesh2DV2::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return xRankSize_;
}

HcclResult InsTempAlltoAllMesh2DV2::FastLaunch(
    const OpParam &param,
    const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::KernelRun(
    const OpParam &param,
    const TemplateDataParams &tempAlgParams,
    TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV2][KernelRun] Run Start");

    dataType_ = param.DataDes.dataType;
    myRank_ = param.userRank;

    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV2] sliceSize is zero, skip.");
        return HCCL_SUCCESS;
    }

    u64 perRankChunk = tempAlgParams.inputSliceStride;
    allRankSize_ = tempAlgParams.sliceSize / perRankChunk;
    assert(allRankSize_ > 0 && tempAlgParams.sliceSize % perRankChunk == 0);

    isPcieProtocol_ = IsPcieProtocol(templateResource.channels);
    HCCL_DEBUG("[InsTempAlltoAllMesh2DV2][KernelRun] isPcieProtocol_[%d]", isPcieProtocol_);

    ThreadHandle mainThread = templateResource.threads[0];

    CHK_RET(LocalDataCopy(mainThread, tempAlgParams));

    if (xRankSize_ > 1) {
        std::vector<ThreadHandle> subThreads(
            templateResource.threads.begin() + 1,
            templateResource.threads.end());

        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

        CHK_RET(RunAlltoAllMeshX(subThreads, templateResource.channels, tempAlgParams));

        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }

    CHK_RET(PostLocalCopy(mainThread, tempAlgParams));

    HCCL_INFO("[InsTempAlltoAllMesh2DV2][KernelRun] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::RunAlltoAllMeshX(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 perRankChunk = params.inputSliceStride;

    std::vector<u64> srcAddrs(allRankSize_), dstAddrs(allRankSize_);
    for (u32 i = 0; i < allRankSize_; i++) {
        srcAddrs[i] = params.buffInfo.inBuffBaseOff + perRankChunk * i;
        dstAddrs[i] = params.buffInfo.hcclBuffBaseOff + perRankChunk * i;
    }

    for (u64 neighborIdx = 0; neighborIdx < xRankSize_ - 1; neighborIdx++) {
        u64 remoteXRingRank = (myXRingRank_ + 1 + neighborIdx) % xRankSize_;
        u32 connectedAxisRank = static_cast<u32>(remoteXRingRank);
        u64 remoteRank = subCommRanks_[0][remoteXRingRank];

        auto it = channels.find(remoteRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllMesh2DV2][RunAlltoAllMeshX] no channels for rank[%llu]", remoteRank);
            return HCCL_E_INTERNAL;
        }

        u32 totalLinks = it->second.size();
        u32 selectedLinkIdx = (myXRingRank_ + remoteXRingRank) % totalLinks;
        const ChannelInfo &linkRemote = it->second[selectedLinkIdx];

        bool toScratch = (params.buffInfo.outBuffType == BufferType::HCCL_BUFFER);
        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;

        for (u32 i = 0; i < allRankSize_; i++) {
            if (i % xRankSize_ == connectedAxisRank) {
                void *sPtr = toScratch ? params.buffInfo.inputPtr : params.buffInfo.hcclBuff.addr;
                void *dPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                u64 sc = perRankChunk / dataTypeSize;
                u32 txDstGlobalIdx = (i / xRankSize_) * xRankSize_ + myXRingRank_;
                txSrc.emplace_back(sPtr, srcAddrs[i], perRankChunk, sc);
                txDst.emplace_back(dPtr, dstAddrs[txDstGlobalIdx], perRankChunk, sc);
            }
            if (i % xRankSize_ == myXRingRank_) {
                void *rxSPtr = toScratch ? linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
                void *rxDPtr = toScratch ? params.buffInfo.hcclBuff.addr : params.buffInfo.outputPtr;
                u64 sc = perRankChunk / dataTypeSize;
                u32 rxSrcGlobalIdx = (i / xRankSize_) * xRankSize_ + connectedAxisRank;
                rxSrc.emplace_back(rxSPtr, srcAddrs[rxSrcGlobalIdx], perRankChunk, sc);
                rxDst.emplace_back(rxDPtr, dstAddrs[i], perRankChunk, sc);
            }
        }

        if (txSrc.empty() && rxSrc.empty()) {
            continue;
        }

        TxRxSlicesList slicesList({txSrc, txDst}, {rxSrc, rxDst});
        TxRxChannels txRxChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(txRxChannels, slicesList);

        if (isPcieProtocol_) {
            CHK_RET(SendRecvRead(sendRecvInfo, threads[neighborIdx]));
        } else {
            CHK_RET(SendRecvWrite(sendRecvInfo, threads[neighborIdx]));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::LocalDataCopy(
    const ThreadHandle &thread,
    const TemplateDataParams &params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 perRankChunk = params.inputSliceStride;
    u64 sliceCount = perRankChunk / dataTypeSize;

    u64 srcOffset = params.buffInfo.inBuffBaseOff + params.inputSliceStride * myRank_;
    u64 dstOffset = params.buffInfo.hcclBuffBaseOff + perRankChunk * myRank_;

    DataSlice srcSlice(params.buffInfo.inputPtr, srcOffset, perRankChunk, sliceCount);
    DataSlice dstSlice(params.buffInfo.hcclBuff.addr, dstOffset, perRankChunk, sliceCount);

    CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV2::PostLocalCopy(
    const ThreadHandle &thread,
    const TemplateDataParams &params)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 perRankChunk = params.inputSliceStride;
    u64 sliceCount = perRankChunk / dataTypeSize;
    u64 rankSize = params.sliceSize / params.inputSliceStride;

    for (u64 i = 0; i < rankSize; i++) {
        u64 srcOffset = params.buffInfo.hcclBuffBaseOff + perRankChunk * i;
        u64 dstOffset = params.buffInfo.outBuffBaseOff + params.outputSliceStride * i;

        DataSlice srcSlice(params.buffInfo.hcclBuff.addr, srcOffset, perRankChunk, sliceCount);
        DataSlice dstSlice(params.buffInfo.outputPtr, dstOffset, perRankChunk, sliceCount);

        CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
    }

    return HCCL_SUCCESS;
}

void InsTempAlltoAllMesh2DV2::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    if (xRankSize_ <= 1) {
        return;
    }
    notifyIdxMainToSub.assign(xRankSize_ - 1, 0);
}

void InsTempAlltoAllMesh2DV2::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    if (xRankSize_ <= 1) {
        return;
    }
    notifyIdxSubToMain.resize(xRankSize_ - 1);
    for (u64 i = 0; i < xRankSize_ - 1; i++) {
        notifyIdxSubToMain[i] = static_cast<u32>(i);
    }
}

}  // namespace ops_hccl

REGISTER_TEMPLATE_V2("AlltoAllMesh2DV2", InsTempAlltoAllMesh2DV2)
