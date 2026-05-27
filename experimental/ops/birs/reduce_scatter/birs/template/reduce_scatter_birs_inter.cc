/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "reduce_scatter_birs_inter.h"

namespace ops_hccl_experimental {
using ops_hccl::NOTIFY_IDX_ACK;
using ops_hccl::CUSTOM_TIMEOUT;
using ops_hccl::AlgTemplateRegistry;
using ops_hccl::HCCL_MIN_SLICE_ALIGN_910B;
using ops_hccl::PostSyncInterThreads;
using ops_hccl::NOTIFY_IDX_DATA_SIGNAL;
using ops_hccl::RoundUpWithDivisor;
using ops_hccl::PreSyncInterThreads;
using ops_hccl::TemplateType;
using ops_hccl::DefaultTemplateCreator;

ReduceScatterBIRSInter::ReduceScatterBIRSInter() : ReduceScatterBIRS()
{
}

ReduceScatterBIRSInter::~ReduceScatterBIRSInter()
{
}

HcclResult ReduceScatterBIRSInter::Prepare(u32 serverNum, u32 interRankSize)
{
    serverNum_ = serverNum;
    intraRankSize_ = interRankSize / serverNum_;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::LocalCopyPreproc(ThreadHandle &thread, const u32 rank, u64 sliceSize, u64 localStrideSize) {
    void* src;
    void* dst;
    u32 cnt = 0;
    for (u32 round = 0; round < intraRankSize_ / 2; round++) {
        for (u32 i = 0; i < serverNum_; i++)
        {
            src = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + (2 * round + i * intraRankSize_ + rank % 2) * sliceSize);
            dst = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + cnt * localStrideSize);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, sliceSize)));
            cnt++;
        }
    }

    for (u32 cnt = 0; cnt < serverNum_; cnt++)
    {
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + (hccs_neighbour_rank[0] % intraRankSize_ + cnt * intraRankSize_) * sliceSize);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + (intraRankSize_) * localStrideSize * serverNum_ + cnt * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    HCCL_INFO("ReduceScatterBIRSInter run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.addr, outputMem_.addr, count_);
    return ReduceScatterBIRS::Preprocess(rank, rankSize, channels);
}

HcclResult ReduceScatterBIRSInter::HCCSInterStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round != 0) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = hccs_ranks[round - 1] % intraRankSize_ / rankSizeX_ * localStrideSize * serverNum_;
        u64 remoteOffsetByte = ((intraRankSize_ / rankSizeX_) + rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_;
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte);
        
        HcommWriteOnThread(subThreads[0], hccs_links[round - 1].handle, dst, src, localStrideSize * serverNum_);

        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::SIOInterStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round != hccs_ranks.size()) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        
        u64 localOffsetByte = (intraRankSize_ + (round % 2)) * localStrideSize * serverNum_;
        u64 remoteOffsetByte = (hccs_ranks[round] % intraRankSize_) / rankSizeX_ * localStrideSize * serverNum_;
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(sio_link.remoteOutput.addr) + remoteOffsetByte);

        HcommWriteReduceOnThread(mainThread, sio_link.handle, dst, src, localStrideSize * serverNum_ / unitSize, static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_));
    } else {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = (intraRankSize_ + (round % 2)) * localStrideSize * serverNum_;
        u64 remoteOffsetByte = (rank % intraRankSize_) / rankSizeX_ * localStrideSize * serverNum_;
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(sio_link.remoteOutput.addr) + remoteOffsetByte);
        HcommWriteReduceOnThread(mainThread, sio_link.handle, dst, src, localStrideSize * serverNum_ / unitSize, static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::LocalCopyInterStep(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round < hccs_ranks.size() - 1) {
        for (u32 cnt = 0; cnt < serverNum_; cnt++)
        {  
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + ((hccs_neighbour_rank[round + 1]) % intraRankSize_ + cnt * intraRankSize_) * sliceSize);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + (intraRankSize_ + ((round + 1) % 2)) * localStrideSize * serverNum_ + cnt * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
        }
    } 
    if (round == hccs_ranks.size() - 1) {
        for (u32 cnt = 0; cnt < serverNum_; cnt++)
        {
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + ((sio_rank) % intraRankSize_ + cnt * intraRankSize_) * sliceSize);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + (intraRankSize_ + ((round + 1) % 2)) * localStrideSize * serverNum_ + cnt * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::PreprocIntra(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize, std::vector<ChannelInfo> &channels)
{
    for (u32 i = 0; i < (intraRankSize_ / rankSizeX_); i++){
        if (i == (rank % intraRankSize_ / rankSizeX_)) {
            vec_offsets.push_back((rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_);
        } else {
            vec_offsets.push_back(((intraRankSize_ / rankSizeX_) + i) * localStrideSize * serverNum_);
        }
    }
    auto ind = intraRankSize_ / rankSizeX_;;
    for (u32 i = 1; i < ind; i+=2){
        LocalReduceCCLToCCL(vec_offsets[i], vec_offsets[i - 1], localStrideSize * serverNum_, mainThread);
    }
    for (size_t i = 2; i < ind; i+=4){
        LocalReduceCCLToCCL(vec_offsets[i], vec_offsets[i - 2], localStrideSize * serverNum_, mainThread);
    }
    for (size_t i = 4; i < ind; i+=8){
        LocalReduceCCLToCCL(vec_offsets[i], vec_offsets[i - 4], localStrideSize * serverNum_, mainThread);
    }

    u64 remoteOffsetByte = localStrideSize * serverNum_;
    u64 localOffsetByte = vec_offsets[0] + localStrideSize * (((rank) % rankSize) / intraRankSize_);
    void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
    void* dst = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + remoteOffsetByte);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dst, src, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::IntraLoop(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize, std::vector<ChannelInfo> &channels)
{
    for (u32 round = 1; round < serverNum_; round++)
    {
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channels[((round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        void *remMemPtr = nullptr;
        u64 remoteOffsetByte =  localStrideSize * serverNum_  + localStrideSize * round;
        u64 localOffsetByte = vec_offsets[0] + localStrideSize * ((((round) * intraRankSize_ + rank) % rankSize) / intraRankSize_);
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(channels[((round) * intraRankSize_ + rank) % rankSize].remoteOutput.addr) + remoteOffsetByte);
        HcommWriteOnThread(mainThread, channels[((round) * intraRankSize_ + rank) % rankSize].handle, dst, src, sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channels[((round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }
    u64 ptrSz = (serverNum_ == 1 ? vec_offsets[0] : localStrideSize * serverNum_);
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    for (u32 step = 1; step < serverNum_; step *= 2)
    {
        for (u32 i = 0; i + step < serverNum_; i+=2 * step){
            LocalReduceCCLToCCL(ptrSz + sliceSize * (i + step), ptrSz + sliceSize * i, sliceSize, mainThread);
        }
    }    
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + ptrSz);
    void* dstSlice =  static_cast<void *>(static_cast<u8 *>(outputMem_.addr));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    return HCCL_SUCCESS;
}

// scatter的入口函数
HcclResult ReduceScatterBIRSInter::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    Preprocess(rank, rankSize, channels);
    u32 rankSizeX_ = 2;
    if (intraRankSize_ % rankSizeX_ != 0) {
        HCCL_ERROR("[ReduceScatterBIRS][RunAsync]intraRankSize_[%u] is not evenly divisible by rankSizeX_[%u]", intraRankSize_, rankSizeX_);
        return HCCL_E_INTERNAL;
    }
    u32 rankSizeY_ = intraRankSize_ / rankSizeX_;

    sio_rank = rank ^ 1;
    sio_link = channels[sio_rank];

    for (u32 i = 1; i < rankSizeY_; ++i) {
        u32 current_hccs_rank = (rank % intraRankSize_ + rankSizeX_ * i) % (intraRankSize_) + rank / intraRankSize_ * intraRankSize_;
        hccs_ranks.push_back(current_hccs_rank);
        hccs_neighbour_rank.push_back(current_hccs_rank ^ 1);
        hccs_links.push_back(channels[hccs_ranks[i-1]]);
    }
    hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());
    
    u64 sliceSize = count_ * unitSize;
    u64 localStrideSize = sliceSize; 

    //MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    
    LocalCopyPreproc(mainThread, rank, sliceSize, localStrideSize);
    
    //SubRecordMain + MainWaitSub
    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        //MainRecordSub + SubWaitMain
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        
        HCCSInterStep(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        SIOInterStep(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        LocalCopyInterStep(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        //SubRecordMain + MainWaitSub
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }

    // MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);

    PreprocIntra(rank, rankSize, rankSizeX_, sliceSize, localStrideSize, channels);

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    
    IntraLoop(rank, rankSize, rankSizeX_, sliceSize, localStrideSize, channels);

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    HCCL_INFO("ReduceScatterBIRS finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCE_SCATTER_BIRS_INTER, ReduceScatterBIRSInter);
}
