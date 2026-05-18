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

namespace ops_hccl {
ReduceScatterBIRSInter::ReduceScatterBIRSInter() : AlgTemplateBase()
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

HcclResult ReduceScatterBIRSInter::Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem,
                                 const u64 count,
                                 const HcclDataType dataType, ThreadHandle thread, const std::vector<ThreadHandle> &slaveThreads,
                                 const HcclReduceOp reductionOp,
                                 const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                                 const bool disableDMAReduce)
{
    mainThread = thread;
    subThreads = slaveThreads;
    AlgTemplateBase::Prepare(inputMem, outputMem, scratchMem,
                                 count, dataType, thread, reductionOp, root, slices, baseOffset,
                                 disableDMAReduce);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread) {
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + srcOffset);
    void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + dstOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, dstSlice, srcSlice, size / unitSize, static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_))));
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
    return HCCL_SUCCESS;
}

void ReduceScatterBIRSInter::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 3;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void ReduceScatterBIRSInter::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 3;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

void ReduceScatterBIRSInter::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = totalCount * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
}

// scatter的入口函数
HcclResult ReduceScatterBIRSInter::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    HCCL_INFO("ReduceScatterBIRSInter run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.addr, outputMem_.addr, count_);
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }
    if (channels.size() < rankSize) {
        HCCL_ERROR("[ReduceScatterBIRSInter][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, channels.size(), rankSize);
        return HCCL_E_INTERNAL;
    }
    unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterBIRSInter][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }

    u32 rankSizeX_ = 2;
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
    u64 localStrideSize = sliceSize; //TODO: RoundUpWithDivisor(sliceSize, HCCL_MIN_SLICE_ALIGN_910B); 
    
    //MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    
    LocalCopyPreproc(mainThread, rank, sliceSize, localStrideSize);
    for (u32 cnt = 0; cnt < serverNum_; cnt++)
    {
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + (hccs_neighbour_rank[0] % intraRankSize_ + cnt * intraRankSize_) * sliceSize);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + (intraRankSize_) * localStrideSize * serverNum_ + cnt * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    }
    //SubRecordMain + MainWaitSub
    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        //MainRecordSub + SubWaitMain
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        
        if (round != 0) { //subThreads[0]
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

        //SubRecordMain + MainWaitSub
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }

    // MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);

    std::vector<u32> vec;
    for (u32 i = 0; i < (intraRankSize_ / rankSizeX_); i++){
        if (i == (rank % intraRankSize_ / rankSizeX_)) {
            vec.push_back((rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_);
        } else {
            vec.push_back(((intraRankSize_ / rankSizeX_) + i) * localStrideSize * serverNum_);
        }
    }

    auto ind = intraRankSize_ / rankSizeX_;;
    for (u32 i = 1; i < ind; i+=2){
        LocalReduceCCLToCCL(vec[i], vec[i - 1], localStrideSize * serverNum_, mainThread);
    }
    for (size_t i = 2; i < ind; i+=4){
        LocalReduceCCLToCCL(vec[i], vec[i - 2], localStrideSize * serverNum_, mainThread);
    }
    for (size_t i = 4; i < ind; i+=8){
        LocalReduceCCLToCCL(vec[i], vec[i - 4], localStrideSize * serverNum_, mainThread);
    }
    HCCL_INFO("ReduceScatterBIRSInter pass intrarank part on rank[%u]", rank);
    {
        u64 remoteOffsetByte = localStrideSize * serverNum_;
        u64 localOffsetByte = vec[0] + localStrideSize * (((rank) % rankSize) / intraRankSize_);
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + remoteOffsetByte);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dst, src, sliceSize)));
    }

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    for (u32 round = 1; round < serverNum_; round++)
    {
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channels[((round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        void *remMemPtr = nullptr;
        u64 remoteOffsetByte =  localStrideSize * serverNum_  + localStrideSize * round;
        u64 localOffsetByte = vec[0] + localStrideSize * ((((round) * intraRankSize_ + rank) % rankSize) / intraRankSize_);
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(channels[((round) * intraRankSize_ + rank) % rankSize].remoteOutput.addr) + remoteOffsetByte);
        HcommWriteOnThread(mainThread, channels[((round) * intraRankSize_ + rank) % rankSize].handle, dst, src, sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channels[((round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }

    u64 ptrSz = (serverNum_ == 1 ? vec[0] : localStrideSize * serverNum_);
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

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    HCCL_INFO("ReduceScatterBIRS finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCE_SCATTER_BIRS_INTER, ReduceScatterBIRSInter);
}
