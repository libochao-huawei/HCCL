/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_2D.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
InsTempReduceScatterMesh2D::InsTempReduceScatterMesh2D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    xThreadNum_ = subCommRanks[0].size() - 1; // x轴的卡数-1
    yThreadNum_ = subCommRanks[1].size() - 1; // y轴的卡数-1
    xRankSize_ = subCommRanks[0].size(); // x轴的卡数
    yRankSize_ = subCommRanks[1].size(); // y轴的卡数
}

InsTempReduceScatterMesh2D::~InsTempReduceScatterMesh2D()
{
}

u64 InsTempReduceScatterMesh2D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u32 xyMaxRankSize = std::max(xRankSize_, yRankSize_);
    u64 scratchMultiple = xyMaxRankSize * (xRankSize_ + yRankSize_);
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh2D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // Mesh 需要的 thread Num 为 subCommRanks_[0].size() + subCommRanks_[1].size() - 2
    u32 threadNum = (xRankSize_ > 1 && yRankSize_ > 1) ? (xThreadNum_ + yThreadNum_): 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestMesh2D(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::GetRes(AlgResourceRequest& resourceRequest)
{
    u32 threadNum = (xRankSize_ > 1 && yRankSize_ > 1) ? (xThreadNum_ + yThreadNum_): 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], xAlgRankId_)); // 得到当前卡在x轴上的编号
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[1], yAlgRankId_)); // 得到当前卡在y轴上的编号
    threadNum_ = xThreadNum_ + yThreadNum_;
    dataType_ = param.DataDes.dataType;
    u64 sliceNum = tempAlgParams.sliceSize / DATATYPE_SIZE_TABLE[dataType_]; // 先计算得到本次迭代处理的数据量
    halfDataSize_ = sliceNum / PARALLEL_SIZE * DATATYPE_SIZE_TABLE[dataType_]; // 前一半数据的size
    HCCL_INFO("[InsTempReduceScatterMesh2D] Run Start");
    // 这里不支持绕路的时候，应该就用原始的tempInsQues就行
    CHK_PRT_RET(threadNum_ != templateResource.threads.size(),
                HCCL_ERROR("[CollAlgFactory] [InsTempReduceScatterMesh2D] Rank [%d], requiredQue Error.", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    PreCopy(tempAlgParams, templateResource.threads); // stream 0作为主流，负责把本卡的数据拷贝到scratchbuffer上
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunFirstLevel(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunFirstReduce(templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    RunSecondLevel(templateResource.channels, templateResource.threads, tempAlgParams);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    RunSecondReduce(templateResource.threads, tempAlgParams);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::PreCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 xyMaxRankSize = std::max(xRankSize_, yRankSize_);
    u64 remainDataSize = tempAlgParams.sliceSize - halfDataSize_;
    // 前一半数据，将本卡数据从input拷贝到scratchbuffer
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
        for (u32 yRankId = 0; yRankId < yRankSize_; yRankId++) {
            u32 rankId = yRankId * xRankSize_ + xAlgRankId_;  // 同y轴平面的所有卡，
            DataSlice inputRankSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.buffInfo.inBuffBaseOff + rankId * tempAlgParams.inputSliceStride +
                    rpt * tempAlgParams.inputRepeatStride, halfDataSize_);
            DataSlice scratchRankSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff +
                    tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId + xAlgRankId_) + scratchRepeatStride, halfDataSize_);
            CHK_RET(LocalCopy(threads[0], inputRankSlice, scratchRankSlice));
            // HCCL_DEBUG("[InsTempReduceScatterMesh2D][PreCopy] myRank[%d] top inputRankSlice: %s, scratchRankSlice: %s",
            //     myRank_, inputRankSlice.Describe().c_str(), scratchRankSlice.Describe().c_str());
        }
    }
    // 后一半数据，将本卡数据从input拷贝到scratchbuffer
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum) +
            tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
        for (u32 xRankId = 0; xRankId < xRankSize_; xRankId++) {
            u32 rankId = yAlgRankId_ * xRankSize_ + xRankId;  // 同x轴平面的所有卡，
            DataSlice inputRankSlice = DataSlice(tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.buffInfo.inBuffBaseOff + rankId * tempAlgParams.inputSliceStride + halfDataSize_ +
                    rpt * tempAlgParams.inputRepeatStride, remainDataSize);
            DataSlice scratchRankSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff +
                    tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId + yAlgRankId_) + scratchRepeatStride,
                    remainDataSize);
            CHK_RET(LocalCopy(threads[0], inputRankSlice, scratchRankSlice));
            // HCCL_DEBUG("[InsTempReduceScatterMesh2D][PreCopy] myRank[%d] bottom inputRankSlice: %s, scratchRankSlice: %s",
            //     myRank_, inputRankSlice.Describe().c_str(), scratchRankSlice.Describe().c_str());
        }
    }
    HCCL_INFO("[InsTempReduceScatterMesh2D][PreCopy], copy from userIn to scratch");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::SendRecvProcess(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    std::vector<std::vector<DataSlice>> allSliceVec,
    const std::vector<ThreadHandle> &threads, u32 remoteRank, u32 threadIdx) const
{
    HCCL_DEBUG("[InsTempReduceScatterMesh2D][SendRecvProcess] SendRecvProcess start");
    const std::vector<ChannelInfo> &linkRecv = channels.at(remoteRank);
    const std::vector<ChannelInfo> &linkSend = channels.at(remoteRank);
    SendRecvInfo sendRecvInfo{{linkSend[0], linkRecv[0]},
                                {{allSliceVec[2], allSliceVec[3]}, {allSliceVec[0], allSliceVec[1]}}};
    // 做了DMA消减之后只支持PUT
    CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempReduceScatterMesh2D] RunReduceScatter SendReduce failed"),
                HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

// 前一半数据的先x轴 和 后一半数据的先y轴
HcclResult InsTempReduceScatterMesh2D::RunFirstLevel(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[InsTempReduceScatterMesh2D][RunFirstLevel] myRank[%d]", myRank_);
    u32 xyMaxRankSize = std::max(xRankSize_, yRankSize_);
    u64 processSize;
    for (u32 threadIdx = 0; threadIdx < threadNum_; threadIdx++) {
        u32 remoteRank;
        u32 index;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        if (threadIdx < xThreadNum_) {  // 前xRankSize-1个stream，首先拉取前一半数据
            index = (xAlgRankId_ + 1 + threadIdx) % (subCommRanks_[0].size());
            remoteRank = subCommRanks_[0][index];
            void* remoteCclBuffAddr = channels.at(remoteRank)[0].remoteCclMem.addr;
            processSize = halfDataSize_;
            HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunFirstLevel] queID < xQueNum myRank[%d] toRank[%u] fromRank[%u] rpt[%u], index[%u]",
                myRank_, remoteRank, remoteRank, tempAlgParams.repeatNum, index);
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
                for (u32 yRankId = 0; yRankId < yRankSize_; yRankId++) {
                    u32 readRankId = yRankId * xRankSize_ + xAlgRankId_;
                    u32 writeRankId = yRankId * xRankSize_ + index;
                    // 数据从其他卡，传输到本卡，接收数据
                    rxSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr,
                        tempAlgParams.buffInfo.inBuffBaseOff + readRankId * tempAlgParams.inputSliceStride +
                            rpt * tempAlgParams.inputRepeatStride, processSize);
                    rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                        tempAlgParams.buffInfo.hcclBuffBaseOff +
                            tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId + index) + scratchRepeatStride, processSize);
                    txSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr,
                        tempAlgParams.buffInfo.inBuffBaseOff + writeRankId * tempAlgParams.inputSliceStride +
                            rpt * tempAlgParams.inputRepeatStride, processSize);
                    txDstSlices.emplace_back(remoteCclBuffAddr,
                        tempAlgParams.buffInfo.hcclBuffBaseOff +
                            tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankId + xAlgRankId_) + scratchRepeatStride, processSize);
                    // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunFirstLevel] queID < xQueNum myRank[%d] *****sendrecv*****, "
                    //     "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s", myRank_,
                    //     rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                    //     txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
                }
            }
        } else {  // 后yRankSize-1个stream,首先拉取后一半数据
            index = (yAlgRankId_ + 1 + threadIdx - xThreadNum_) % (subCommRanks_[1].size());
            remoteRank = subCommRanks_[1][index];
            void* remoteCclBuffAddr = channels.at(remoteRank)[0].remoteCclMem.addr;
            processSize = tempAlgParams.sliceSize - halfDataSize_;
            HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunFirstLevel] queId >= xQueNum myRank[%d] toRank[%u] fromRank[%u], rpt[%u], index[%u]",
                myRank_, remoteRank, remoteRank, tempAlgParams.repeatNum, index);
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum) +
                    tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
                for (u32 xRankId = 0; xRankId < xRankSize_; xRankId++) {
                    u32 readRankId = yAlgRankId_ * xRankSize_ + xRankId;  // 同x轴平面的所有卡，
                    u32 writeRankId = index * xRankSize_ + xRankId;
                    rxSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr,
                        tempAlgParams.buffInfo.inBuffBaseOff + readRankId * tempAlgParams.inputSliceStride +
                            halfDataSize_ + rpt * tempAlgParams.inputRepeatStride, processSize);
                    rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr,
                        tempAlgParams.buffInfo.hcclBuffBaseOff +
                            tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId + index) + scratchRepeatStride,
                            processSize);
                    txSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr,
                        tempAlgParams.buffInfo.inBuffBaseOff + writeRankId * tempAlgParams.inputSliceStride +
                            halfDataSize_ + rpt * tempAlgParams.inputRepeatStride, processSize);
                    txDstSlices.emplace_back(remoteCclBuffAddr,//tempAlgParams.buffInfo.scratBuffType,
                        tempAlgParams.buffInfo.hcclBuffBaseOff +
                            tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankId + yAlgRankId_) + scratchRepeatStride,
                            processSize);
                    // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunFirstLevel] queId >= xQueNum myRank[%d] *****sendrecv*****, "
                    //     "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s", myRank_,
                    //     rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                    //     txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
                }
            }
        }
        if (processSize == 0) {
            continue;
        }
        std::vector<std::vector<DataSlice>> allSliceVec = {rxSrcSlices, rxDstSlices, txSrcSlices, txDstSlices};
        CHK_RET(SendRecvProcess(channels, allSliceVec, threads, remoteRank, threadIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::RunFirstReduce(
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[InsTempReduceScatterMesh2D][RunFirstReduce] myRank[%d] rpt[%u]", myRank_, tempAlgParams.repeatNum);
    u32 xyMaxRankSize = std::max(xRankSize_, yRankSize_);
    u64 processSize = 0;
    // 这里的stream 0和stream xRankSize-1分别负责前一半数据与后一半数据的本地reduce
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
        for (u32 tmpRank = 0; tmpRank < yRankSize_; tmpRank++) {  // 前一半数据做local reduce,由这部分的第一个stream做
            processSize = halfDataSize_;
            for (u32 dataIdx = 1; dataIdx < xRankSize_; dataIdx++) {  // 原始这个位置已经有数据了，因此从后一片数据开始累加
                DataSlice srcDataSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        (xyMaxRankSize * tmpRank + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                        processSize,
                        processSize / DATATYPE_SIZE_TABLE[dataType_]
                        );
                DataSlice dstDataSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        (xyMaxRankSize * tmpRank) * tempAlgParams.outputSliceStride + scratchRepeatStride, 
                        processSize,
                        processSize / DATATYPE_SIZE_TABLE[dataType_]);
                // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunFirstReduce] myRank[%d] queId < xQueNum *****LocalReduce*****, "
                //     "srcDataSlice: %s, dstDataSlice: %s", myRank_, srcDataSlice.Describe().c_str(),
                //     dstDataSlice.Describe().c_str());
                CHK_RET(LocalReduce(threads[0], srcDataSlice, dstDataSlice, dataType_, reduceOp_));
            }
        }
    }
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum) +
            tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
        for (u32 tmpRank = 0; tmpRank < xRankSize_; tmpRank++) {  // 后一半数据做local reduce，由这部分的第一个stream做
            processSize = tempAlgParams.sliceSize - halfDataSize_;
            for (u32 dataIdx = 1; dataIdx < yRankSize_; dataIdx++) {  // 原始这个位置已经有数据了，因此从后一片数据开始累加
                DataSlice srcDataSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        (xyMaxRankSize * tmpRank + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize,
                    processSize / DATATYPE_SIZE_TABLE[dataType_]);
                DataSlice dstDataSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        (xyMaxRankSize * tmpRank) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize,
                    processSize / DATATYPE_SIZE_TABLE[dataType_]);
                // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunFirstReduce] myRank[%d] queId >= xQueNum *****LocalReduce*****, "
                //     "srcDataSlice: %s, dstDataSlice: %s", myRank_, srcDataSlice.Describe().c_str(),
                //     dstDataSlice.Describe().c_str());
                CHK_RET(LocalReduce(threads[xThreadNum_], srcDataSlice, dstDataSlice, dataType_, reduceOp_));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 后一半数据的后x轴 和 前一半数据的后y轴
HcclResult InsTempReduceScatterMesh2D::RunSecondLevel(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    u32 xyMaxRankSize = std::max(xRankSize_, yRankSize_);
    u64 processSize;
    for (u32 threadIdx = 0; threadIdx < threadNum_; threadIdx++) {
        u32 remoteRank;
        u32 index;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        if (threadIdx < xThreadNum_) { // 前xRankSize-1个stream，后一半数据
            index = (xAlgRankId_ + 1 + threadIdx) % (subCommRanks_[0].size());
            remoteRank = subCommRanks_[0][index];
            void* remoteCclBuffAddr = channels.at(remoteRank)[0].remoteCclMem.addr;
            HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunSecondLevel] threadIdx < xQueNum myRank[%d] toRank[%u] fromRank[%u]",
                myRank_, remoteRank, remoteRank);
            processSize = tempAlgParams.sliceSize - halfDataSize_;
            // 这里过来的数据，直接按照queIdx的顺序放置，不一定是按照rankId顺序排列的
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum) +
                    tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
                DataSlice rxSrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * xAlgRankId_) + scratchRepeatStride, processSize);
                DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * xAlgRankId_ + threadIdx + 1) + scratchRepeatStride, processSize);
                DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * index) + scratchRepeatStride, processSize);
                DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * index + threadIdx + 1) + scratchRepeatStride, processSize);

                rxSrcSlices.emplace_back(rxSrcSlice);
                rxDstSlices.emplace_back(rxDstSlice);
                txSrcSlices.emplace_back(txSrcSlice);
                txDstSlices.emplace_back(txDstSlice);

                // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunSecondLevel] queId < xQueNum myRank[%d] *****sendrecv*****, "
                //     "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s", myRank_,
                //     rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                //     txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
            }
        } else { // 后yRankSize-1个stream,前一半数据
            index = (yAlgRankId_ + 1 + threadIdx - xThreadNum_) % (subCommRanks_[1].size());
            remoteRank = subCommRanks_[1][index];
            void* remoteCclBuffAddr = channels.at(remoteRank)[0].remoteCclMem.addr;
            HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunSecondLevel] queId >= xQueNum myRank[%d] toRank[%u] fromRank[%u] rpt[%u]",
                myRank_, remoteRank, remoteRank, tempAlgParams.repeatNum);
            processSize = halfDataSize_;
            for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
                u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
                DataSlice rxSrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * yAlgRankId_) + scratchRepeatStride, processSize);
                DataSlice rxDstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * yAlgRankId_ + threadIdx - xThreadNum_ + 1) + scratchRepeatStride, processSize);
                DataSlice txSrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * index) + scratchRepeatStride, processSize);
                DataSlice txDstSlice = DataSlice(remoteCclBuffAddr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff +
                        tempAlgParams.outputSliceStride * (xyMaxRankSize * index + threadIdx - xThreadNum_ + 1) + scratchRepeatStride, processSize);

                rxSrcSlices.emplace_back(rxSrcSlice);
                rxDstSlices.emplace_back(rxDstSlice);
                txSrcSlices.emplace_back(txSrcSlice);
                txDstSlices.emplace_back(txDstSlice);

                // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunSecondLevel] queId >= xQueNum myRank[%d] *****sendrecv*****, "
                //     "rxSrcSlice: %s, rxDstSlice: %s, txSrcSlice: %s, txDstSlice: %s", myRank_,
                //     rxSrcSlices.back().Describe().c_str(), rxDstSlices.back().Describe().c_str(),
                //     txSrcSlices.back().Describe().c_str(), txDstSlices.back().Describe().c_str());
            }
        }
        if (processSize == 0) {
            continue;
        }
        std::vector<std::vector<DataSlice>> allSliceVec = {rxSrcSlices, rxDstSlices, txSrcSlices, txDstSlices};
        CHK_RET(SendRecvProcess(channels, allSliceVec, threads, remoteRank, threadIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh2D::RunSecondReduce(
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[InsTempReduceScatterMesh2D][RunSecondReduce] myRank[%d] rpt[%u]", myRank_, tempAlgParams.repeatNum);
    u32 xyMaxRankSize = std::max(xRankSize_, yRankSize_);
    u64 processSize = 0;
    // 这里的stream 0和stream xRankSize-1分别负责后一半数据与前一半数据的本地reduce
    // 后一半数据做local reduce,由这部分的第一个stream做
    processSize = tempAlgParams.sliceSize - halfDataSize_;
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * tempAlgParams.repeatNum) +
            tempAlgParams.outputSliceStride * (xyMaxRankSize * xRankSize_ * rpt);
        for (u32 dataIdx = 0; dataIdx < xRankSize_; dataIdx++) {
            DataSlice srcSecDataSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff +
                    (xyMaxRankSize * xAlgRankId_ + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize,
                    processSize / DATATYPE_SIZE_TABLE[dataType_]);
            u64 outOffset = tempAlgParams.buffInfo.outBuffBaseOff + halfDataSize_ + rpt * tempAlgParams.outputRepeatStride;
            DataSlice dstSecDataSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,   // BufferType::OUTPUT,
                outOffset,
                processSize,
                processSize / DATATYPE_SIZE_TABLE[dataType_]);
            // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunSecondReduce] myRank[%d] queId < xQueNum *****LocalReduce*****, "
            //     "srcDataSlice: %s, dstDataSlice: %s", myRank_, srcSecDataSlice.Describe().c_str(),
            //     dstSecDataSlice.Describe().c_str());
            if (srcSecDataSlice.addr_ != dstSecDataSlice.addr_ || 
                srcSecDataSlice.offset_ != dstSecDataSlice.offset_ || 
                srcSecDataSlice.size_ != dstSecDataSlice.size_) {
                if (dataIdx == 0) {
                    CHK_RET(LocalCopy(threads[0], srcSecDataSlice, dstSecDataSlice));
                } else {
                    CHK_RET(LocalReduce(threads[0], srcSecDataSlice, dstSecDataSlice, dataType_, reduceOp_));
                }
            }
        }
    }
    // 前一半数据做local reduce，由这部分的第一个stream做
    processSize = halfDataSize_;
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; rpt++) {
        u64 scratchRepeatStride = tempAlgParams.outputSliceStride * (xyMaxRankSize * yRankSize_ * rpt);
        bool hasInplace = false;
        std::vector<DataSlice> srcFirDataSlices;
        std::vector<DataSlice> dstFirDataSlices;
        for (u32 dataIdx = 0; dataIdx < yRankSize_; dataIdx++) {
            DataSlice srcFirDataSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff +
                    (xyMaxRankSize * yAlgRankId_ + dataIdx) * tempAlgParams.outputSliceStride + scratchRepeatStride,
                    processSize,
                    processSize / DATATYPE_SIZE_TABLE[dataType_]);
            u64 outOffset = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
            DataSlice dstFirDataSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,   // BufferType::OUTPUT,
                outOffset, processSize, processSize / DATATYPE_SIZE_TABLE[dataType_]);
            // HCCL_DEBUG("[InsTempReduceScatterMesh2D][RunSecondReduce] myRank[%d] queId >= xQueNum *****LocalReduce*****, "
            //     "srcDataSlice: %s, dstDataSlice: %s", myRank_, srcFirDataSlice.Describe().c_str(),
            //     dstFirDataSlice.Describe().c_str());
            if (srcFirDataSlice.addr_ != dstFirDataSlice.addr_ || 
                srcFirDataSlice.offset_ != dstFirDataSlice.offset_ || 
                srcFirDataSlice.size_ != dstFirDataSlice.size_) {
#if 1
                srcFirDataSlices.push_back(srcFirDataSlice);
                dstFirDataSlices.push_back(dstFirDataSlice);
#else
                if (dataIdx == 0) {
                    CHK_RET(LocalCopy(threads[xThreadNum_], srcFirDataSlice, dstFirDataSlice));
                } else {
                    CHK_RET(LocalReduce(threads[xThreadNum_], srcFirDataSlice, dstFirDataSlice, dataType_, reduceOp_));
                }
#endif
            } else {
                hasInplace = true;
            }
        }
        for (u32 dataIdx = 0; dataIdx < srcFirDataSlices.size(); dataIdx++) {
            if (!hasInplace && dataIdx == 0) {
                CHK_RET(LocalCopy(threads[xThreadNum_], srcFirDataSlices[dataIdx], dstFirDataSlices[dataIdx]));
            } else {
                CHK_RET(LocalReduce(threads[xThreadNum_], srcFirDataSlices[dataIdx], dstFirDataSlices[dataIdx], dataType_, reduceOp_));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterMesh2D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = (xRankSize_ > 1 && yRankSize_ > 1) ? (xThreadNum_ + yThreadNum_): 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempReduceScatterMesh2D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = (xRankSize_ > 1 && yRankSize_ > 1) ? (xThreadNum_ + yThreadNum_): 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

u64 InsTempReduceScatterMesh2D::GetThreadNum()
{
    return xThreadNum_ + yThreadNum_;
}

} // namespace Hccl