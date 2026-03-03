/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_aicpu_reduce_nhr.h"

namespace ops_hccl {
InsTempReduceScatterAicpuReduceNHR::InsTempReduceScatterAicpuReduceNHR(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempReduceScatterAicpuReduceNHR::~InsTempReduceScatterAicpuReduceNHR()
{
}

HcclResult InsTempReduceScatterAicpuReduceNHR::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                            AlgResourceRequest& resourceRequest) 
{
    // NHR 需要的 que Num 为 1
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR][CalcRes] slaveThreadNum: [%u], notifyNumOnMainThread: [%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::GetRes(AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceNHR::GetThreadNum()
{
    return 1;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::KernelRun(const OpParam& param,
                                              const TemplateDataParams& tempAlgParams,
                                              const TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR] KernelRun start");

    tempAlgParams_       = tempAlgParams;
    channels_            = templateResource.channels;
    dataType_            = param.DataDes.dataType;
    
    // Step 1: 本地拷贝，将本rank的数据拷贝到自己hcclBuffer上相应的位置
    CHK_RET(LocalDataCopy(templateResource.threads));

    if (templateRankSize_ <= 1) {
        // 单卡场景，直接拷贝到输出
        CHK_RET(PostLocalReduce(templateResource.threads));
        return HcclResult::HCCL_SUCCESS;
    }
    
    // Step 2: AllGather，每个rank通过write方式将其他rank需要的数据写到目的rank的hcclbuffer
    CHK_RET(RunAllGather(templateResource.threads));
    if (dataType_ == HCCL_DATA_TYPE_INT64 || dataType_ == HCCL_DATA_TYPE_UIN || dataType_ == HCCL_DATA_TYPE_FP64
        || reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }
    // Step 3: 在hcclbuffer上做aicpureduce（使用PostLocalReduce函数完成规约和拷贝）
    CHK_RET(PostLocalReduce(templateResource.threads));
    
    HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    CHK_PRT_RET(threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR][LocalDataCopy] empty threads"), HcclResult::HCCL_E_INTERNAL);
    
    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    
    // 获取本rank在算法中的索引
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    
    // 把属于自己的slice从input拷贝到hcclBuff的对应位置
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        // 计算input上的偏移：使用inputSliceStride计算本rank的数据偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                              rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.inputSliceStride * myAlgIdx;
        
        // 计算hcclBuffer上的偏移
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                                rpt * tempAlgParams_.outputRepeatStride;
        const u64 scOff = scratchBase + tempAlgParams_.sliceSize * myAlgIdx;

        DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff, tempAlgParams_.sliceSize, tempAlgParams_.count);

        HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR][LocalDataCopy] rpt[%u] inOff[%llu] scOff[%llu] sliceSize[%llu]",
            rpt, inOff, scOff, tempAlgParams_.sliceSize);

        // 如果源地址和目标地址相同，则不需要做拷贝
        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.hcclBuffType || inOff != scOff) { 
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}


HcclResult InsTempReduceScatterAicpuReduceNHR::PostLocalReduce(const std::vector<ThreadHandle> &threads)
{
    CHK_PRT_RET(threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR][PostLocalReduce] empty threads"), HcclResult::HCCL_E_INTERNAL);

    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    ThreadHandle q = threads[0];

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff
                             + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff
                              + rpt * tempAlgParams_.outputRepeatStride;

        // 本rank需要的数据在hcclBuffer中的起始位置
        const u64 myScratchOff = scratchBase + tempAlgParams_.sliceSize * myAlgIdx;

        HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR][PostLocalReduce] rpt[%u] myAlgIdx[%u] myScratchOff[%llu] outBaseOff[%llu]",
            rpt, myAlgIdx, myScratchOff, outBaseOff);

        // 对于ReduceScatter，本rank需要规约所有rank的对应slice数据
        // 首先将本rank的数据从hcclBuffer拷贝到output
        DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, myScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff, tempAlgParams_.sliceSize, tempAlgParams_.count);

        CHK_RET(LocalCopy(q, srcSlice, dstSlice));

        // 然后将其余rank的数据规约到output
        for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
            if (rankIdx == myAlgIdx) {
                continue;  // 跳过本rank，已经拷贝过了
            }

            const u64 otherScratchOff = scratchBase + tempAlgParams_.sliceSize * rankIdx;
            DataSlice reduceSrcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, otherScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
            DataSlice reduceDstSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff, tempAlgParams_.sliceSize, tempAlgParams_.count);

            HCCL_INFO("[InsTempReduceScatterAicpuReduceNHR][PostLocalReduce] reduce from rank[%u] otherScratchOff[%llu]",
                rankIdx, otherScratchOff);

            // 使用AicpuReduce进行规约
            CHK_RET(LocalReduce(q, reduceSrcSlice, reduceDstSlice, dataType_, reduceOp_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterAicpuReduceNHR::RunAllGather(const std::vector<ThreadHandle> &threads)
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_);
    
    // 获取本rank在算法中的索引
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    
    for (u32 rpt = 0; rpt < rptNum; ++rpt) {
        // 计算本次repeat的基地址
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + 
                              rpt * tempAlgParams_.inputRepeatStride;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + 
                                rpt * tempAlgParams_.outputRepeatStride;

        for (u32 step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));

            const ChannelInfo &channelRecv = channels_.at(GetRankFromMap(stepInfo.fromRank))[0];
            const ChannelInfo &channelSend = channels_.at(GetRankFromMap(stepInfo.toRank))[0];
            
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            
            void *sendCclBuffAddr = channelSend.remoteCclMem.addr;
            void *recvCclBuffAddr = channelRecv.remoteCclMem.addr;

            HCCL_DEBUG("[InsTempReduceScatterAicpuReduceNHR][RunAllGather] rank[%d] step[%u/%u] recvFrom[%u] sendTo[%u] nSlices[%u]",
                myRank_, step, nSteps, stepInfo.fromRank, stepInfo.toRank, stepInfo.nSlices);

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];  // 要发送的slice索引
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];  // 要接收的slice索引

                // 发送：从本rank的input上读取txIdx对应的数据，发送到对方hcclBuffer的txIdx位置
                // 注意：对于ReduceScatter，数据在input上是按rank分布的，每个rank需要发送其他rank需要的数据
                const u64 txInOff = inBaseOff + tempAlgParams_.inputSliceStride * txIdx;
                const u64 txScratchOff = scratchBase + tempAlgParams_.sliceSize * txIdx;

                // 接收：从对方hcclBuffer的rxIdx位置接收数据，写入本rank hcclBuffer的rxIdx位置
                const u64 rxScratchOff = scratchBase + tempAlgParams_.sliceSize * rxIdx;

                txSrcSlices.emplace_back(tempAlgParams_.buffInfo.inputPtr, txInOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
                txDstSlices.emplace_back(sendCclBuffAddr, txScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
                rxSrcSlices.emplace_back(recvCclBuffAddr, rxScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
                rxDstSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff, tempAlgParams_.sliceSize, tempAlgParams_.count);
                
                HCCL_DEBUG("[InsTempReduceScatterAicpuReduceNHR] i[%u] txIdx[%u] rxIdx[%u] txInOff[%llu] txScratchOff[%llu] rxScratchOff[%llu]",
                    i, txIdx, rxIdx, txInOff, txScratchOff, rxScratchOff);
            }
            
            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            TxRxChannels sendRecvChannels(channelSend, channelRecv);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]),
                        HCCL_ERROR("[InsTempReduceScatterAicpuReduceNHR] SendRecvWrite failed (step=%u, rpt=%u)", step, rpt),
                        HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterAicpuReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    HCCL_INFO(
        "[InsTempReduceScatterAicpuReduceNHR][CalcScratchMultiple] templateScratchMultiplier[%llu]", templateRankSize_);
    return templateRankSize_;
}

u32 InsTempReduceScatterAicpuReduceNHR::GetRankFromMap(const u32 algRankIdx)
{
    return subCommRanks_[0].at(algRankIdx);
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceScatterAicpuReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
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

void InsTempReduceScatterAicpuReduceNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    (void)notifyIdxMianToSub;
}

void InsTempReduceScatterAicpuReduceNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    (void)notifyIdxSubToMain;
}
}
