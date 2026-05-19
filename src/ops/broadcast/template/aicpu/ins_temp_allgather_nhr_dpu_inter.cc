/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_allgather_nhr_dpu_inter.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "hcomm_primitives.h"

namespace ops_hccl {
constexpr u32 DPU_TIMEOUT = 180000;

InsTempAllGatherNHRDPUInter::InsTempAllGatherNHRDPUInter(const OpParam& param, const uint32_t rankId,
    const std::vector<std::vector<uint32_t>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks) {}

HcclResult InsTempAllGatherNHRDPUInter::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempAllGatherNHRDPUInter][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNHRDPUInter::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO(
        "[InsTempAllGatherNHRDPUInter][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempAllGatherNHRDPUInter::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHRDPUInter] Run Start");
        HCCL_DEBUG("[InsTempAllGatherNHRDPUInter][KernelRun] check myRank[%u]", myRank_);
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUInter] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }
    
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempAllGatherNHRDPUInter";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

    u32 sendMsgId = 0;
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
        static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempAllGatherNHRDPUInter] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPUInter::DPUKernelRun(const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    tempRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(tempAlgParams, channels, subCommRanks));
#endif
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherNHRDPUInter::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo &stepInfo)
{
    uint32_t rankIdx = 0;

    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;
 
    // 计算通信对象
    uint32_t deltaRank = 1 << (nSteps - 1 - step);
    uint32_t recvFrom = (rankIdx + tempRankSize_ - deltaRank) % tempRankSize_;
    uint32_t sendTo = (rankIdx + deltaRank) % tempRankSize_;
 
    // 数据份数和数据编号增量
    uint32_t nSlices = (tempRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    uint32_t deltaSliceIndex = 1 << (nSteps - step);
    uint32_t txSliceIdx = rankIdx;
    uint32_t rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + tempRankSize_) % tempRankSize_;
 
    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;
 
    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
 
        HCCL_DEBUG("[AllGatherNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
 
        txSliceIdx = (txSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
        rxSliceIdx = (rxSliceIdx + tempRankSize_ - deltaSliceIndex) % tempRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}
 
u32 InsTempAllGatherNHRDPUInter::GetRankFromMap(const uint32_t rankIdx)
{
    return subCommRanks_[0].at(rankIdx);
}
 
HcclResult InsTempAllGatherNHRDPUInter::LocalDataCopy(const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    uint32_t algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRankIdx);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRankIdx);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(algRankIdx);

    if (sliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    for (uint64_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
 
        const u64 inOff = inBaseOff + sliceOffset;
        const u64 scOff = scratchBaseoff + sliceOffset;
 
        DataSlice srcSlices(tempAlgParams.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, sliceSize, sliceCount);
        HCCL_DEBUG("[InsTempAllGatherNHRDPUInter][LocalCopy] LocalDataCopy RankID [%d] dataAlgRank[%d] "
            "srcOff[%d] dstOff[%d] sliceOffset[%d] sliceSize[%d].", myRank_, algRankIdx, inOff, scOff, sliceOffset,
            sliceSize);
        LocalCopy(templateResource.threads[0], srcSlices, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherNHRDPUInter::RunNHR(const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    const uint32_t nSteps = GetNHRStepNum(tempRankSize_);
 
    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const uint64_t scratchRepeatStride = tempAlgParams.sliceSize * tempRankSize_;
        const uint64_t scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        
        for (uint32_t step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));
 
            HCCL_DEBUG("[InsTempAllGatherNHRDPUInter] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u] GetRankFromMap[%u]",
                myRank_, tempRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices, GetRankFromMap(stepInfo.fromRank));

            auto rxChannel = channels.at(GetRankFromMap(stepInfo.fromRank));
            auto txChannel = channels.at(GetRankFromMap(stepInfo.toRank));
 
            void *sendCclBuffAddr = txChannel[0].remoteCclMem.addr;
            void *recvCclBuffAddr = rxChannel[0].remoteCclMem.addr;

            // 收集本 step 中所有非零 slice 的操作
            struct SliceOp {
                DataSlice txSrc, txDst, rxSrc, rxDst;
            };
            std::vector<SliceOp> bothOps;   // sendSize>0 && recvSize>0
            std::vector<SliceOp> sendOps;   // only send
            std::vector<SliceOp> recvOps;   // only recv

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];

                u64 sendOffset = tempAlgParams.allRankDispls.at(txIdx);
                u64 sendSize = tempAlgParams.allRankSliceSize.at(txIdx);
                u64 sendCount = tempAlgParams.allRankProcessedDataCount.at(txIdx);

                u64 recvOffset = tempAlgParams.allRankDispls.at(rxIdx);
                u64 recvSize = tempAlgParams.allRankSliceSize.at(rxIdx);
                u64 recvCount = tempAlgParams.allRankProcessedDataCount.at(rxIdx);

                const u64 txScratchOff = scratchBase + sendOffset;
                const u64 rxScratchOff = scratchBase + recvOffset;

                if (sendSize == 0 && recvSize == 0) {
                    continue;
                }

                DataSlice txSrc(tempAlgParams.buffInfo.hcclBuff.addr, txScratchOff, sendSize, sendCount);
                DataSlice txDst(sendCclBuffAddr, txScratchOff, sendSize, sendCount);
                DataSlice rxSrc(recvCclBuffAddr, rxScratchOff, recvSize, recvCount);
                DataSlice rxDst(tempAlgParams.buffInfo.hcclBuff.addr, rxScratchOff, recvSize, recvCount);

                if (sendSize > 0 && recvSize > 0) {
                    bothOps.push_back({txSrc, txDst, rxSrc, rxDst});
                } else if (sendSize > 0) {
                    sendOps.push_back({txSrc, txDst, rxSrc, rxDst}); // rx* ignored
                } else if (recvSize > 0) {
                    recvOps.push_back({txSrc, txDst, rxSrc, rxDst}); // tx* ignored
                }
            }

            const ChannelInfo& txCh = txChannel[0];
            const ChannelInfo& rxCh = rxChannel[0];
            const bool samePeer = (txCh.remoteRank == rxCh.remoteRank);
            const bool txFirst = (txCh.remoteRank < rxCh.remoteRank);

            // ========== 前同步阶段 ==========
            if (!bothOps.empty() || !sendOps.empty() || !recvOps.empty()) {
                if (samePeer) {
                    // SendRecv 模式：Record(recv) + Wait(send)
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));
                } else {
                    // 不同 peer：各自 ACK（与原 SendWrite/RecvWrite 一致）
                    if (!sendOps.empty() || !bothOps.empty()) {
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, txCh.handle, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));
                    }
                    if (!recvOps.empty() || !bothOps.empty()) {
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_ACK)));
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));
                    }
                }
            }

            // ========== 数据写操作阶段 ==========
            // both + send：Write(tx) + Wait(DATA)
            auto doSendWrite = [&](const SliceOp& op) {
                const DataSlice& src = op.txSrc;
                const DataSlice& dst = op.txDst;
                void* dstPtr = static_cast<void*>(static_cast<s8*>(dst.addr_) + dst.offset_);
                void* srcPtr = static_cast<void*>(static_cast<s8*>(src.addr_) + src.offset_);
                CHK_RET(static_cast<HcclResult>(
                    HcommWriteWithNotifyNbiOnThread(0, txCh.handle, dstPtr, srcPtr, src.size_, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
            };
            // recv：Record(recv, DATA) + Wait(recv, DATA)
            auto doRecvWrite = [&](const SliceOp& op) {
                const DataSlice& src = op.rxSrc; // not really used for recv only, but for consistency
                (void)src;
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
            };

            if (samePeer) {
                // samePeer: both + send + recv 都走 txCh / rxCh (实际同一)
                for (const auto& op : bothOps) doSendWrite(op);  // Write + Wait on sendCh, but Wait is on recvCh in real SendRecv
                // 注意：对于 samePeer，Wait(DATA) 应该在 recvCh 上，这里简化用 txCh 句柄（因为同一）
                // 实际 SendRecvWrite 中 Wait(DATA) 在 recvChannel 上
                // 为正确，这里需要区分
                // 重新实现 samePeer 的 data 阶段以匹配 SendRecvWrite
                for (const auto& op : bothOps) {
                    const DataSlice& src = op.txSrc;
                    const DataSlice& dst = op.txDst;
                    void* dstPtr = static_cast<void*>(static_cast<s8*>(dst.addr_) + dst.offset_);
                    void* srcPtr = static_cast<void*>(static_cast<s8*>(src.addr_) + src.offset_);
                    CHK_RET(static_cast<HcclResult>(
                        HcommWriteWithNotifyNbiOnThread(0, txCh.handle, dstPtr, srcPtr, src.size_, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
                }
                for (const auto& op : sendOps) {
                    const DataSlice& src = op.txSrc;
                    const DataSlice& dst = op.txDst;
                    void* dstPtr = static_cast<void*>(static_cast<s8*>(dst.addr_) + dst.offset_);
                    void* srcPtr = static_cast<void*>(static_cast<s8*>(src.addr_) + src.offset_);
                    CHK_RET(static_cast<HcclResult>(
                        HcommWriteWithNotifyNbiOnThread(0, txCh.handle, dstPtr, srcPtr, src.size_, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
                }
                for (const auto& op : recvOps) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
                }
            } else {
                // 不同 peer：根据 remoteRank 顺序
                // 如果 txFirst (tx.remote < rx.remote)：先做 send 方向，再 recv
                // 否则相反
                if (txFirst) {
                    for (const auto& op : bothOps) doSendWrite(op);
                    for (const auto& op : sendOps) doSendWrite(op);
                    for (const auto& op : recvOps) doRecvWrite(op);
                    for (const auto& op : bothOps) doRecvWrite(op);  // both 的 recv 部分
                } else {
                    for (const auto& op : recvOps) doRecvWrite(op);
                    for (const auto& op : bothOps) doRecvWrite(op);
                    for (const auto& op : bothOps) doSendWrite(op);
                    for (const auto& op : sendOps) doSendWrite(op);
                }
            }

            // ========== 后同步阶段 ==========
            if (!bothOps.empty() || !sendOps.empty() || !recvOps.empty()) {
                if (samePeer) {
                    // SendRecvWrite 后同步顺序
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, txCh.handle, NOTIFY_IDX_FIN_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_FIN_ACK)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                    CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh.handle)));
                    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
                } else {
                    // 各自独立 FIN + Fence
                    if (!sendOps.empty() || !bothOps.empty()) {
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, txCh.handle, NOTIFY_IDX_FIN_ACK)));
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                        CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh.handle)));
                    }
                    if (!recvOps.empty() || !bothOps.empty()) {
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_FIN_ACK)));
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                        CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, rxCh.handle)));
                    }
                    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
                }
            }
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPUInter::PostLocalCopy(const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    if (tempAlgParams.buffInfo.outputPtr == nullptr) {
        return HcclResult::HCCL_SUCCESS;
    }

    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            
            u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRank);
            u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRank);
            u64 sliceOffset = tempAlgParams.allRankDispls.at(algRank);

            u64 scratchOffset = scratchBase + sliceOffset;
            u64 outOffset = outBaseOff + sliceOffset;
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherNHRDPUInter][LocalCopy] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "srcOff[%d] dstOff[%d] sliceOffset[%d] sliceSize[%d].",
                       myRank_, rank, algRank, scratchOffset, outOffset, sliceOffset, sliceSize);
            LocalCopy(templateResource.threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNHRDPUInter", InsTempAllGatherNHRDPUInter);
}