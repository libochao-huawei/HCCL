/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_nhr_dpu.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "hcomm_primitives.h"

namespace ops_hccl {
constexpr u32 DPU_TIMEOUT = 180000;

InsTempAllGatherNHRDPU::InsTempAllGatherNHRDPU(const OpParam& param, const uint32_t rankId,
                                               const std::vector<std::vector<uint32_t>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks) {}

HcclResult InsTempAllGatherNHRDPU::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                           AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempAllGatherNHRDPU][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNHRDPU::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO(
        "[InsTempAllGatherNHRDPU][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempAllGatherNHRDPU::KernelRun(const OpParam& param,
                                             const TemplateDataParams& tempAlgParams,
                                             TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHRDPU] Run Start");

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherNHRDPU] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(LocalDataCopy(tempAlgParams, templateResource));

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempAllGatherNHRDPU";
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
    HCCL_INFO("HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalCopy(tempAlgParams, templateResource));

    HCCL_INFO("[InsTempAllGatherNHRDPU] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::DPUKernelRun(const TemplateDataParams& tempAlgParams,
                                                const std::map<u32, std::vector<ChannelInfo>>& channels,
                                                const u32 myRank,
                                                const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(tempAlgParams, channels));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo &stepInfo) const
{
    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    uint32_t deltaRank = 1 << (nSteps - 1 - step);
    uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    uint32_t nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    uint32_t deltaSliceIndex = 1 << (nSteps - step);
    uint32_t txSliceIdx = rankIdx;
    uint32_t rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[InsTempAllGatherNHRDPU][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllGatherNHRDPU::GetRankFromMap(const uint32_t rankIdx) const
{
    return subCommRanks_[0].at(rankIdx);
}

HcclResult InsTempAllGatherNHRDPU::LocalDataCopy(const TemplateDataParams& tempAlgParams,
                                                 const TemplateResource& templateResource)
{
    uint32_t algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));

    for (uint64_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        const u64 inOff = tempAlgParams.inputSliceStride * algRankIdx + inBaseOff;
        const u64 scOff = tempAlgParams.sliceSize * algRankIdx + scratchBaseoff;

        DataSlice srcSlices(tempAlgParams.buffInfo.inputPtr, inOff, tempAlgParams.sliceSize, tempAlgParams.count);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, tempAlgParams.sliceSize,
                           tempAlgParams.count);
        LocalCopy(templateResource.threads[0], srcSlices, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::RunNHR(const TemplateDataParams& tempAlgParams,
                                          const std::map<u32, std::vector<ChannelInfo>>& channels) const
{
#ifndef AICPU_COMPILE
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const uint64_t scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const uint64_t scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (uint32_t step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));

            HCCL_DEBUG("[InsTempAllGatherNHRDPU] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
                myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

            auto rxChannel = channels.at(GetRankFromMap(stepInfo.fromRank));
            auto txChannel = channels.at(GetRankFromMap(stepInfo.toRank));

            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            void *sendCclBuffAddr = txChannel[0].remoteCclMem.addr;
            void *recvCclBuffAddr = rxChannel[0].remoteCclMem.addr;

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];

                const u64 txScratchOff = scratchBase + tempAlgParams.sliceSize * txIdx;
                const u64 rxScratchOff = scratchBase + tempAlgParams.sliceSize * rxIdx;

                txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txScratchOff, tempAlgParams.sliceSize,
                                         tempAlgParams.count);
                txDstSlices.emplace_back(sendCclBuffAddr, txScratchOff, tempAlgParams.sliceSize, tempAlgParams.count);
                rxSrcSlices.emplace_back(recvCclBuffAddr, rxScratchOff, tempAlgParams.sliceSize, tempAlgParams.count);
                rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxScratchOff, tempAlgParams.sliceSize,
                                         tempAlgParams.count);
            }
            // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
            // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
            if (txChannel[0].remoteRank == rxChannel[0].remoteRank) {
                const ChannelInfo& ch = txChannel[0];
                const u32 sliceNum = txSrcSlices.size();

                // 前同步阶段：完成ACK握手
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, ch.handle, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, ch.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));

                // 数据写操作阶段：集中下发所有写操作
                for (u32 i = 0; i < sliceNum; ++i) {
                    const DataSlice& srcSlice = txSrcSlices[i];
                    const DataSlice& dstSlice = txDstSlices[i];
                    void* dst = static_cast<void*>(static_cast<s8*>(dstSlice.addr_) + dstSlice.offset_);
                    void* src = static_cast<void*>(static_cast<s8*>(srcSlice.addr_) + srcSlice.offset_);
                    CHK_RET(static_cast<HcclResult>(
                        HcommWriteWithNotifyNbiOnThread(0, ch.handle, dst, src, srcSlice.size_, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, ch.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));

                    // 接收侧对应信号（同一channel）
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, ch.handle, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, ch.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
                }

                // 后同步阶段：完成FIN_ACK和Fence
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, ch.handle, NOTIFY_IDX_FIN_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, ch.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, ch.handle)));

                CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
            } else {
                const ChannelInfo& txCh = txChannel[0];
                const ChannelInfo& rxCh = rxChannel[0];
                const u32 sliceNum = txSrcSlices.size();

                // 前同步阶段：先完成所有方向的ACK握手
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, txCh.handle, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));

                // 数据写操作阶段：集中下发所有写操作
                for (u32 i = 0; i < sliceNum; ++i) {
                    const DataSlice& srcSlice = txSrcSlices[i];
                    const DataSlice& dstSlice = txDstSlices[i];
                    void* dst = static_cast<void*>(static_cast<s8*>(dstSlice.addr_) + dstSlice.offset_);
                    void* src = static_cast<void*>(static_cast<s8*>(srcSlice.addr_) + srcSlice.offset_);
                    CHK_RET(static_cast<HcclResult>(
                        HcommWriteWithNotifyNbiOnThread(0, txCh.handle, dst, src, srcSlice.size_, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));

                    // 接收侧对应信号
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL)));
                    CHK_RET(static_cast<HcclResult>(
                        HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
                }

                // 后同步阶段：完成FIN_ACK和Fence
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, txCh.handle, NOTIFY_IDX_FIN_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, txCh.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh.handle)));

                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, rxCh.handle, NOTIFY_IDX_FIN_ACK)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, rxCh.handle, NOTIFY_IDX_FIN_ACK, DPU_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, rxCh.handle)));

                CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
            }
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPU::PostLocalCopy(const TemplateDataParams& tempAlgParams,
                                                 const TemplateResource& templateResource)
{
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams.outputSliceStride * algRank + outBaseOff;
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, tempAlgParams.sliceSize,
                               tempAlgParams.count);
            DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, outOffset, tempAlgParams.sliceSize,
                               tempAlgParams.count);
            LocalCopy(templateResource.threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNHRDPU", InsTempAllGatherNHRDPU);
}