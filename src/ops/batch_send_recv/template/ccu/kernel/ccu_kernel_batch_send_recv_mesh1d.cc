/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_batch_send_recv_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;
constexpr int BSR_OUTPUT_XN_ID = 1;
constexpr int BSR_TOKEN_XN_ID  = 2;
constexpr int BSR_CKE_IDX_0    = 0; // pre sync
constexpr int BSR_CKE_IDX_1    = 1; // post sync
constexpr int BSR_CONST_ONE    = 1;

CcuKernelBatchSendRecvMesh1D::CcuKernelBatchSendRecvMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgBatchSendRecvMesh1D *kernelArg =
        dynamic_cast<const CcuKernelArgBatchSendRecvMesh1D *>(&arg);
    peerCount_ = kernelArg->actualPeerCount_;
    selfIdx_ = kernelArg->selfIdx_;
    channels_ = kernelArg->channels;
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] peerCount_ = %lu, selfIdx_ = %u",
        peerCount_, selfIdx_);
}

HcclResult CcuKernelBatchSendRecvMesh1D::InitResource()
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] InitResource!");
    if (channels_.empty() && (peerCount_ > 1 || selfIdx_ == UINT32_MAX)) {
        HCCL_ERROR("[CcuKernelBatchSendRecvMesh1D] channels is empty but has remote peers!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint16_t channelIdx = 0;
    for (uint64_t i = 0; i < peerCount_; i++) {
        if (i == selfIdx_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
            src_.push_back(CreateLocalAddr());
            myDst_ = CreateLocalAddr();
            dst_.push_back({});
        } else {
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], BSR_OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], BSR_TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            src_.push_back(CreateLocalAddr());
            dst_.push_back(CreateRemoteAddr());
            channelIdx++;
        }
    }

    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();

    selfBit_ = (selfIdx_ != UINT32_MAX) ? (1 << selfIdx_) : 0;
    allBit_ = (1 << peerCount_) - 1;

    completedPeerCount_ = CreateVariable();
    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();
    xnConst1_ = CreateVariable();

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelBatchSendRecvMesh1D::LoadArgs()
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] LoadArgs!");
    // 加载 token（self 的 token）
    if (selfIdx_ != UINT32_MAX) {
        Load(token_[selfIdx_]);
    } else {
        CcuRep::Variable dummyToken = CreateVariable();
        Load(dummyToken);
    }
    Load(srcOffset_);
    Load(dstOffset_);
    Load(xnMaxTransportGoSize_);

    // 加载每个 peer 的发送接收信息
    sendRecvInfo_.resize(peerCount_);
    for (uint64_t i = 0; i < peerCount_; i++) {
        LoadPeerSendRecvInfo(sendRecvInfo_[i]);
    }
}

void CcuKernelBatchSendRecvMesh1D::LoadPeerSendRecvInfo(PeerSendRecvInfo &info)
{
    info.tailSize = CreateVariable();
    info.loopNum = CreateVariable();
    info.sendAddr = CreateVariable();
    info.recvAddr = CreateVariable();
    info.tailGoSize = CreateGroupOpSize();
    Load(info.tailSize);
    Load(info.loopNum);
    Load(info.sendAddr);
    Load(info.recvAddr);
    Load(info.tailGoSize);
}

void CcuKernelBatchSendRecvMesh1D::PreSync()
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] PreSync!");
    // 将本 rank 的 recvAddr 写到对端的 output_ Variable 中，交换接收地址
    uint16_t allBit = 1 << BSR_OUTPUT_XN_ID | 1 << BSR_TOKEN_XN_ID;
    uint16_t channelIdx = 0;
    for (uint64_t i = 0; i < peerCount_; i++) {
        if (i == selfIdx_) {
            continue;
        }
        // 将本 rank 对 peer i 的 recvAddr 写到 peer i 的 output_ Variable
        NotifyRecord(channels_[channelIdx], BSR_CKE_IDX_0, BSR_OUTPUT_XN_ID,
            sendRecvInfo_[i].recvAddr, 1 << BSR_OUTPUT_XN_ID);
        // 将本 rank 的 token 写到 peer i 的 token_ Variable
        if (selfIdx_ != UINT32_MAX) {
            NotifyRecord(channels_[channelIdx], BSR_CKE_IDX_0, BSR_TOKEN_XN_ID,
                token_[selfIdx_], 1 << BSR_TOKEN_XN_ID);
        }
        channelIdx++;
    }

    for (auto &ch : channels_) {
        NotifyWait(ch, BSR_CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] PreSync end");
}

void CcuKernelBatchSendRecvMesh1D::CalcSrcDst()
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] CalcSrcDst!");
    for (uint64_t i = 0; i < peerCount_; i++) {
        // src 使用本地 sendAddr
        src_[i].token = (selfIdx_ != UINT32_MAX) ? token_[selfIdx_] : token_[i];
        src_[i].addr = sendRecvInfo_[i].sendAddr;
        src_[i].addr += srcOffset_;

        if (i == selfIdx_) {
            // self: dst 使用本地 recvAddr
            myDst_.token = token_[selfIdx_];
            myDst_.addr = sendRecvInfo_[i].recvAddr;
            myDst_.addr += dstOffset_;
        } else {
            // 非 self: dst 使用 PreSync 交换得到的对端 recvAddr
            dst_[i].token = token_[i];
            dst_[i].addr = output_[i];
            dst_[i].addr += dstOffset_;
        }
    }
}

void CcuKernelBatchSendRecvMesh1D::DoSendRecvLoop()
{
    HCCL_DEBUG("[CcuKernelBatchSendRecvMesh1D] DoSendRecvLoop start");
    xnMaxTransportSize_ = UB_MAX_TRANS_SIZE;
    completedPeerCount_ = 0;
    xnConst1_ = 1;
    uint32_t channelId = 0;

    CCU_WHILE(completedPeerCount_ != peerCount_) {
        channelId = 0;
        // 先处理非 self 的远端 peer
        for (uint64_t i = 0; i < peerCount_; i++) {
            if (i == selfIdx_) {
                continue;
            }
            event_.SetMask(1 << i);

            CCU_IF(sendRecvInfo_[i].loopNum == UINT64_MAX) {
                RecordEvent(event_);
            }
            CCU_IF(sendRecvInfo_[i].loopNum != UINT64_MAX) {
                CCU_IF(sendRecvInfo_[i].loopNum == UINT64_MAX - 1) {
                    CCU_IF(sendRecvInfo_[i].tailSize == 0) {
                        RecordEvent(event_);
                    }
                    CCU_IF(sendRecvInfo_[i].tailSize != 0) {
                        WriteNb(channels_[channelId], dst_[i], src_[i],
                            sendRecvInfo_[i].tailSize, event_);
                    }
                    completedPeerCount_ += xnConst1_;
                }
                CCU_IF(sendRecvInfo_[i].loopNum != UINT64_MAX - 1) {
                    WriteNb(channels_[channelId], dst_[i], src_[i],
                        xnMaxTransportSize_, event_);
                    src_[i].addr += xnMaxTransportSize_;
                    dst_[i].addr += xnMaxTransportSize_;
                }
                sendRecvInfo_[i].loopNum += xnConst1_;
            }
            channelId++;
        }

        // 处理 self peer（使用 GroupCopy + RecordEvent）
        if (selfIdx_ != UINT32_MAX) {
            event_.SetMask(1 << selfIdx_);
            CCU_IF(sendRecvInfo_[selfIdx_].loopNum == UINT64_MAX) {
                RecordEvent(event_);
            }
            CCU_IF(sendRecvInfo_[selfIdx_].loopNum != UINT64_MAX) {
                CCU_IF(sendRecvInfo_[selfIdx_].loopNum == UINT64_MAX - 1) {
                    CCU_IF(sendRecvInfo_[selfIdx_].tailSize == 0) {
                        RecordEvent(event_);
                    }
                    CCU_IF(sendRecvInfo_[selfIdx_].tailSize != 0) {
                        GroupCopy(myDst_, src_[selfIdx_], sendRecvInfo_[selfIdx_].tailGoSize);
                        RecordEvent(event_);
                    }
                    completedPeerCount_ += xnConst1_;
                }
                CCU_IF(sendRecvInfo_[selfIdx_].loopNum != UINT64_MAX - 1) {
                    GroupCopy(myDst_, src_[selfIdx_], xnMaxTransportGoSize_);
                    RecordEvent(event_);
                    src_[selfIdx_].addr += xnMaxTransportSize_;
                    myDst_.addr += xnMaxTransportSize_;
                }
                sendRecvInfo_[selfIdx_].loopNum += xnConst1_;
            }
        }

        // 等待本轮所有 peer 完成
        event_.SetMask(allBit_);
        WaitEvent(event_);
    }
}

void CcuKernelBatchSendRecvMesh1D::PostSync()
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] PostSync!");
    for (auto &ch : channels_) {
        NotifyRecord(ch, BSR_CKE_IDX_1, 1 << BSR_CONST_ONE);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, BSR_CKE_IDX_1, 1 << BSR_CONST_ONE);
    }
}

HcclResult CcuKernelBatchSendRecvMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] Algorithm run.");

    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    CalcSrcDst();
    DoSendRecvLoop();
    PostSync();

    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] Algorithm end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelBatchSendRecvMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("[CcuKernelBatchSendRecvMesh1D] GeneArgs!");
    const CcuTaskArgBatchSendRecvMesh1D *taskArg =
        dynamic_cast<const CcuTaskArgBatchSendRecvMesh1D *>(&arg);

    uint64_t token = taskArg->token_;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    std::vector<uint64_t> processReturn = {token, srcOffset, dstOffset};

    // xnMaxTransportGoSize
    uint64_t xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize);
    for (auto val : xnMaxTransportGoSize) {
        processReturn.push_back(val);
    }

    // 每个 peer 的 {tailSize, loopNum, sendAddr, recvAddr, tailGoSize}
    uint64_t peerCount = taskArg->peerSendInfo_.size();
    for (uint64_t i = 0; i < peerCount; i++) {
        uint64_t sendAddr = taskArg->peerSendInfo_[i].sendAddr;
        uint64_t sendLen = taskArg->peerSendInfo_[i].sendLen;
        uint64_t recvAddr = taskArg->peerRecvAddr_[i];

        uint64_t tailSize = 0;
        uint64_t loopNum = UINT64_MAX; // 默认已完成（无数据）
        if (sendLen > 0) {
            tailSize = sendLen % UB_MAX_TRANS_SIZE;
            loopNum = UINT64_MAX - 1 - (sendLen / UB_MAX_TRANS_SIZE);
        }

        auto tailGoSize = CalGoSize(tailSize);
        processReturn.push_back(tailSize);
        processReturn.push_back(loopNum);
        processReturn.push_back(sendAddr);
        processReturn.push_back(recvAddr);
        for (auto val : tailGoSize) {
            processReturn.push_back(val);
        }

        HCCL_DEBUG("[CcuKernelBatchSendRecvMesh1D][GeneArgs] peer[%lu] sendAddr[%lu] "
            "sendLen[%lu] recvAddr[%lu] tailSize[%lu] loopNum[%lu]",
            i, sendAddr, sendLen, recvAddr, tailSize, loopNum);
    }

    return processReturn;
}

} // namespace ops_hccl
