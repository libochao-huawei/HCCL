/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_BATCH_SEND_RECV_MESH1D_H
#define HCCL_CCU_KERNEL_BATCH_SEND_RECV_MESH1D_H

#include <vector>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "../ccu_temp_batch_send_recv_mesh_1D.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgBatchSendRecvMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgBatchSendRecvMesh1D(uint64_t actualPeerCount, uint32_t selfIdx,
        const OpParam &opParam, const std::vector<std::vector<uint32_t>> &subCommRanks)
        : actualPeerCount_(actualPeerCount),
          selfIdx_(selfIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgBatchSendRecvMesh1D] actualPeerCount: %lu, selfIdx: %u",
            actualPeerCount_, selfIdx_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgBatchSendRecvMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t actualPeerCount_;
    uint32_t selfIdx_;  // self 在 actualPeers 中的索引，UINT32_MAX 表示无 self peer
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgBatchSendRecvMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgBatchSendRecvMesh1D(uint64_t token,
        const std::vector<PeerSendInfo> &peerSendInfo,
        const std::vector<uint64_t> &peerRecvAddr)
        : token_(token), peerSendInfo_(peerSendInfo), peerRecvAddr_(peerRecvAddr)
    {
        HCCL_DEBUG("[CcuTaskArgBatchSendRecvMesh1D] token: %lu, peerCount: %zu",
            token_, peerSendInfo_.size());
    }

    uint64_t token_;
    std::vector<PeerSendInfo> peerSendInfo_;
    std::vector<uint64_t> peerRecvAddr_;
};

class CcuKernelBatchSendRecvMesh1D : public CcuKernelAlgBase {
public:
    explicit CcuKernelBatchSendRecvMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelBatchSendRecvMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    // 每个 peer 的发送接收信息（从 SQE 加载）
    struct PeerSendRecvInfo {
        CcuRep::Variable tailSize;
        CcuRep::Variable loopNum;
        CcuRep::Variable sendAddr;
        CcuRep::Variable recvAddr;
        GroupOpSize tailGoSize;
    };

    uint64_t peerCount_{0};
    uint32_t selfIdx_{UINT32_MAX};
    std::vector<ChannelHandle> channels_;

    std::vector<CcuRep::Variable> output_;   // 对端交换后的 recv 地址
    std::vector<CcuRep::Variable> token_;
    std::vector<CcuRep::LocalAddr> src_;
    std::vector<CcuRep::RemoteAddr> dst_;
    CcuRep::LocalAddr myDst_;

    CcuRep::Variable srcOffset_;
    CcuRep::Variable dstOffset_;
    CcuRep::Variable completedPeerCount_;
    CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_;
    CcuRep::Variable xnConst1_;
    CcuRep::CompletedEvent event_;

    std::vector<PeerSendRecvInfo> sendRecvInfo_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};

    HcclResult InitResource();
    void LoadArgs();
    void LoadPeerSendRecvInfo(PeerSendRecvInfo &info);
    void PreSync();
    void CalcSrcDst();
    void DoSendRecvLoop();
    void PostSync();
};

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_BATCH_SEND_RECV_MESH1D_H
