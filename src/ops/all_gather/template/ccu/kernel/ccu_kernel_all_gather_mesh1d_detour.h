/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_ALL_GATHER_MESH_1D_DETOUR_H
#define HCCLV2_CCU_KERNEL_ALL_GATHER_MESH_1D_DETOUR_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllGatherMeshDetour1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherMeshDetour1D(std::vector<uint64_t> dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks, 
                                                    uint64_t singleChannelSize, uint64_t detourPathNum, 
                                                    uint64_t pathNumPerPeer, std::vector<u32> &channelsIndexVec)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          singleChannelSize_(singleChannelSize),
          detourPathNum_(detourPathNum),
          pathNumPerPeer_(pathNumPerPeer),
          channelsIndexVec_(channelsIndexVec)
    {
        HCCL_DEBUG("[CcuKernelArgAllGatherMeshDetour1D] dimSize: %lu, rankId: %u",
                   dimSize_, rankId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherMeshDetour1D", opParam_, subCommRanks_);
        return signature;
    }
    std::vector<uint64_t>                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    uint64_t singleChannelSize_;
    uint64_t detourPathNum_;
    uint64_t pathNumPerPeer_;
    std::vector<u32> channelsIndexVec_;
};

class CcuTaskArgAllGatherMeshDetour1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherMeshDetour1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        uint64_t baseOffset, uint64_t tailOffset, uint64_t tailSize, uint64_t loopIterNum,
        const std::vector<uint64_t> &lengths)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), baseOffset_(baseOffset), tailOffset_(tailOffset),
        tailSize_(tailSize), loopIterNum_(loopIterNum), lengths_(lengths) 
    {
        HCCL_DEBUG("[CcuTaskArgAllGatherMeshDetour1D] inputAddr: %lu, outputAddr: %lu, baseOffset: %lu, offset: %lu, "
                   "tailOffset: %lu, tailSize: %lu, tailloopIterNumSize: %lu, lengths: %lu",
                   inputAddr_, outputAddr_, token_, baseOffset_, tailOffset, tailSize_, loopIterNum_, lengths_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t baseOffset_;
    uint64_t tailOffset_;
    uint64_t tailSize_;  // 尾块数据量
    uint64_t loopIterNum_;
    std::vector<uint64_t> lengths_;  // 每个loop迭代一次搬运时每个ms上的数据量
};

class CcuKernelAllGatherMeshDetour1D : public CcuKernelAlgBase {
public:
    CcuKernelAllGatherMeshDetour1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherMeshDetour1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    void ProcessChannels(const std::vector<ChannelHandle> &channels);
    void AllocDetourRes();
    void CreateMultiOpBroadcastDetour();
    void GroupBroadcastDetour(std::vector<CcuRep::Variable> &lengths, std::vector<CcuRep::LocalAddr> &src,
        std::vector<CcuRep::RemoteAddr> &dst);
    void FirstStep();
    void SecondStep();

    // kernel args
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint64_t singleChannelSize_{0};  // 每个loop单次传输的总数据量，通信域级别
    uint64_t detourPathNum_{0};
    uint64_t pathNumPerPeer_{0};  // 到每个rank有几个transport，包括重复的
    std::vector<std::vector<ChannelHandle>> detourChannels_;  // 默认是transport的每rankSize-1个为一组，第一组是直连链路

    // task args
    CcuRep::Variable input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable baseOffset_;  // rankId*sliceSize，待广播分片的基础偏移
    CcuRep::Variable tailOffset_;  // 尾块相对偏移，singleTransportSize*128*iterNum
    CcuRep::Variable loopIterNum_;  // for detour loopgroup only
    std::vector<CcuRep::Variable> lengths_;  // 每组transport对应一个len
    GroupOpSize groupOpSize_;  // 只处理尾块
    std::vector<u32> channelsIndexVec_;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H
