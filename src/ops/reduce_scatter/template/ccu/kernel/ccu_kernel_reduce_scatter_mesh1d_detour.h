/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DETOUR
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DETOUR

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgReduceScatterMeshDetour1D : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterMeshDetour1D(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    uint64_t singleTransportSize, uint64_t detourPathNum, uint64_t pathNumPerPeer,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          singleTransportSize_(singleTransportSize),
          detourPathNum_(detourPathNum),
          pathNumPerPeer_(pathNumPerPeer),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterMeshDetour1D] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterMeshDetour1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t dimSize_;
    uint32_t rankId_;
    uint64_t singleTransportSize_;
    uint64_t detourPathNum_;
    uint64_t pathNumPerPeer_;
    OpParam  opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgReduceScatterMeshDetour1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMeshDetour1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t offset, 
        uint64_t token, uint64_t iterNum, uint64_t tailOffset, uint64_t tailSize, const std::vector<uint64_t> &lengths) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), offset_(offset), token_(token), iterNum_(iterNum), tailOffset_(tailOffset),
        tailSize_(tailSize), lengths_(lengths)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterMeshDetour1D] inputAddr: %lu, outputAddr: %lu, offset: %lu, token: %lu, "
                   "iterNum: %lu, tailOffset: %lu, tailSize: %lu",
                   inputAddr_, outputAddr_, offset_, token_, iterNum_, tailOffset_, tailSize_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t offset_;
    uint64_t token_;
    uint64_t iterNum_;
    uint64_t tailOffset_;
    uint64_t tailSize_;
    std::vector<uint64_t> lengths_;
};

class CcuKernelReduceScatterMeshDetour1D : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterMeshDetour1D(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterMeshDetour1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    void LoadArgs();
    void PreSync();
    void PostSync();
    void AllocGoResourceDetour();
    HcclResult InitResource();
    HcclResult CreateMultiOpReduceDetour(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType);
    HcclResult GroupReduceDetour(std::vector<CcuRep::RemoteAddr> &src, std::vector<CcuRep::LocalAddr> &dst,
        HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType);
    void LoopForReduceDetour(std::vector<std::vector<CcuRep::CcuBuf>> &bufs,
        std::vector<CcuRep::RemoteAddr> &src, std::vector<CcuRep::LocalAddr> &dst,
        std::vector<CcuRep::Variable> &lengths, std::vector<CcuRep::CompletedEvent> &events,
        HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType);
    HcclResult GroupReduceForTailData();
    HcclResult GroupReduceDetourFunc();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint64_t singleTransportSize_{0};  // 每个loop单次传输的总数据量，通信域级别
    uint64_t detourPathNum_{0};
    uint64_t pathNumPerPeer_{0};        // 到每个rank有几个transport，包括重复的
    std::vector<ChannelHandle> channels_;
    std::vector<std::vector<ChannelHandle>> detourChannels_;

    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    CcuRep::Variable iterNum_;
    CcuRep::Variable tailSize_;
    CcuRep::Variable tailOffset_;
    GroupOpSize groupOpSize_;
    std::vector<CcuRep::Variable> lengths_;
};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DETOUR