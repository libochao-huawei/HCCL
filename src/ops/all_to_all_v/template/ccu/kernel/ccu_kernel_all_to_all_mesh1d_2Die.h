/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_1D_2DIE_H_
#define HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_1D_2DIE_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllToAllMesh1D2Die : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllToAllMesh1D2Die(uint64_t rankSize, uint32_t rankId, bool withMyRank,  
            std::vector<uint32_t> &rankIdGroup, u32 bitNum, const std::vector<std::vector<uint32_t>>& subCommRanks, const OpParam& opParam) 
        :   rankSize_(rankSize), 
            rankId_(rankId), 
            withMyRank_(withMyRank), 
            rankIdGroup_(rankIdGroup),
            subCommRanks_(subCommRanks),
            bitNum_(bitNum) 
        {
            HCCL_DEBUG("[CcuKernelArgAllToAllMesh1D2Die] dimSize: %lu, rankId: %u, withMyRank: %u, bitNum: %u",
                   dimSize_, rankId_, withMyRank_, bitNum_);
        }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllToAllMesh1D2Die", opParam_, subCommRanks_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint64_t rankSize_;
    uint32_t rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    bool withMyRank_;
    std::vector<uint32_t>                   rankIdGroup_;
    u32 bitNum_;
};

class CcuTaskArgAllToAllMesh1D2Die : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllMesh1D2Die(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
        uint64_t token, uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t outBuffBaseOff) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize),
        token_(token), inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),
        outBuffBaseOff_(outBuffBaseOff)
        {
            HCCL_DEBUG("[CcuTaskArgAlltoAllMesh1D] inputAddr: %lu, outputAddr: %lu, sliceSize: %lu, inputSliceStride: %lu, "
            "outputSliceStride: %lu, outBuffBaseOff: %lu",
            inputAddr_, outputAddr_, sliceSize_, inputSliceStride_, outputSliceStride_, outBuffBaseOff_);
        }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t token_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t outBuffBaseOff_;
};

class CcuKernelAllToAllMesh1D2Die : public CcuKernelAlgBase {
public:
    CcuKernelAllToAllMesh1D2Die(const CcuKernelArg &arg, const std::vector<CcuTransport*> &transports,
                              const CcuTransportGroup &group);
    ~CcuKernelAllToAllMesh1D2Die() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    uint16_t virRankSize{0};
    uint64_t logicRankSize{0};
    uint16_t selfBit{0};
    uint16_t allBit{0};
    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    bool withMyRank_ = true;  // 发数据是否包含本rank
    std::vector<RankId> rankGroup_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable inputSliceStride_;
    hcomm::CcuRep::Variable outputoffset_;
    hcomm::CcuRep::Variable outBuffBaseOff_;
    GroupOpSize groupOpSize_;
    uint32_t signalNum_; //需要使用的signal数量
    uint16_t bitNumPerCKE_;
    void InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    uint32_t CalcDstRank(uint32_t peerId) const;
    void DoRepeatAllToAll();

    hcomm::CcuRep::CompletedEvent event_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_TO_ALL_MESH_1D_2DIE_H_