/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_MESH_1D
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_MESH_1D

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

// 为ReduceMesh1D实现的CCUIns、CCUKernelArg与CCUTaskArg
class CcuKernelArgReduceScatterOmniPipeMesh1D : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterOmniPipeMesh1D(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[%s]dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   __func__, dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterOmniPipeMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceScatterOmniPipeMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterOmniPipeMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
        uint64_t offSet, uint64_t token, uint64_t repeatNum, uint64_t localCopyFlag, uint64_t inputSliceStride,
        uint64_t outputSliceStride, std::vector<uint64_t> inputOmniPipeSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), offSet_(offSet), token_(token),
          repeatNum_(repeatNum), localCopyFlag_(localCopyFlag), inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride), inputOmniPipeSliceStride_(inputOmniPipeSliceStride)
    {
        HCCL_DEBUG("[%s]inputAddr[%lu] outputAddr[%lu] sliceSize[%lu] offSet[%lu] repeatNum[%lu] localCopyFlag[%lu]",
                   __func__, inputAddr_, outputAddr_, sliceSize_, offSet_, repeatNum_, localCopyFlag_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t offSet_;
    uint64_t token_;
    uint64_t repeatNum_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t localCopyFlag_;
    std::vector<uint64_t> inputOmniPipeSliceStride_;
};

class CcuKernelReduceScatterOmniPipeMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterOmniPipeMesh1D(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterOmniPipeMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatReduceScatter();
    void DoLocalCopy();

    uint64_t rankSize_{0}; // templateRankSize_
    uint32_t rankId_{0};
    uint32_t userRank_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    // std::vector<CcuRep::Variable> output_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offSet_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable inputSliceStride_;
    CcuRep::Variable outputSliceStride_;
    CcuRep::Variable repeatNum_;
    // CcuRep::Variable sendCount_;
    CcuRep::Variable localCopyFlag_;
    std::vector<CcuRep::Variable> inputOmniPipeSliceStride_;
    GroupOpSize groupOpSize_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    CcuRep::CompletedEvent event_;

};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_MESH_1D