/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

// 为ReduceMesh1D实现的CCUIns、CCUKernelArg与CCUTaskArg
class CcuKernelArgReduceScatterMesh1D : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterMesh1D(uint64_t dimSize, uint32_t rankId, const OpParam& opParam, uint32_t axisId,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          axisId_(axisId),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterMesh1D] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d, axisId: %u",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType, axisId_);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    uint32_t                                axisId_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceScatterMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize, uint64_t offset,
        uint64_t token) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), die0Size_(die0Size), die1Size_(die1Size), offset_(offset), token_(token)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterMesh1D] inputAddr: %lu, outputAddr: %lu, die0Size: %lu, die1Size: %lu, offset: %lu",
                   inputAddr_, outputAddr_, die0Size_, die1Size_, offset_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t die0Size_;
 	uint64_t die1Size_;
    uint64_t offset_;
    uint64_t token_;
};

class CcuKernelReduceScatterMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterMesh1D(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    GroupOpSize groupOpSize_;
    CcuRep::Variable die0Size_;
    uint32_t axisId_{0};
};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D