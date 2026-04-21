/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_H
#define HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllGatherMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherMesh1D(uint64_t dimSize, uint32_t rankId, uint32_t axisId,
                                                    const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          axisId_(axisId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgAllGatherMesh1D] dimSize: %lu, rankId: %u, axisId: %u",
                   dimSize_, rankId_, axisId_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                axisId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgAllGatherMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                uint64_t offset,
                                                uint64_t die0Size, uint64_t die1Size,
                                                uint64_t die0LastSize, uint64_t die1LastSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), offset_(offset),
        die0Size_(die0Size), die1Size_(die1Size), die0LastSize_(die0LastSize), die1LastSize_(die1LastSize)
    {
        HCCL_DEBUG("[CcuTaskArgAllGatherMesh1D] inputAddr: %lu, outputAddr: %lu, offset: %lu, "
                   "die0Size: %lu, die1Size: %lu",
                   inputAddr_, outputAddr_, offset_, die0Size_, die1Size_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t offset_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t die0LastSize_;
    uint64_t die1LastSize_;
};

class CcuKernelAllGatherMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelAllGatherMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoAllGather();

    // CcuKernelAlgDataWrapper algWrapper;
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t axisId_{0};
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable die0LastSize_;
    CcuRep::Variable die1LastSize_;
    GroupOpSize groupOpSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    hcomm::CcuRep::LocalAddr src;
    std::vector<hcomm::CcuRep::RemoteAddr> dst;
};

}// namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_H
