/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgGatherOmniPipeMesh1DMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgGatherOmniPipeMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, uint32_t rootId,
                                                      const OpParam& opParam,
                                                      const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgGatherOmniPipeMesh1DMem2Mem] dimSize=%llu, rankId=%u, rootId=%u",
                   dimSize_, rankId_, rootId_);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgGatherOmniPipeMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgGatherOmniPipeMesh1DMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgGatherOmniPipeMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                    uint64_t sliceSize, uint64_t sliceStride,
                                                    uint64_t localCopyFlag, uint64_t inputOmniPipeSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token),
          sliceSize_(sliceSize), sliceStride_(sliceStride),
          localCopyFlag_(localCopyFlag), inputOmniPipeSliceStride_(inputOmniPipeSliceStride)
    {
        HCCL_DEBUG("[CcuTaskArgGatherOmniPipeMesh1DMem2Mem] inputAddr=%llu, outputAddr=%llu, token=%llu, "
                   "sliceSize=%llu, sliceStride=%llu, localCopyFlag=%llu, inputOmniPipeSliceStride=%llu",
                   inputAddr_, outputAddr_, token_, sliceSize_, sliceStride_, localCopyFlag_,
                   inputOmniPipeSliceStride_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t sliceStride_;
    uint64_t localCopyFlag_;
    uint64_t inputOmniPipeSliceStride_;
};

class CcuKernelGatherOmniPipeMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelGatherOmniPipeMesh1DMem2Mem(const hcomm::CcuKernelArg& arg);
    ~CcuKernelGatherOmniPipeMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg& arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatGather();

    uint64_t rankSize_{0};
    uint32_t userRank_{0};
    uint32_t rankIdx_{0};
    uint32_t rootIdx_{0};
    std::vector<std::vector<uint32_t>> subCommRanks_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    hcomm::CcuRep::Variable sliceStride_;
    hcomm::CcuRep::Variable localCopyFlag_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    GroupOpSize groupOpSize_;
    hcomm::CcuRep::Variable inputOmniPipeSliceStride_;
    hcomm::CcuRep::CompletedEvent event_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H