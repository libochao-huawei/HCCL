/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_H
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
using namespace hcomm;
class CcuKernelArgGatherOmniPipeMesh1D : public CcuKernelArg {
public:
    explicit CcuKernelArgGatherOmniPipeMesh1D(uint64_t dimSize, uint32_t rankId, uint32_t rootId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgGatherOmniPipeMesh1D] dimSize: %lu, rankId: %u, rootId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, rootId_, opParam.reduceType, opParam.DataDes.dataType);

        for (int i = 0; i < subCommRanks_.size(); i++) {
            for (int j = 0; j < subCommRanks_[i].size(); j++) {
                HCCL_DEBUG("kernel arg myRank_ is %u, subCommRanks_[%d][%d] is %llu", rankId_, i, j, subCommRanks_[i][j]);
            }
        }
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgGatherOmniPipeMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    uint32_t                                rootId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgGatherOmniPipeMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgGatherOmniPipeMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        uint64_t sliceSize, uint64_t sliceStride, uint64_t localCopyFlag,
        uint64_t inputOmniPipeSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize),
          sliceStride_(sliceStride), localCopyFlag_(localCopyFlag), inputOmniPipeSliceStride_(inputOmniPipeSliceStride)
    {
        HCCL_DEBUG("[%s] inputAddr:%lu outputAddr:%lu token:%lu "
                   "sliceSize:%lu sliceStride:%lu localCopyFlag:%lu inputOmniPipeSliceStride:%lu",
            __func__, inputAddr_, outputAddr_, token_, sliceSize_, sliceStride_, localCopyFlag_,
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

class CcuKernelGatherOmniPipeMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelGatherOmniPipeMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelGatherOmniPipeMesh1D() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

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
    CcuRep::Variable inputOmniPipeSliceStride_;
    CcuRep::CompletedEvent event_;
};

}
#endif