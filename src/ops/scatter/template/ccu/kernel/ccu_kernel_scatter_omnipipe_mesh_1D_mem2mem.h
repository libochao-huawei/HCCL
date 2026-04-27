/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgScatterOmniPipeMesh1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgScatterOmniPipeMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, uint32_t rootId,
        const OpParam& opParam, const std::vector<std::vector<uint32_t>>& subCommRanks, uint32_t currentStep, bool isSameAxisAsRoot, uint32_t totalStep)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          currentStep_(currentStep),
          isSameAxisAsRoot_(isSameAxisAsRoot),
          totalStep_(totalStep)
    {
        HCCL_DEBUG("[%s]dimSize: %lu, rankId: %u, dataType: %d rootId: %u currentStep: %u isSameAxisAsRoot: %d totalStep: %u", 
                   __func__, dimSize_, rankId_, opParam.DataDes.dataType, rootId_, currentStep_, isSameAxisAsRoot_, totalStep_);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgScatterOmniPipeMesh1DMem2Mem", 
            opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    uint32_t rootId_;                               // 根节点rank ID（在通信域中的rank ID）
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    uint32_t currentStep_ ;
    bool isSameAxisAsRoot_;
    uint32_t totalStep_;
};

class CcuTaskArgScatterOmniPipeMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgScatterOmniPipeMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr,
        uint64_t sliceSize, uint64_t offSet, uint64_t token, uint64_t localCopyFlag,
        uint64_t inputSliceStride, uint64_t outputSliceStride,
        uint64_t inputOmniPipeSliceStride, uint64_t outputOmniPipeSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), scratchAddr_(scratchAddr),
          sliceSize_(sliceSize), offSet_(offSet), token_(token), localCopyFlag_(localCopyFlag),
          inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),
          inputOmniPipeSliceStride_(inputOmniPipeSliceStride),
          outputOmniPipeSliceStride_(outputOmniPipeSliceStride)
    {
        HCCL_DEBUG("[%s]inputAddr[%lu] outputAddr[%lu] scratchAddr[%lu] sliceSize[%lu] offSet[%lu] "
                   "localCopyFlag[%lu]",
                   __func__, inputAddr_, outputAddr_, scratchAddr_, sliceSize_, offSet_, 
                   localCopyFlag_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t sliceSize_;
    uint64_t offSet_;
    uint64_t token_;
    uint64_t localCopyFlag_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride_;
};

class CcuKernelScatterOmniPipeMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelScatterOmniPipeMesh1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelScatterOmniPipeMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatScatter();
    void DoScatter();
    void DoScatterFromScratch();
    void DoScatter2Out();
        
    void DoLocalCopy();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    uint32_t currentStep_{0};
    bool isSameAxisAsRoot_{false};
    uint32_t totalStep_{0};
    std::vector<std::vector<uint32_t>> subCommRanks_;  // 子通信域rank列表
    HcclDataType dataType_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offSet_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable inputSliceStride_;
    CcuRep::Variable outputSliceStride_;
    CcuRep::Variable inputOmniPipeSliceStride_;
    CcuRep::Variable outputOmniPipeSliceStride_;
    CcuRep::Variable localCopyFlag_;
    std::vector<CcuRep::LocalAddr> scratchMem_;
    std::vector<CcuRep::RemoteAddr> remoteScratchMem_;

    CcuRep::CompletedEvent event_;
    std::vector<CcuRep::LocalAddr> inputMem_;
    std::vector<CcuRep::RemoteAddr> outputMem_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM
