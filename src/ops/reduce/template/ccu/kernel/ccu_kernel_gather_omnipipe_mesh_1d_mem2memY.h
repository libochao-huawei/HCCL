/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEMY_H
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEMY_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgGatherOmniPipeMesh1DMem2MemY : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgGatherOmniPipeMesh1DMem2MemY(uint64_t dimSize, uint32_t rankId, uint32_t rootId,
                                                      const OpParam& opParam,
                                                      const std::vector<std::vector<uint32_t>>& subCommRanks, std::map<u32, u32> subRankIdx2RankIdx, bool ifRealRoot, uint32_t realrank)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          subRankIdx2RankIdx_(subRankIdx2RankIdx),
          ifRealRoot_(ifRealRoot),
          myrealrank_(realrank)
    {
        HCCL_DEBUG("[CcuKernelArgGatherOmniPipeMesh1DMem2MemY] dimSize=%llu, rankId=%u, rootId=%u",
                   dimSize_, rankId_, rootId_);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgGatherOmniPipeMesh1DMem2MemY", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    OpParam opParam_;
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    bool ifRealRoot_;
    uint32_t myrealrank_;
    
};

class CcuTaskArgGatherOmniPipeMesh1DMem2MemY : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgGatherOmniPipeMesh1DMem2MemY(uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t localCopyFlag,
                                                    uint64_t sliceSize,
                                                    uint64_t inputOmniPipeSliceStride, uint64_t outputOmniPipeSliceStride,
        bool isStepOne, bool isLastStep, bool ifNewRoot)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), localCopyFlag_(localCopyFlag),
          sliceSize_(sliceSize), inputOmniPipeSliceStride_(inputOmniPipeSliceStride), outputOmniPipeSliceStride_(outputOmniPipeSliceStride),
          isStepOne_(isStepOne),
          isLastStep_(isLastStep),
          ifNewRoot_(ifNewRoot)
    {
        HCCL_DEBUG("[CcuTaskArgGatherOmniPipeMesh1DMem2MemY] inputAddr=%llu, outputAddr=%llu, token=%llu, "
                   "sliceSize=%llu, localCopyFlag=%llu, inputOmniPipeSliceStride=%llu",
                   inputAddr_, outputAddr_, token_, sliceSize_, localCopyFlag_, inputOmniPipeSliceStride_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t localCopyFlag_;
    uint64_t sliceSize_;
    uint64_t inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride_;
    bool isStepOne_;
    bool isLastStep_;
    bool ifNewRoot_;
};

class CcuKernelGatherOmniPipeMesh1DMem2MemY : public CcuKernelAlgBase {
public:
    CcuKernelGatherOmniPipeMesh1DMem2MemY(const hcomm::CcuKernelArg& arg);
    ~CcuKernelGatherOmniPipeMesh1DMem2MemY() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg& arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatGather();
    void DoGatherIn2Out();
    void DoGatherScratch2Out();
    void DoGather();

    uint64_t rankSize_{0};
    uint32_t userRank_{0};
    uint32_t myrealrank_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    hcomm::CcuRep::Variable localCopyFlag_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    GroupOpSize groupOpSize_;
    hcomm::CcuRep::Variable inputOmniPipeSliceStride_;
    hcomm::CcuRep::Variable outputOmniPipeSliceStride_;
    HcclReduceOp reduceOp_;
    CcuRep::Variable isStepOne_;
    CcuRep::Variable isLastStep_;
    CcuRep::Variable ifNewRoot_;
    bool ifRealRoot_{false};

    hcomm::CcuRep::CompletedEvent event_;
    std::vector<CcuRep::RemoteAddr> inputMem_;
    std::vector<CcuRep::LocalAddr> outputMem_;
    std::vector<CcuRep::LocalAddr> scratchMem_;
    std::vector<CcuRep::RemoteAddr> remoteScratchMem_;
    std::vector<CcuRep::RemoteAddr> remoteInputhMem_;

    // hcomm::CcuRep::CompletedEvent event_;
    // std::vector<CcuRep::LocalAddr> inputMem_;
    // std::vector<CcuRep::RemoteAddr> outputMem_;
    // std::vector<CcuRep::LocalAddr> scratchMem_;
    // std::vector<CcuRep::RemoteAddr> remoteScratchMem_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEMY_H