/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using NHRStepInfoRS = struct NHRStepInfoDefRS {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;

    NHRStepInfoDefRS() : nSlices(0)
    {
    }
};

class CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem(uint64_t rankSize_, uint32_t rankId,
                                                const OpParam& opParam,
                                                const std::vector<NHRStepInfoRS> stepInfoVector,
                                                const std::map<uint32_t, uint32_t> rank2ChannelIdx,
                                                const std::vector<std::vector<uint32_t>>& subCommRanks)
        : rankSize_(rankSize_),
          rankId_(rankId),
          opParam_(opParam),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[%s] kernelArg: rankId_[%u], rankSize_[%u]", __func__, rankId_, rankSize_);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                rankSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<NHRStepInfoRS>                stepInfoVector_;
    std::map<uint32_t, uint32_t>            rank2ChannelIdx_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    // std::map<uint32_t, uint32_t>            subRankIdx2RankIdx_;
};

class CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        uint64_t sliceSize, uint64_t sliceStride, uint64_t localCopyFlag,
        uint64_t inputOmniPipeSliceStride, std::vector<uint64_t> inputOmniSliceStrideVec, uint64_t inputSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize),
          sliceStride_(sliceStride), localCopyFlag_(localCopyFlag), inputOmniPipeSliceStride_(inputOmniPipeSliceStride),
          inputOmniSliceStrideVec_(inputOmniSliceStrideVec), inputSliceStride_(inputSliceStride)
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
    std::vector<uint64_t> inputOmniSliceStrideVec_;
    uint64_t inputSliceStride_;
};

class CcuKernelReduceScatterOmniPipeNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterOmniPipeNHR1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelReduceScatterOmniPipeNHR1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    HcclResult LoadArgs();
    HcclResult PreSync();
    HcclResult PostSync();
    HcclResult DoRepeatReduceScatterNHR();
    HcclResult DoRepeatReduceScatterNHRSingleStep(const NHRStepInfoRS &nhrStepInfo);

    uint64_t rankSize_{0};
    uint32_t userRank_{0};
    uint32_t rankId_{0};
    uint32_t myRankIdx_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    std::vector<NHRStepInfoRS> stepInfoVector_;
    std::map<uint32_t, uint32_t> rank2ChannelIdx_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    hcomm::CcuRep::Variable sliceStride_;
    hcomm::CcuRep::Variable localCopyFlag_;
    std::vector<ChannelHandle> channels_;
    std::vector<hcomm::CcuRep::Variable> input_;
    hcomm::CcuRep::Variable output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    GroupOpSize groupOpSize_;
    CcuRep::Variable inputOmniPipeSliceStride_;
    CcuRep::CompletedEvent event_;
    // std::map<uint32_t, uint32_t> subRankIdx2RankIdx_;
    std::vector<hcomm::CcuRep::Variable> inputOmniSliceStrideVec_;
    hcomm::CcuRep::Variable inputSliceStride_;

    HcclReduceOp reduceOp_;
    HcclDataType dataType_;
};

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H