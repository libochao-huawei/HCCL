/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_NHR_1D_OMNIPIPE_MEM2MEM_H
#define HCCL_CCU_KERNEL_GATHER_NHR_1D_OMNIPIPE_MEM2MEM_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using GatherNHRStepInfo = struct GatherNHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices = 0;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    GatherNHRStepInfoDef() : nSlices(0)
    {
    }
};

class CcuKernelArgGatherNHROmniPipe1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgGatherNHROmniPipe1D(uint64_t dimSize, uint32_t mySubCommRankId, uint32_t mySubCommRootId,
                                              uint32_t axisId, const std::vector<GatherNHRStepInfo>& stepInfoVector,
                                              const std::map<u32, u32>& rank2ChannelIdx, const OpParam& opParam,
                                              const std::vector<std::vector<uint32_t>>& subCommRanks, uint32_t axisSize)
        : dimSize_(dimSize),
          mySubCommRankId_(mySubCommRankId),
          mySubCommRootId_(mySubCommRootId),
          axisId_(axisId),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          axisSize_(axisSize)
    {
        HCCL_DEBUG("[CcuKernelArgGatherNHROmniPipe1D] dimSize=%llu, mySubCommRankId=%u, mySubCommRootId=%u, axisId=%u, axisSize=%u",
                   dimSize_, mySubCommRankId_, mySubCommRootId_, axisId_, axisSize_);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgGatherNHROmniPipe1D", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t dimSize_;
    uint64_t mySubCommRankId_;
    uint64_t mySubCommRootId_;
    uint64_t axisId_;
    std::vector<GatherNHRStepInfo> stepInfoVector_;
    std::map<u32, u32> rank2ChannelIdx_;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    uint32_t axisSize_;
};

class CcuTaskArgGatherNHROmniPipe1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgGatherNHROmniPipe1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                            uint64_t sliceStride, uint64_t localCopyFlag)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token),
          sliceStride_(sliceStride), localCopyFlag_(localCopyFlag)
    {
        HCCL_DEBUG("[CcuTaskArgGatherNHROmniPipe1D] inputAddr=%llu, outputAddr=%llu, token=%llu, sliceStride=%llu, localCopyFlag=%llu",
                   inputAddr_, outputAddr_, token_, sliceStride_, localCopyFlag_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t sliceStride_;
    uint64_t localCopyFlag_;
};

class CcuKernelGatherNHROmniPipe1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelGatherNHROmniPipe1DMem2Mem(const hcomm::CcuKernelArg& arg);
    ~CcuKernelGatherNHROmniPipe1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg& arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoGatherNHR();
    void DoGatherNHRSingleStep(const GatherNHRStepInfo& stepInfo);

    uint64_t rankSize_{0};
    uint32_t userRank_{0};
    uint32_t rankIdx_{0};
    uint32_t rootIdx_{0};
    uint32_t axisId_{0};
    uint32_t axisSize_{0};
    std::vector<std::vector<uint32_t>> subCommRanks_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    hcomm::CcuRep::Variable sliceStride_;
    hcomm::CcuRep::Variable localCopyFlag_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    GroupOpSize groupOpSize_;

    std::vector<GatherNHRStepInfo> stepInfoVector_;
    std::map<u32, u32> rank2ChannelIdx_;
    CcuRep::CompletedEvent event_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_NHR_1D_OMNIPIPE_MEM2MEM_H