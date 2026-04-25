/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_NHR1D_MULTI_JETTY_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_NHR1D_MULTI_JETTY_MEM2MEM_H

#include <vector>
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using CcuOmniPipeNHRStepInfo = struct CcuOmniPipeNHRStepInfoDef {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices = 0;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;
};


class CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
        uint64_t rankSize, uint32_t rankId, const OpParam &opParam,
        uint32_t jettyNum, const CcuOmniPipeNHRStepInfo &stepInfo,
        uint32_t toChannelIdx, uint32_t fromChannelIdx,
        const std::vector<std::vector<uint32_t>> &subCommRanks)
        : rankSize_(rankSize), rankId_(rankId), jettyNum_(jettyNum), opParam_(opParam),
          stepInfo_(stepInfo), toChannelIdx_(toChannelIdx), fromChannelIdx_(fromChannelIdx), subCommRanks_(subCommRanks)
    {
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override;

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t jettyNum_{0};
    OpParam opParam_{};
    CcuOmniPipeNHRStepInfo stepInfo_;
    uint32_t toChannelIdx_{0};
    uint32_t fromChannelIdx_{0};
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t srcOffset, uint64_t dstOffset,
        uint64_t sliceSize, uint64_t sliceSizePerJetty, uint64_t lastSliceSizePerJetty,
        uint64_t doStepPreSync, uint64_t doStepPostSync)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), srcOffset_(srcOffset), dstOffset_(dstOffset),
          sliceSize_(sliceSize), sliceSizePerJetty_(sliceSizePerJetty),
          lastSliceSizePerJetty_(lastSliceSizePerJetty), doStepPreSync_(doStepPreSync),
          doStepPostSync_(doStepPostSync)
    {
    }

    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t token_{0};
    uint64_t srcOffset_{0};
    uint64_t dstOffset_{0};
    uint64_t sliceSize_{0};
    uint64_t sliceSizePerJetty_{0};
    uint64_t lastSliceSizePerJetty_{0};
    uint64_t doStepPreSync_{0};
    uint64_t doStepPostSync_{0};
};

class CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem : public CcuKernelAlgBase {
public:
    explicit CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherOmniPipeNHR1DMultiJettyMem2Mem() override = default;

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResources();
    HcclResult LoadArgs();
    HcclResult PreSync();
    HcclResult PostSync();
    HcclResult DoOmniPipeNHRSendRecv();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint32_t localSize_{0};
    uint32_t myRankIdx_{0};
    uint32_t jettyNum_{0};
    CcuOmniPipeNHRStepInfo stepInfo_;
    uint32_t toChannelIdx_{0};
    uint32_t fromChannelIdx_{0};
    std::vector<ChannelHandle> channels_;

    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable sliceSizePerJetty_;
    hcomm::CcuRep::Variable lastSliceSizePerJetty_;
    hcomm::CcuRep::Variable doStepPreSync_;
    hcomm::CcuRep::Variable doStepPostSync_;
    hcomm::CcuRep::LocalAddr srcMem_;
    hcomm::CcuRep::RemoteAddr dstMem_;
    hcomm::CcuRep::LocalAddr srcMemTmp_;
    hcomm::CcuRep::RemoteAddr dstMemTmp_;
    hcomm::CcuRep::CompletedEvent event_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_NHR1D_MULTI_JETTY_MEM2MEM_H
