/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_OPS_SCATTER_TEMPLATE_CCU_KERNEL_CCU_KERNEL_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H_
#define HCCL_SRC_OPS_SCATTER_TEMPLATE_CCU_KERNEL_CCU_KERNEL_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H_

#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {
using namespace hcomm;

using NHRStepInfo = struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0) {}
};

class CcuKernelArgScatterOmniPipeNHR1DMem2Mem : public CcuKernelArg {
public:
    CcuKernelArgScatterOmniPipeNHR1DMem2Mem(
        const uint32_t rankId,
        const uint32_t dimSize,
        const uint32_t rootId,
        const uint32_t axisId,
        const uint32_t axisSize,
        const std::vector<NHRStepInfo>& stepInfoVector,
        const std::map<uint32_t, uint32_t>& rank2ChannelIdx,
        const OpParam& opParam,
        const std::vector<std::vector<uint32_t>>& subCommRanks)
        : rankId_(rankId),
        dimSize_(dimSize),
        rootId_(rootId),
        axisId_(axisId),
        axisSize_(axisSize),
        stepInfoVector_(stepInfoVector),
        rank2ChannelIdx_(rank2ChannelIdx),
        opParam_(opParam),
        subCommRanks_(subCommRanks)
    {}

    uint32_t rankId_;
    uint32_t dimSize_;
    uint32_t rootId_;
    uint32_t axisId_;
    uint32_t axisSize_;
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<uint32_t, uint32_t> rank2ChannelIdx_;
    std::vector<ChannelHandle> channels;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgScatterOmniPipeNHR1DMem2Mem : public CcuTaskArg {
public:
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t die0Size_;
    uint64_t die1Size_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
    uint64_t repeatNum_;
    uint64_t die0TailSize_;
    uint64_t die1TailSize_;
    uint64_t isInputOutputEqual_;
    uint64_t sliceSize_;
    uint64_t inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride_;
};

class CcuKernelScatterOmniPipeNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelScatterOmniPipeNHR1DMem2Mem(const CcuKernelArg& arg);
    ~CcuKernelScatterOmniPipeNHR1DMem2Mem() override = default;

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;

protected:
    HcclResult InitResource();
    HcclResult LoadArgs();
    void PreSync();
    void PostSync();
    void DoScatterOmniPipeNHR();
    void DoScatterOmniPipeNHRSingleStep(const NHRStepInfo& nhrStepInfo);
    void DoSendRecvSlice(const u32& toRank, CcuRep::LocalAddr& src, CcuRep::RemoteAddr& dst, bool isLastSlice);
    void DoLocalCopyNb(CcuRep::LocalAddr& dst, CcuRep::LocalAddr& src, CcuRep::Variable& sliceSize,
        CcuRep::CompletedEvent& event_);
    void DoWriteNb(ChannelHandle& sendChannel, CcuRep::RemoteAddr& dst,
        CcuRep::LocalAddr& src, CcuRep::Variable& sliceSize, CcuRep::CompletedEvent& event_);

private:
    uint32_t rankId_;
    uint32_t dimSize_;
    uint32_t rootId_;
    uint32_t axisId_;
    uint32_t axisSize_;
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<uint32_t, uint32_t> rank2ChannelIdx_;
    std::vector<uint32_t> subCommRanks_;
    std::vector<ChannelHandle> channels_;
    HcclDataType dataType_;

    uint32_t localSize_;
    uint32_t myRankIdx_;

    // Variables
    CcuRep::Variable input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> token_;
    std::vector<CcuRep::Variable> outputRemote_;

    CcuRep::Variable die0Size_;
    CcuRep::Variable die1Size_;
    CcuRep::Variable inputSliceStride_;
    CcuRep::Variable outputSliceStride_;
    CcuRep::Variable inputRepeatStride_;
    CcuRep::Variable outputRepeatStride_;
    CcuRep::Variable repeatNumVar_;
    CcuRep::Variable repeatNumVarTemp_;
    CcuRep::Variable repeatTimeflag_;
    CcuRep::Variable cursliceSize_;
    CcuRep::Variable isInputOutputEqual_;
    CcuRep::Variable die0TailSize_;
    CcuRep::Variable die1TailSize_;
    CcuRep::Variable inputOmniPipeSliceStride_;
    CcuRep::Variable outputOmniPipeSliceStride_;

    // Address offsets
    std::vector<CcuRep::Variable> inputOffset_;
    std::vector<CcuRep::Variable> outputOffset_;

    // Memory objects
    CcuRep::LocalAddr srcMem_;
    CcuRep::LocalAddr dstMem_;
    CcuRep::RemoteAddr dstRemoteMem_;
    CcuRep::CompletedEvent event_;
};
}  // namespace ops_hccl

#endif  // HCCL_SRC_OPS_SCATTER_TEMPLATE_CCU_KERNEL_CCU_KERNEL_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H_
