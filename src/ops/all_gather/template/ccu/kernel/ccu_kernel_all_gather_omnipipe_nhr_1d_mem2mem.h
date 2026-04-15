/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_NHR_1D_H
#define HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_NHR_1D_H

#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
// #include "ccu_datatype.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};

class CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem(
                                            uint64_t dimSize, uint32_t rankId,
                                            const std::vector<NHRStepInfo> stepInfoVector,
                                            const std::map<u32, u32> rank2ChannelIdx, const OpParam& opParam,
                                            const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
    std::vector<NHRStepInfo>                stepInfoVector_;    // nhr每一步的信息（发送/接受给谁，发/收哪片数据）
    std::map<u32, u32>                      rank2ChannelIdx_;
};

class CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                               uint64_t sendCount, uint64_t sliceSize, uint64_t repeatNum, uint64_t sliceStride,
                                               uint64_t localCopyFlag,
                                               const std::vector<uint64_t> inputOmniPipeSliceStride,
                                               const std::vector<uint64_t> outputOmniPipeSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sendCount_(sendCount), repeatNum_(repeatNum),
          sliceStride_(sliceStride), sliceSize_(sliceStride), localCopyFlag_(localCopyFlag),
          inputOmniPipeSliceStride_(inputOmniPipeSliceStride), outputOmniPipeSliceStride_(outputOmniPipeSliceStride)
    {
        HCCL_DEBUG("[CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem] inputAddr: %lu, outputAddr: %lu, token: %lu, "
                   "sendCount: %lu, sliceSize: %lu, repeatNum: %lu, sliceStride: %lu, localCopyFlag: %lu ",
                   inputAddr_, outputAddr_, token_, sendCount_, sliceSize_, repeatNum_, sliceStride_, localCopyFlag_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t sendCount_;
    uint64_t sliceSize_;
    uint64_t sliceStride_;
    uint64_t localCopyFlag_;
    uint64_t repeatNum_;
    std::vector<uint64_t> inputOmniPipeSliceStride_;
    std::vector<uint64_t> outputOmniPipeSliceStride_;
};

class CcuKernelAllGatherOmniPipeNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelAllGatherOmniPipeNHR1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherOmniPipeNHR1DMem2Mem() override {}

    HcclResult Algorithm();
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatAllGatherNHR();
    void DoRepeatAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo, const std::vector<CcuRep::Variable> &outputSliceOffset);

    // 构造函数中
    uint32_t rankId_{0};
    uint32_t rankIdx_{0};
    uint64_t rankSize_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    uint32_t myRankIdx_{0};
    uint32_t signalNum_{0};  // 需要使用的signal数量
    uint64_t dimSize_;

    HcclReduceOp reduceOp_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    std::vector<NHRStepInfo> stepInfoVector_;   // nhr算法执行过程中的参数
    std::map<u32, u32> rank2ChannelIdx_;
    uint32_t       axisSize_{0};

    hcomm::CcuRep::Variable repeatNum_;
    hcomm::CcuRep::Variable sliceStride_;
    hcomm::CcuRep::Variable localCopyFlag_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sendCount_;
    hcomm::CcuRep::Variable sliceSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    GroupOpSize groupOpSize_;
    std::vector<hcomm::CcuRep::Variable> inputOmniPipeSliceStride_;
    std::vector<hcomm::CcuRep::Variable> outputOmniPipeSliceStride_;
    CcuRep::CompletedEvent event_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_NHR_1D_H
