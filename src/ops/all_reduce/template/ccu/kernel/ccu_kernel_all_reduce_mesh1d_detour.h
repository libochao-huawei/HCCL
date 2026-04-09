/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_DETOUR_H
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_DETOUR_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllReduceMesh1DDetour : public CcuKernelArg {
public:
    explicit CcuKernelArgAllReduceMesh1DDetour(const std::vector<uint64_t> &dSize, uint32_t rId, const OpParam &op,
        const std::vector<std::vector<u32>> &tempVTopo, uint64_t singleTransferSize, uint64_t detourPathNum,
        uint64_t pathNumPerPeer, std::vector<u32> &channelsIndexVec) :
        dimSize_(dSize), rankId_(rId), op_(op), tempVTopo_(tempVTopo), singleTransferSize_(singleTransferSize),
        detourPathNum_(detourPathNum), pathNumPerPeer_(pathNumPerPeer), channelsIndexVec_(channelsIndexVec){}
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllReduceMesh1DDetour", op_, tempVTopo_);
        return signature;
    }
    std::vector<uint64_t> dimSize_;
    uint32_t rankId_;
    OpParam op_;
    std::vector<std::vector<u32>> tempVTopo_;
    uint64_t singleTransferSize_;
    uint64_t detourPathNum_;
    uint64_t pathNumPerPeer_;
    std::vector<u32> channelsIndexVec_;
};

class CcuTaskArgAllReduceMesh1DDetour : public CcuTaskArg {
public:
    explicit CcuTaskArgAllReduceMesh1DDetour(uint64_t inputAddr, uint64_t outputAddr, uint64_t offset,
        uint64_t token, uint64_t iterNum, uint64_t tailOffset, uint64_t tailSize, const std::vector<uint64_t> &lengths) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), offset_(offset), token_(token), iterNum_(iterNum), tailOffset_(tailOffset),
        tailSize_(tailSize), lengths_(lengths) {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t offset_;
    uint64_t token_;
    uint64_t iterNum_;
    uint64_t tailOffset_;
    uint64_t tailSize_;
    std::vector<uint64_t> lengths_;
};


class CcuKernelAllReduceMesh1DDetour : public CcuKernelAlgBase {
public:
    CcuKernelAllReduceMesh1DDetour(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllReduceMesh1DDetour() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
private:
    void CreateMultiOpReduceDetour(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType);
    void GroupReduceDetour(std::vector<CcuRep::RemoteAddr> &src, std::vector<CcuRep::LocalAddr> &dst,
        HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType);
    void CreateMultiOpBroadcastDetour();
    void GroupBroadcastDetour(std::vector<CcuRep::Variable> &lengths,
        std::vector<CcuRep::LocalAddr> &src, std::vector<CcuRep::RemoteAddr> &dst);
    void ReduceScatterFirstStep();
    void ReduceScatterSecondStep();
    void AllGatherFirstStep();
    void AllGatherSecondStep();
    void InitResources();
    void PreSync();
    void PostSync();
    void CreateResource(uint32_t msInterleave);
    void DoLocalReduce(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType, 
                       std::vector<CcuRep::RemoteAddr> src, 
                       std::vector<std::vector<CcuRep::CcuBuf>> &bufs, std::vector<CcuRep::Variable> &lengths, 
                       std::vector<CcuRep::CompletedEvent> &sems);
    void CreateReduceLoop(HcclDataType &dataType, HcclDataType &outputDataType, HcclReduceOp &opType);
    void CreateBroadcastLoop();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    uint64_t singleTransferSize{0};  // 每个loop单次传输的总数据量，通信域级别
    uint64_t detourPathNum{0};
    uint64_t pathNumPerPeer{0};  // 到每个rank有几个transport，包括重复的
    std::vector<std::vector<ChannelHandle>> detourChannels_;
    CcuRep::Variable tailOffset_;  // 尾块相对偏移，singleTransferSize*128*iterNum

    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    CcuRep::Variable iterNum_;
    CcuRep::Variable tailSize_;
    GroupOpSize groupOpSize_;
    std::vector<CcuRep::Variable> lengths_;
};
}

#endif
