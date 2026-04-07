/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D

#include <unordered_map>
#include <vector>
#include <ios>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_task_param_v1.h"
#include "ccu_rep_context_v1.h"

namespace ops_hccl {
using namespace hcomm;

// 为ReduceMesh1D实现的CCUIns、CCUKernelArg与CCUTaskArg
class CcuKernelArgReduceScatterMesh1D : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterMesh1D(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceMesh1D] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceScatterMesh1D : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMesh1D(uint64_t inputAddr, uint64_t outputAddr,
                                           uint64_t sliceSize, uint64_t offset,
                                           uint64_t token, void* fastLaunchCtxPtr = nullptr) :
        inputAddr_(inputAddr),
        outputAddr_(outputAddr),
        sliceSize_(sliceSize),
        offset_(offset),
        token_(token),
        FastLaunchCtxPtr_(fastLaunchCtxPtr)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterMesh1D] inputAddr: %lu, outputAddr: %lu, sliceSize: %lu, offset: %lu, FastLaunchCtxPtr: %p",
                   inputAddr_, outputAddr_, sliceSize_, offset_, FastLaunchCtxPtr_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t offset_;
    uint64_t token_;
    void* FastLaunchCtxPtr_ = nullptr; // 用于fast launch时传递非标准参数
};

class CcuKernelReduceScatterMesh1D : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterMesh1D(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterMesh1D() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;
    HcclResult GeneTaskParam(const CcuTaskArg &arg, std::vector<CcuTaskParam> &taskParams) override;
    HcclResult GetCcuProfilingInfo(const CcuTaskArg &arg, std::vector<CcuProfilingInfo> &allCcuProfilingInfo) override;
private:
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable offset_;
    GroupOpSize groupOpSize_;
    std::unordered_map<void*, std::vector<CcuTaskParam>> cache;
    std::unordered_map<void*, std::vector<CcuProfilingInfo>> cache1;
};
} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D