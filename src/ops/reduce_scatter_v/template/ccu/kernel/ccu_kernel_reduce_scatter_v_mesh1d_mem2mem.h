/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_V_MESH_1D_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_V_MESH_1D_MEM2MEM

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgReduceScatterVMesh1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterVMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, const OpParam& opParam,
                                                    const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterVMesh1DMem2Mem] dimSize: %lu, rankId: %u, reduceOp: %d, dataType: %d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterVMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                                dimSize_;
    uint32_t                                rankId_;
    OpParam                                 opParam_;
    std::vector<std::vector<uint32_t>>      subCommRanks_;
};

class CcuTaskArgReduceScatterVMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterVMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                    uint64_t scratchAddr, uint64_t scratchInterval,
                                                    uint64_t sliceSize, uint64_t offset)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), scratchAddr_(scratchAddr),
          scratchInterval_(scratchInterval), sliceSize_(sliceSize), offset_(offset)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterVMesh1DMem2Mem] inputAddr: %lu, outputAddr: %lu, scratchAddr: %lu, "
                   "scratchInterval: %lu, sliceSize: %lu, offset: %lu",
                   inputAddr_, outputAddr_, scratchAddr_, scratchInterval, sliceSize, offset);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t scratchAddr_;
    uint64_t scratchInterval_;
    uint64_t sliceSize_;
    uint64_t offset_;
};

class CcuKernelReduceScatterVMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelReduceScatterVMesh1DMem2Mem(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterVMesh1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatReduceScatterV();
    void DoReduceScatterV();
    void PrepareReduceScatterVData(std::vector<CcuRep::RemoteAddr>& reduceScatterVSrc,
        std::vector<CcuRep::LocalAddr>& reduceScatterVDst);
    void CollectAllRanksSlice(std::vector<CcuRep::RemoteAddr>& tmpSrc,
        std::vector<CcuRep::LocalAddr>& tmpDst);

    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
    void ReduceLoopGroup(CcuRep::LocalAddr outDstOrg, CcuRep::LocalAddr srcOrg, std::vector<CcuRep::LocalAddr> &scratchOrg,
        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);

    const std::string LOOP_BLOCK_TAG{"_local_copy_reduce_loop_"};

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    CcuRep::Variable scratch_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable scratchInterval_;
    CcuRep::Variable sliceSize_;
    CcuRep::Variable offset_;
    CcuRep::LocalAddr src;
    std::vector<CcuRep::RemoteAddr> dst;
    std::vector<CcuRep::RemoteAddr> reduceScatterVSrc_;
    std::vector<CcuRep::LocalAddr> reduceScatterVDst_;
    GroupOpSize reduceGosize_;
    CcuRep::CompletedEvent event_;
};

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_V_MESH_1D_MEM2MEM
