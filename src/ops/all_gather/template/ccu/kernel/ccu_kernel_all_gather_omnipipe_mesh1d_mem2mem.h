/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_MESH1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_MESH1D_MEM2MEM_H

#include <vector>
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

class CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem(
        uint64_t dimSize, uint32_t rankId, const OpParam &opParam,
        const std::vector<std::vector<uint32_t>> &subCommRanks)
        : dimSize_(dimSize), rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks)
    {
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override;

    uint64_t dimSize_{0};
    uint32_t rankId_{0};
    OpParam opParam_{};
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                      uint64_t srcOffset, uint64_t dstOffset, uint64_t sliceSize,
                                                      uint64_t isSrcDstEqual)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), srcOffset_(srcOffset), dstOffset_(dstOffset),
          sliceSize_(sliceSize), isSrcDstEqual_(isSrcDstEqual)
    {
    }

    uint64_t inputAddr_{0};
    uint64_t outputAddr_{0};
    uint64_t token_{0};
    uint64_t srcOffset_{0};
    uint64_t dstOffset_{0};
    uint64_t sliceSize_{0};
    uint64_t isSrcDstEqual_{0};
};

class CcuKernelAllGatherOmniPipeMesh1DMem2Mem : public CcuKernelAlgBase {
public:
    explicit CcuKernelAllGatherOmniPipeMesh1DMem2Mem(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherOmniPipeMesh1DMem2Mem() override = default;

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoOmniPipeMeshAllGather();

    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    std::vector<ChannelHandle> channels_;

    hcomm::CcuRep::Variable localInput_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable isSrcDstEqual_;
    GroupOpSize localGoSize_;

    hcomm::CcuRep::LocalAddr src_;
    hcomm::CcuRep::LocalAddr localDst_;
    std::vector<hcomm::CcuRep::RemoteAddr> remoteDst_;
    hcomm::CcuRep::CompletedEvent event_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_MESH1D_MEM2MEM_H
