/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_LOCAL_COPY_H
#define HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_LOCAL_COPY_H

#include <vector>
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

class CcuKernelArgAllGatherOmniPipeLocalCopy : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherOmniPipeLocalCopy(const OpParam &opParam,
                                                    const std::vector<std::vector<uint32_t>> &subCommRanks)
        : opParam_(opParam), subCommRanks_(subCommRanks)
    {
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override;

    OpParam opParam_{};
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgAllGatherOmniPipeLocalCopy : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherOmniPipeLocalCopy(uint64_t srcAddr, uint64_t dstAddr, uint64_t srcToken,
                                                  uint64_t dstToken, uint64_t srcOffset, uint64_t dstOffset,
                                                  uint64_t copySize)
        : srcAddr_(srcAddr), dstAddr_(dstAddr), srcToken_(srcToken), dstToken_(dstToken), srcOffset_(srcOffset),
          dstOffset_(dstOffset), copySize_(copySize)
    {
    }

    uint64_t srcAddr_{0};
    uint64_t dstAddr_{0};
    uint64_t srcToken_{0};
    uint64_t dstToken_{0};
    uint64_t srcOffset_{0};
    uint64_t dstOffset_{0};
    uint64_t copySize_{0};
};

class CcuKernelAllGatherOmniPipeLocalCopy : public CcuKernelAlgBase {
public:
    explicit CcuKernelAllGatherOmniPipeLocalCopy(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherOmniPipeLocalCopy() override = default;

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    HcclResult DoLocalCopy();

    hcomm::CcuRep::Variable srcAddr_;
    hcomm::CcuRep::Variable dstAddr_;
    hcomm::CcuRep::Variable srcToken_;
    hcomm::CcuRep::Variable dstToken_;
    hcomm::CcuRep::Variable srcOffset_;
    hcomm::CcuRep::Variable dstOffset_;
    hcomm::CcuRep::Variable copySize_;
    GroupOpSize goSize_;
    hcomm::CcuRep::LocalAddr src_;
    hcomm::CcuRep::LocalAddr dst_;
    hcomm::CcuRep::CompletedEvent event_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_LOCAL_COPY_H
