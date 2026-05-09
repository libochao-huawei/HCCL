/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_local_copy.h"

namespace ops_hccl {

hcomm::CcuKernelSignature CcuKernelArgAllGatherOmniPipeLocalCopy::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    GenerateCcuKernelSignature(signature, "CcuKernelAllGatherOmniPipeLocalCopy", opParam_, subCommRanks_);
    return signature;
}

CcuKernelAllGatherOmniPipeLocalCopy::CcuKernelAllGatherOmniPipeLocalCopy(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    (void)arg;
}

HcclResult CcuKernelAllGatherOmniPipeLocalCopy::InitResource()
{
    srcAddr_ = CreateVariable();
    dstAddr_ = CreateVariable();
    srcToken_ = CreateVariable();
    dstToken_ = CreateVariable();
    srcOffset_ = CreateVariable();
    dstOffset_ = CreateVariable();
    copySize_ = CreateVariable();
    goSize_ = CreateGroupOpSize();
    src_ = CreateLocalAddr();
    dst_ = CreateLocalAddr();
    event_ = CreateCompletedEvent();
    return HCCL_SUCCESS;
}

void CcuKernelAllGatherOmniPipeLocalCopy::LoadArgs()
{
    Load(srcAddr_);
    Load(dstAddr_);
    Load(srcToken_);
    Load(dstToken_);
    Load(srcOffset_);
    Load(dstOffset_);
    Load(copySize_);
    Load(goSize_);
}

HcclResult CcuKernelAllGatherOmniPipeLocalCopy::DoLocalCopy()
{
    src_.addr = srcAddr_;
    src_.addr += srcOffset_;
    src_.token = srcToken_;
    dst_.addr = dstAddr_;
    dst_.addr += dstOffset_;
    dst_.token = dstToken_;
    event_.SetMask(1);
    CCU_IF(copySize_ != 0)
    {
        CHK_RET(GroupCopy(dst_, src_, goSize_));
        CHK_RET(RecordEvent(event_));
    }
    CCU_IF(copySize_ == 0)
    {
        CHK_RET(RecordEvent(event_));
    }
    event_.SetMask(1);
    CHK_RET(WaitEvent(event_));
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeLocalCopy::Algorithm()
{
    CHK_RET(InitResource());
    LoadArgs();
    CHK_RET(DoLocalCopy());
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeLocalCopy::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherOmniPipeLocalCopy *taskArg =
        dynamic_cast<const CcuTaskArgAllGatherOmniPipeLocalCopy *>(&arg);
    auto goSize = CalGoSize(taskArg->copySize_);
    return {taskArg->srcAddr_, taskArg->dstAddr_, taskArg->srcToken_, taskArg->dstToken_, taskArg->srcOffset_,
            taskArg->dstOffset_, taskArg->copySize_, goSize[0], goSize[1], goSize[2], goSize[3]};
}

} // namespace ops_hccl
