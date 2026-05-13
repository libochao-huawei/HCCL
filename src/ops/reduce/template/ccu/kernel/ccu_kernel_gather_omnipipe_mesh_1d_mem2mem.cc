/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"
#include "ccu_rep.h"

namespace ops_hccl {

CcuKernelGatherOmniPipeMesh1DMem2Mem::CcuKernelGatherOmniPipeMesh1DMem2Mem(const hcomm::CcuKernelArg& arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgGatherOmniPipeMesh1DMem2Mem& kernelArg =
        static_cast<const CcuKernelArgGatherOmniPipeMesh1DMem2Mem&>(arg);
    rankSize_ = kernelArg.dimSize_;
    rankIdx_ = kernelArg.rankId_;
    rootIdx_ = kernelArg.rootId_;
    subCommRanks_ = kernelArg.subCommRanks_;
    dataType_ = kernelArg.opParam_.DataDes.dataType;
    outputDataType_ = kernelArg.opParam_.outputDataType;

    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem] rankSize=%llu, rankIdx=%u, rootIdx=%u",
               rankSize_, rankIdx_, rootIdx_);
}

HcclResult CcuKernelGatherOmniPipeMesh1DMem2Mem::InitResource()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::InitResource] start");

    event_ = CcuRep::CompletedEvent("gather_omnipipe_mesh_mem2mem_event");

    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::InitResource] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::LoadArgs()
{
    const CcuTaskArgGatherOmniPipeMesh1DMem2Mem& taskArg =
        static_cast<const CcuTaskArgGatherOmniPipeMesh1DMem2Mem&>(GetTaskArg());
    input_.SetValue(taskArg.inputAddr_);
    output_.resize(1);
    output_[0].SetValue(taskArg.outputAddr_);
    token_.resize(1);
    token_[0].SetValue(taskArg.token_);
    sliceStride_.SetValue(taskArg.sliceStride_);
    sliceSize_.SetValue(taskArg.sliceSize_);
    localCopyFlag_.SetValue(taskArg.localCopyFlag_);
    inputOmniPipeSliceStride_.SetValue(taskArg.inputOmniPipeSliceStride_);

    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::LoadArgs] input=%llu, output=%llu, token=%llu, "
               "sliceSize=%llu, sliceStride=%llu, localCopyFlag=%llu, inputOmniPipeSliceStride=%llu",
               taskArg.inputAddr_, taskArg.outputAddr_, taskArg.token_, taskArg.sliceSize_,
               taskArg.sliceStride_, taskArg.localCopyFlag_, taskArg.inputOmniPipeSliceStride_);
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::PreSync()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::PreSync] start");
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::PostSync()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::PostSync] start");
}

std::vector<uint64_t> CcuKernelGatherOmniPipeMesh1DMem2Mem::GeneArgs(const hcomm::CcuTaskArg& arg)
{
    const CcuTaskArgGatherOmniPipeMesh1DMem2Mem& taskArg =
        static_cast<const CcuTaskArgGatherOmniPipeMesh1DMem2Mem&>(arg);
    std::vector<uint64_t> args;
    args.push_back(taskArg.inputAddr_);
    args.push_back(taskArg.outputAddr_);
    args.push_back(taskArg.token_);
    args.push_back(taskArg.sliceSize_);
    args.push_back(taskArg.sliceStride_);
    args.push_back(taskArg.localCopyFlag_);
    args.push_back(taskArg.inputOmniPipeSliceStride_);
    return args;
}

HcclResult CcuKernelGatherOmniPipeMesh1DMem2Mem::Algorithm()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::Algorithm] start");
    LoadArgs();
    CHK_RET(InitResource());
    PreSync();

    DoRepeatGather();

    PostSync();
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::Algorithm] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherOmniPipeMesh1DMem2Mem::DoRepeatGather()
{
    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::DoRepeatGather] rankIdx=%u, rootIdx=%u, isRoot=%d",
               rankIdx_, rootIdx_, (rankIdx_ == rootIdx_));

    CcuRep::Variable localCopyFlagVar = localCopyFlag_;
    CcuRep::Variable sliceStrideVar = sliceStride_;
    CcuRep::Variable sliceSizeVar = sliceSize_;
    CcuRep::Variable inputOmniPipeSliceStrideVar = inputOmniPipeSliceStride_;

    // Gather算法：只有非root节点发送数据到root
    if (rankIdx_ != rootIdx_) {
        // 发送数据到root节点
        u32 toRank = rootIdx_;
        ChannelHandle channelTo = GetChannelHandle(toRank);

        hcomm::CcuRep::LocalAddr src;
        src.addr = input_;
        src.size = sliceSizeVar;

        hcomm::CcuRep::RemoteAddr dst;
        dst.addr = output_[0];
        dst.channel = channelTo;

        CcuRep::Variable signal = CcuRep::Const(0);

        CcuRep::Send(dst, src, src.size, signal);

        HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem] Send: rankIdx=%u sending to rootIdx=%u, sliceSize=%llu",
                   rankIdx_, rootIdx_, sliceSize_.GetValue());

        CcuRep::Wait(signal);
    }

    HCCL_DEBUG("[CcuKernelGatherOmniPipeMesh1DMem2Mem::DoRepeatGather] end");
}

} // namespace ops_hccl