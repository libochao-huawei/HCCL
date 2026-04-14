/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aiv_utils.h"
#include "aiv/aiv_temp_reduce_mesh_1D_twoshot.h"

namespace ops_hccl {

AivTempReduceMesh1DTwoShot::AivTempReduceMesh1DTwoShot(const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{
}

AivTempReduceMesh1DTwoShot::~AivTempReduceMesh1DTwoShot()
{
}

u64 AivTempReduceMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 2;
}

HcclResult AivTempReduceMesh1DTwoShot::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    uint32_t threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (uint32_t index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempReduceMesh1DTwoShot::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    if (numBlocksLimit >= (tempRankSize_ + 1)) {
        uint32_t coreNumPerRank = numBlocksLimit / (tempRankSize_ + 1);
        numBlocks = coreNumPerRank * (tempRankSize_ + 1);
    } else {
        numBlocks = numBlocksLimit;
    }
    HCCL_INFO("[AivTempReduceMesh1DTwoShot] Actually use core num[%u]", numBlocks);
    return HCCL_SUCCESS;
}

HcclResult AivTempReduceMesh1DTwoShot::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempReduceMesh1DTwoShot] KernelRun start");

    IncSliceId();
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivReduceArgs;
    aivReduceArgs.cmdType = HcclCMDType::HCCL_CMD_REDUCE;
    aivReduceArgs.argsType = KernelArgsType::ARGS_TYPE_TWO_SHOT;
    aivReduceArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivReduceArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivReduceArgs.rank = static_cast<uint32_t>(myRank_);
    aivReduceArgs.rankSize = tempRankSize_;
    aivReduceArgs.count = tempAlgParams.sliceSize / SIZE_TABLE[dataType_];
    aivReduceArgs.dataType = dataType_;
    aivReduceArgs.op = param.reduceType;
    aivReduceArgs.root = root_;
    aivReduceArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivReduceArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivReduceArgs.stream = param.stream;
    aivReduceArgs.isOpBase = (param.opMode == OpMode::OPBASE);
    aivReduceArgs.xRankSize = subCommRanks_[0].size();

    for (uint32_t i = 0; i < subCommRanks_[0].size(); i++) {
        aivReduceArgs.topo_[i] = subCommRanks_[0][i];
    }
    if (subCommRanks_.size() > 1) {
        aivReduceArgs.yRankSize = subCommRanks_[1].size();
        for (uint32_t i = 0; i < subCommRanks_[1].size(); i++) {
            aivReduceArgs.topo_[TOPO_LEN_Y_OFFSET + i] = subCommRanks_[1][i];
        }
    }
    if (subCommRanks_.size() > 2) {
        aivReduceArgs.zRankSize = subCommRanks_[2].size();
        for (uint32_t i = 0; i < subCommRanks_[2].size(); i++) {
            aivReduceArgs.topo_[TOPO_LEN_Z_OFFSET + i] = subCommRanks_[2][i];
        }
    }

    CHK_RET(CalNumBlocks(aivReduceArgs.numBlocks, tempAlgParams.sliceSize, param.numBlocksLimit));
    aivReduceArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivReduceArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivReduceArgs.repeatNum = tempAlgParams.repeatNum;
    aivReduceArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivReduceArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivReduceArgs));

    HCCL_INFO("[AivTempReduceMesh1DTwoShot] KernelRun finished");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
