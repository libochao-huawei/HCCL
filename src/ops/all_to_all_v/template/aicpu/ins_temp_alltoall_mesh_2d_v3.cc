/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_2d_v3.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

InsTempAlltoAllMesh2DV3::InsTempAlltoAllMesh2DV3(const OpParam &param, const u32 rankId,
                                                  const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAlltoAllMesh2DV2(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMesh2DV3::~InsTempAlltoAllMesh2DV3() {}

u64 InsTempAlltoAllMesh2DV3::GetThreadNum() const
{
    return portCount_;
}

HcclResult InsTempAlltoAllMesh2DV3::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = portCount_;
    resourceRequest.slaveThreadNum = threadNum > 1 ? threadNum - 1 : 0;
    if (resourceRequest.slaveThreadNum > 0) {
        resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum > 1 ? threadNum - 1 : 0;
    return HCCL_SUCCESS;
}

void InsTempAlltoAllMesh2DV3::SetPortCount(u32 portCount)
{
    portCount_ = portCount;
    channelsPerRank_ = portCount;
}

void InsTempAlltoAllMesh2DV3::SetBorrowedLink(bool isBorrowed, u32 linkIndex)
{
    hasBorrowedLink_ = isBorrowed;
    borrowedLinkIndex_ = linkIndex;
}

void InsTempAlltoAllMesh2DV3::SetSharedPortMode(bool enable)
{
    sharedPortMode_ = enable;
}

HcclResult InsTempAlltoAllMesh2DV3::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (totalRankSize_ == 0) {
        HCCL_ERROR("[InsTempAlltoAllMesh2DV3][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    if (cellSize == 0) {
        cellSize = totalSliceSize;
    }

    u64 perPeerMeshSize = (totalSliceSize + xRankSize_ - 1) / xRankSize_;
    if (perPeerMeshSize == 0) {
        perPeerMeshSize = totalSliceSize;
    }

    u64 effectiveCellStride = (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER)
        ? tempAlgParams_.outputSliceStride : cellSize;

    for (u32 d = 0; d < totalRankSize_; d++) {
        u32 dx = d % xRankSize_;
        u32 dy = d / xRankSize_;

        u64 offsetInSlice = cellSize * d;
        u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
        u64 actualChunkSize = std::min(cellSize, remainingAtOffset);
        if (actualChunkSize == 0) {
            continue;
        }
        u64 chunkCount = actualChunkSize / dataTypeSize;

        u64 inOff = tempAlgParams_.buffInfo.inBuffBaseOff + offsetInSlice;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, actualChunkSize, chunkCount);

        u64 outOff = tempAlgParams_.buffInfo.outBuffBaseOff + tempAlgParams_.outputSliceStride * d;
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr &&
                            inOff == outOff);
        bool isScratchToOutput = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                                   tempAlgParams_.buffInfo.outBuffType == BufferType::OUTPUT);
        if (!skipOutCopy && !isScratchToOutput && tempAlgParams_.buffInfo.outBuffType != BufferType::HCCL_BUFFER) {
            DataSlice dstOutSlice(tempAlgParams_.buffInfo.outputPtr, outOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, dstOutSlice);
        }

        u64 cclOff = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                     perPeerMeshSize * dx + effectiveCellStride * dy;
        bool skipCclCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr &&
                            inOff == cclOff);
        if (!skipCclCopy) {
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, cclDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV3][PostLocalCopy] skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    if (xRankSize_ == 0 || yRankSize_ == 0 || totalRankSize_ == 0) {
        HCCL_ERROR("[InsTempAlltoAllMesh2DV3][PostLocalCopy] invalid rank sizes.");
        return HCCL_E_INTERNAL;
    }

    u64 totalSliceSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSliceSize + totalRankSize_ - 1) / totalRankSize_;
    u64 perPeerSize = (totalSliceSize + xRankSize_ - 1) / xRankSize_;

    u64 effectiveCellStride = tempAlgParams_.inputSliceStride;

    for (auto rank : subCommRanks_[0]) {
        if (rank == myRank_) {
            continue;
        }
        u32 algRank = 0;
        CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));

        u32 sx = rank % xRankSize_;
        u32 sy = rank / xRankSize_;

        for (u32 dx = 0; dx < xRankSize_; dx++) {
            u32 d = dx + sy * xRankSize_;
            u64 offsetInSlice = cellSize * d;
            u64 remainingAtOffset = (offsetInSlice < totalSliceSize) ? (totalSliceSize - offsetInSlice) : 0;
            u64 actualChunkSize = std::min(cellSize, remainingAtOffset);
            if (actualChunkSize == 0) {
                continue;
            }
            u64 chunkCount = actualChunkSize / dataTypeSize;

            u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                                dx * perPeerSize + sy * effectiveCellStride;
            u64 outOffset = tempAlgParams_.buffInfo.outBuffBaseOff +
                            tempAlgParams_.outputSliceStride * d;

            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset,
                               actualChunkSize, chunkCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset,
                               actualChunkSize, chunkCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
