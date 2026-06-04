/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALLTOALL_MESH_CLOS_V3_NO_MEMCPY_H
#define INS_TEMP_ALLTOALL_MESH_CLOS_V3_NO_MEMCPY_H

#include "ins_temp_alltoall_mesh_2d_v3_no_memcpy.h"

namespace ops_hccl {

class InsTempAlltoAllMeshClosV3NoMemcpy : public InsTempAlltoAllMesh2DV3NoMemcpy {
public:
    InsTempAlltoAllMeshClosV3NoMemcpy() = default;
    explicit InsTempAlltoAllMeshClosV3NoMemcpy(const OpParam &param, const u32 rankId,
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllMeshClosV3NoMemcpy() override;

    std::string Describe() const override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;
    u64 GetThreadNum() const override;

    // Local copy: own data from input → output + scratch for all destination slots
    HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads) override;

    // Post copy: received data from scratch → output for all column peers
    // C-9 fix: virtual so clos can override with dy-based formula
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads) override;
    
protected:
    HcclResult RunAlltoAllMesh(
        const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels) override;

private:
    struct ClosNoMemcpySlot {
        u32 txRank = 0;
        u32 rxRank = 0;
        u32 channelIdx = 0;

        ClosNoMemcpySlot() = default;
        ClosNoMemcpySlot(u32 tx, u32 rx, u32 channel) : txRank(tx), rxRank(rx), channelIdx(channel)
        {
        }
    };

    HcclResult CalcClosNoMemcpyRoundPlan(u32 round, std::vector<ClosNoMemcpySlot> &slotPlans) const;
    HcclResult SelectClosNoMemcpyChannel(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                         u32 remoteRank, u32 channelIdx, ChannelInfo &channel) const;
    HcclResult RunClosNoMemcpySlot(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                   const ClosNoMemcpySlot &slotPlan,
                                   const ThreadHandle &thread,
                                   u32 round,
                                   u64 actualChunkSize,
                                   u64 chunkCount,
                                   bool isPcie);
    u32 GetRowNum() const;
    u32 GetClosSlotNum() const;
    HcclResult RunAlltoAllOnLink(
        const std::vector<ThreadHandle> &commThreads,
        const std::map<u32, std::vector<ChannelInfo>> &channels,
        u32 linkIdx, u32 step, u32 numSteps);

    u32 GetCopyNotifySlotCount() const;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALLTOALL_MESH_CLOS_V3_NO_MEMCPY_H
