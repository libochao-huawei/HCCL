/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_MESH_1D_Z_AXIS_DETOUR_H
#define INS_TEMP_ALL_GATHER_MESH_1D_Z_AXIS_DETOUR_H

#include "ins_temp_all_gather_mesh_1D.h"

namespace ops_hccl {

class InsTempAllGatherMesh1D1DZAxisDetour : public InsTempAllGatherMesh1D {
public:
    InsTempAllGatherMesh1D1DZAxisDetour() = default;
    explicit InsTempAllGatherMesh1D1DZAxisDetour(const OpParam &param, const u32 rankId,
                                    const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherMesh1D1DZAxisDetour() override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;

    u64 GetThreadNum() const override;

    HcclResult CalcDataSplitByPortGroup(const u64 totalDataCount, const u64 dataTypeSize,
                                        const std::vector<ChannelInfo> &channels,
                                        std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
                                        std::vector<u64> &elemOffset) override;
    HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>> &channels) override;
    u32 channelsSize;
protected:
    HcclResult RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                        const std::map<u32, std::vector<ChannelInfo>> &channels) override;
private:
    std::vector<ChannelInfo> PrepareMergedChannels(
        const std::map<u32, std::vector<ChannelInfo>> &channels);
    u64 CalcSliceSizeForChannel(u32 myAlgRank, u32 connectedAlgRank, bool dmaRead) const;
    void BuildDataSlicesForChannel(
        u32 connectedRank, u32 myAlgRank, u32 connectedAlgRank, u32 idx,
        const ChannelInfo &linkRemote, void *remoteCclBuffAddr,
        std::vector<DataSlice> &txSrcSlicesAll, std::vector<DataSlice> &txDstSlicesAll,
        std::vector<DataSlice> &rxDstSlicesAll, std::vector<DataSlice> &rxSrcSlicesAll);
    HcclResult ExecuteSendRecvForChannel(
        u32 threadIdx, bool dmaRead, const std::vector<ThreadHandle> &threads,
        const ChannelInfo &linkRemote,
        const std::vector<DataSlice> &txSrcSlicesAll, const std::vector<DataSlice> &txDstSlicesAll,
        const std::vector<DataSlice> &rxSrcSlicesAll, const std::vector<DataSlice> &rxDstSlicesAll);
    HcclResult ProcessSingleChannel(
        u32 threadIdx, u32 myAlgRank, bool dmaRead, const u32 dataTypeSize,
        const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ChannelInfo> &mergedChannels);
    static bool isNew;
    u32 level0ChannelNumPerRank_{1};
    u32 level1ChannelNumPerRank_{0};
    float level0DataRatio_{1.0f};
    std::vector<u64> elemCountOut_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_GATHER_MESH_1D_H