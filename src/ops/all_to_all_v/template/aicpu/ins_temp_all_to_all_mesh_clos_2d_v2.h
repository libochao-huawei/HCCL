/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_MESH_CLOS_2D_V2_H
#define INS_TEMP_ALL_TO_ALL_MESH_CLOS_2D_V2_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

class InsTempAlltoAllMeshClos2DV2 : public InsAlgTemplateBase {
public:
    InsTempAlltoAllMeshClos2DV2() = default;
    explicit InsTempAlltoAllMeshClos2DV2(const OpParam &param, const u32 rankId,
                                         const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllMeshClos2DV2() override;

    std::string Describe() const override
    {
        std::string info = "Template of AlltoAll MeshClos2DV2 totalRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunAxisMeshStage(const std::vector<ThreadHandle> &threads,
                                const std::map<u32, std::vector<ChannelInfo>> &channels,
                                const std::vector<u32> &axisRanks, u32 axisRankSize,
                                u32 myAxisRank, u32 threadBaseIdx, u32 channelsPerRank,
                                const std::vector<u64> &srcAddrs, const std::vector<u64> &dstAddrs,
                                u64 dataSize, bool toScratch,
                                BufferType srcType, BufferType dstType);

    HcclResult RunAxisMeshYClos(const std::vector<ThreadHandle> &threads,
                                const std::map<u32, std::vector<ChannelInfo>> &channels,
                                const std::vector<u32> &axisRanks, u32 axisRankSize,
                                u32 myAxisRank, u32 threadBaseIdx, u32 channelsPerRank,
                                const std::vector<u64> &srcAddrs, const std::vector<u64> &dstAddrs,
                                u64 dataSize, BufferType srcType, BufferType dstType);

    void ComputeStageAddrs(const TemplateDataParams &tp, u64 dataOffset, u64 dataSize,
                           bool fromInput, BufferType srcType, BufferType dstType,
                           std::vector<u64> &srcAddrs, std::vector<u64> &dstAddrs);

    u32 xRankSize_{0};
    u32 yRankSize_{0};
    u32 myXRank_{0};
    u32 myYRank_{0};
    u32 totalRankSize_{0};
    u64 dataTypeSize_{0};
    u64 sliceSize_{0};
    TemplateDataParams tempAlgParams_;
    u32 xChannelsPerRank_{1};
    u32 yChannelsPerRank_{1};
    u32 xThreads_{0};
    u32 yThreads_{0};
};

}  // namespace ops_hccl

#endif
