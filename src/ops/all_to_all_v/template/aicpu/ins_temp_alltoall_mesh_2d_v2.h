/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALLTOALL_MESH_2D_V2_H
#define INS_TEMP_ALLTOALL_MESH_2D_V2_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

class InsTempAlltoAllMesh2DV2 : public InsAlgTemplateBase {
public:
    InsTempAlltoAllMesh2DV2() = default;
    explicit InsTempAlltoAllMesh2DV2(const OpParam &param, const u32 rankId,
                                     const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempAlltoAllMesh2DV2() override;

    std::string Describe() const override
    {
        std::string info = "Template of AlltoAll Mesh2DV2 xRankSize ";
        info += std::to_string(xRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;

    u64 GetThreadNum() const override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    HcclResult KernelRun(const OpParam &param,
                         const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;

    HcclResult FastLaunch(const OpParam &param,
                          const TemplateFastLaunchCtx &tempFastLaunchCtx) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult CalcChannelRequestMesh2D(
        HcclComm comm,
        const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest);

    HcclResult RunAlltoAllMeshX(
        const std::vector<ThreadHandle> &threads,
        const std::map<u32, std::vector<ChannelInfo>> &channels,
        const TemplateDataParams &params);

    HcclResult LocalDataCopy(
        const ThreadHandle &thread,
        const TemplateDataParams &params);

    HcclResult PostLocalCopy(
        const ThreadHandle &thread,
        const TemplateDataParams &params);

    u64 xRankSize_{0};
    u64 myXRingRank_{0};
    u64 allRankSize_{0};
    bool isPcieProtocol_{false};
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALLTOALL_MESH_2D_V2_H
