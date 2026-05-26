/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_1D_TWO_SHOT_H
#define INS_TEMP_REDUCE_SCATTER_MESH_1D_TWO_SHOT_H

#include "alg_data_trans_wrapper.h"
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"

namespace ops_hccl {

class InsTempReduceScatterMesh1DTwoShot : public InsAlgTemplateBase {
public:
    InsTempReduceScatterMesh1DTwoShot() = default;
    explicit InsTempReduceScatterMesh1DTwoShot(const OpParam &param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterMesh1DTwoShot() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh1D two shot with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
        TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunReduceScatterOneShot(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads);
    HcclResult PostCopyOneShot(const OpParam &param, const TemplateDataParams &tempAlgParams,
        const std::vector<ThreadHandle> &threads);
    HcclResult RunReduceScatterTwoShot(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads);
    HcclResult RunReduceScatterShot(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads, bool secondShot);
    HcclResult PostCopyTwoShot(const OpParam &param, const TemplateDataParams &tempAlgParams,
        const std::vector<ThreadHandle> &threads);

    u64 GetSliceSize(u32 rankIdx, const TemplateDataParams &tempAlgParams) const;
    u64 GetSliceCount(u64 sliceSize) const;
    u64 GetScratchSlotOffset(const TemplateDataParams &tempAlgParams, u32 repeatIdx, u32 slotIdx) const;
    u32 GetRemoteOrderIdx(u32 ownerIdx, u32 srcIdx) const;
    u32 GetPairedRemoteOrderIdx(u32 ownerIdx, u32 srcIdx) const;
    u32 GetPairedSrcIdx(u32 ownerIdx, u32 srcIdx) const;
    u32 GetGroupSlotIdx(u32 ownerIdx, u32 srcIdx) const;
    bool IsSecondShotRemote(u32 ownerIdx, u32 srcIdx) const;
    bool IsUnpairedRemote(u32 ownerIdx, u32 srcIdx) const;
    bool CanRunTwoShot(const TemplateDataParams &tempAlgParams) const;
    u64 GetFirstHalfSize(u64 sliceSize) const;
    u64 GetPutReduceDstOffset(u64 sliceSize, u32 ownerIdx, u32 srcIdx) const;
    u64 GetShotOffset(u64 sliceSize, u32 ownerIdx, u32 srcIdx, bool secondShot) const;
    u64 GetShotSize(u64 sliceSize, u32 ownerIdx, u32 srcIdx, bool secondShot) const;

    u64 dataTypeSize_{0};
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_MESH_1D_TWO_SHOT_H
