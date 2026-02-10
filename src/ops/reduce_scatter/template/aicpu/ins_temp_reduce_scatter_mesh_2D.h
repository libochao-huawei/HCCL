/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_2D_H
#define INS_TEMP_REDUCE_SCATTER_MESH_2D_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"

namespace ops_hccl {

constexpr u32 PARALLEL_SIZE = 2;
class InsTempReduceScatterMesh2D : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterMesh2D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterMesh2D() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh2D with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         const TemplateResource& templateResource) override;
    
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;
    u64 GetThreadNum() override;

private:
    HcclResult PreCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult SendRecvProcess(const std::map<u32, std::vector<ChannelInfo>> &channels,
                               std::vector<std::vector<DataSlice>> allSliceVec,
                               const std::vector<ThreadHandle> &threads, u32 remoteRank, u32 queIdx) const;
    HcclResult RunFirstLevel(const std::map<u32, std::vector<ChannelInfo>> &channels,
                             const std::vector<ThreadHandle> &threads,
                             const TemplateDataParams &tempAlgParams);
    HcclResult RunFirstReduce(const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams);
    HcclResult RunSecondLevel(const std::map<u32, std::vector<ChannelInfo>> &channels,
                              const std::vector<ThreadHandle> &threads,
                              const TemplateDataParams &tempAlgParams);
    HcclResult RunSecondReduce(const std::vector<ThreadHandle> &threads,
                               const TemplateDataParams &tempAlgParams);
    u32      xThreadNum_ = 0;
    u32      yThreadNum_ = 0;
    u32      xRankSize_ = 0;
    u32      yRankSize_ = 0;
    u32      xAlgRankId_ = 0;
    u32      yAlgRankId_ = 0;
    u64      halfDataSize_ = 0;
};


} // namespace Hccl

#endif //INS_TEMP_REDUCE_SCATTER_MESH_2D_H