/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_TEMP_ALLGATHER_MESH_1D_DETOUR_H
#define HCCL_CCU_TEMP_ALLGATHER_MESH_1D_DETOUR_H
 
#include "ccu_alg_template_base.h"
#include "utils.h"
 
namespace ops_hccl {
 
class CcuTempAllGatherMesh1DDetour : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllGatherMesh1DDetour(const OpParam& param, 
                                           const u32 rankId,
                                           const std::vector<std::vector<u32>> &subCommRanks) = default;
    ~CcuTempAllGatherMesh1DDetour() override;
 
    std::string Describe() const override
    {
        return StringFormat("Template of All Reduce ccu mesh 1D detour with tempRankSize [%u].", subCommRanks_[0].size());
    }
 
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
 
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource);
 
private:
    void CalcDetourOffset(uint64_t sliceSize, uint64_t &tailOffset, uint64_t &tailSize, uint64_t &iterNum);
    HcclResult ProcessDetourChannels(std::vector<HcclChannelDesc> &channels);
    HcclResult CalcSliceInfo(const u64 dataSize, RankSliceInfo &sliceInfoVec);
    HcclResult CalcSliceInfoAllGather(const u32 rankSize, const u64 dataSize,
                                      RankSliceInfo &sliceInfoVec);
    HcclReduceOp reduceOp_;
    HcclDataType dataType_;
    uint64_t detourPathNum_{0};  // 到每个对端有几个绕路路径
    uint64_t pathNumPerPeer_{0};
    std::vector<uint64_t> lengths_;
    uint64_t singleTransferSize_{0};
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
    uint32_t mySubCommRank_ = 0;
};
}
#endif // HCCLV2_CCU_TEMP_ALL_GATHER_MESH_DETOUR_1D_H_