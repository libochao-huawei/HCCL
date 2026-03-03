/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: 算法模板CcuTempAllToAllMesh2Die类头文件
 */

#ifndef HCCL_CCU_TEMP_ALLTOALL_MESH_2DIE_H
#define HCCL_CCU_TEMP_ALLTOALL_MESH_2DIE_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

using RankId = u32;
using RankGroup = std::vector<RankId>;

class CcuTempAllToAllMesh2Die : public CcuAlgTemplateBase{
public:
     CcuTempAllToAllMesh2Die(const OpParam &param, RankId rankId, const std::vector<std::vector<u32>> &subCommRanks);
     ~CcuTempAllToAllMesh2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoall ccu mesh 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        const TemplateResource &templateResource) override;

private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs);

    const uint32_t DIE_NUM = 2; // 2Die

    std::map<uint32_t, std::vector<HcclChannelDesc>> channels_; // key is DieId
    std::map<uint32_t, RankGroup> rankGroup_;
};
}// namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALLTOALL_MESH_2DIE_H
