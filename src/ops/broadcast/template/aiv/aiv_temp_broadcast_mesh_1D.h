/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 算法模板AivTempBroadcastMesh1D类头文件
 * Author: zhangzuyu
 * Create: 2025-09-15
 */

#ifndef AIV_TEMP_ALL_BROADCAST_MESH_1D
#define AIV_TEMP_ALL_BROADCAST_MESH_1D

#include <cstring>
#include "aiv_alg_template_base.h"
// #include "alg_v2_template_register.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class AivTempBroadcastMesh1D : public AivAlgTemplateBase {
public:
    explicit AivTempBroadcastMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~AivTempBroadcastMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of broadcast Mesh with tempRankSize ";
        info += std::to_string(tempRankSize_);
        return info;
    }
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         const TemplateResource& templateResource) override;
    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) override;
};
}  // namespace Hccl

#endif  // AIV_TEMP_BROADCAST_MESH_1D