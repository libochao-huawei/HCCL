/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 算法库CcuTemAllGatherNHR1DMultiJettyMem2Mem类实现
 * Author: xxx
 * Create: 2026-xx-xx
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM_H
#define HCCL_CCU_TEMP_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM_H

#include "ccu_alg_template_base.h"
#include "utils.h"
#include "ccu_kernel_all_gather_nhr_1D_multijetty_mem2mem.h"

namespace ops_hccl {

class CcuTemAllGatherNHR1DMultiJettyMem2Mem : public CcuAlgTemplateBase {
public:
    explicit  CcuTemAllGatherNHR1DMultiJettyMem2Mem(const OpParam& param,
                                                const u32 rankId,
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTemAllGatherNHR1DMultiJettyMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All Gather CCU NHR 1D Multijetty Mem2Mem with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;
    
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    HcclResult CalcNHRInfo(std::vector<NHRStepInfo> &stepInfoVector);

    u32 GetNHRStepNum(u32 rankSize);

    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);

    uint32_t RemoteRankId2RankId(const uint32_t remoteRankId);

    HcclResult GetRes(AlgResourceRequest &resourceRequest);

    u64 GetThreadNum();

private:
    uint32_t mySubCommRank_ = 0;
    uint32_t jettyNum_ = 0;
    std::vector<std::vector<u32>> subCommRanks_;
    uint32_t tempRankSize_ = 0;
};

} // namespace ops_hccl

#endif// HCCL_CCU_TEMP_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM_H