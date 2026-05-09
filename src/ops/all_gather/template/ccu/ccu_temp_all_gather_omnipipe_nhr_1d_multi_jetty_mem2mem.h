/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_NHR_1D_MULTI_JETTY_MEM2MEM_H
#define HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_NHR_1D_MULTI_JETTY_MEM2MEM_H

#include "ccu_alg_template_base.h"
#include "ccu_temp_all_gather_omnipipe_common.h"
#include "ccu_kernel_all_gather_omnipipe_nhr1d_multi_jetty_mem2mem.h"
#include "utils.h"

namespace ops_hccl {

class CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem() = default;
    explicit CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
        const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks);
    CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem(
        const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks,
        CommTopo priorityTopo);
    ~CcuTempAllGatherOmniPipeNHR1DMultiJettyMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat(
            "Template of All Gather CCU OmniPipe NHR1D Multi-Jetty Mem2Mem skeleton with tempRankSize [%u].",
            templateRankSize_);
    }

    HCCL_CCU_OMNIPIPE_TEMPLATE_METHODS();

private:
    HcclResult CalcNHRInfo(std::vector<CcuOmniPipeNHRStepInfo> &stepInfoVector) const;
    u32 GetNHRStepNum(u32 rankSize) const;
    HcclResult GetStepInfo(u32 step, u32 nSteps, CcuOmniPipeNHRStepInfo &stepInfo) const;
    HcclResult RemoteRankId2RankId(uint32_t remoteRankId, uint32_t &rankId) const;
    HcclResult LaunchStepSlice(const OpParam &param, const TemplateDataParams &templateDataParams,
                               TemplateResource &templateResource, const CcuOmniPipeNHRStepInfo &stepInfo,
                               u32 stepIdx);

    uint32_t jettyNum_{4};
    CommTopo priorityTopo_{CommTopo::COMM_TOPO_CLOS};
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_GATHER_OMNIPIPE_NHR_1D_MULTI_JETTY_MEM2MEM_H
