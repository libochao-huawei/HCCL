#ifndef HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#define HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct HDStagePlan {
    uint32_t powerSteps = 0;
    uint32_t noPower = 1;
    uint32_t finalSteps = 0;
    bool needNoPowerPath = false;
    bool needPowerPath = false;
};

class AllGatherHDStageCore {
public:
    AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes);

    // HDStage 总控入口：决定先走 noPower 路径还是 power 路径，并串起 NHR 子模板。
    HcclResult RunAsync();

private:
    HcclResult ValidateStageInput() const;
    HcclResult BuildStagePlan(HDStagePlan &plan) const;
    HcclResult ValidateStagePlan(const HDStagePlan &plan) const;
    HcclResult RunNoPowerPath(const HDStagePlan &plan) const;
    HcclResult RunPowerPath(const HDStagePlan &plan) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
