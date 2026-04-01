#ifndef HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#define HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct HDStagePlan {
    uint32_t powerFactor = 1;
    uint32_t powerSteps = 0;
    uint32_t noPower = 1;
    uint32_t finalSteps = 0;
    uint32_t remainingPowerSteps = 0;
    bool needNoPowerPath = false;
    bool needPowerPath = false;
    bool needFinalPath = false;
    bool useNoPowerAsPrimary = false;
    bool usePowerAsPrimary = false;
    bool useFinalAsPrimary = false;
};

class AllGatherHDStageCore {
public:
    AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes);

    // HDStage 总控入口：决定主通信路径以及最终收尾路径，并串起 NHR 子模板。
    HcclResult RunAsync();

private:
    HcclResult ValidateStageInput() const;
    HcclResult BuildStagePlan(HDStagePlan &plan) const;
    HcclResult ValidateStagePlan(const HDStagePlan &plan) const;
    HcclResult ValidateProtocolDistribution() const;
    HcclResult RunNHR(const char *pathTag) const;
    HcclResult RunNoPowerPath(const HDStagePlan &plan) const;
    HcclResult RunPowerPath(const HDStagePlan &plan) const;
    HcclResult RunFinalPath(const HDStagePlan &plan) const;
    HcclResult RunFinalDirect(const HDStagePlan &plan) const;
    HcclResult RunFinalLastOne(const HDStagePlan &plan) const;
    HcclResult RunFinalLastTwo(const HDStagePlan &plan) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
