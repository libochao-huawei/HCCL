#ifndef HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#define HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#include <vector>
#include "common.h"
namespace ops_hccl_allgatherbatch {
struct NHRSubgroupCtx;
struct HDStagePlan {
    uint32_t powerFactor = 1;
    uint32_t powerSteps = 0;
    uint32_t noPower = 1;
    uint32_t finalSteps = 0;
    uint32_t remainingPowerSteps = 0;
    bool needNoPowerPath = false;
    bool needPowerPath = false;
    bool needFinalPath = false;
};
class AllGatherHDStageCore {
public:
    AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes);
    // HDStage ???????????????????????????????
    HcclResult RunAsync();
private:
    HcclResult ValidateStageInput() const;
    HcclResult BuildStagePlan(HDStagePlan &plan) const;
    HcclResult ValidateStagePlan(const HDStagePlan &plan) const;
    HcclResult ValidateProtocolDistribution() const;
    HcclResult BuildNHRSubgroupCtx(const HDStagePlan &plan, NHRSubgroupCtx &subgroupCtx) const;
    HcclResult RunAllGatherStage(const HDStagePlan &plan) const;
    HcclResult RunPreCopy() const;
    HcclResult RunNHR(const char *pathTag, const NHRSubgroupCtx &subgroupCtx) const;
    HcclResult RunAllGatherNoPower(const HDStagePlan &plan) const;
    HcclResult RunAllGatherPower(const HDStagePlan &plan) const;
    HcclResult RunAllGatherFinal(const HDStagePlan &plan) const;
    HcclResult RunAllGatherLast(const HDStagePlan &plan) const;
    HcclResult RunAllGatherLastOne(const HDStagePlan &plan) const;
    HcclResult RunAllGatherLastTwo(const HDStagePlan &plan) const;
    HcclResult RunPowerBit(const HDStagePlan &plan, uint32_t bit, const char *stageTag) const;
    HcclResult BuildPartnerRanksForBit(const HDStagePlan &plan, uint32_t bit, std::vector<uint32_t> &partnerRanks) const;
    HcclResult ReadPartnerRanks(uint32_t partnerRank, const std::vector<uint32_t> &partnerRanks, const char *stageTag) const;
    const ChannelResource *FindChannel(uint32_t remoteRank) const;
    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
};
}  // namespace ops_hccl_allgatherbatch
#endif
