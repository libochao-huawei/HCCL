#ifndef HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H
#define HCCL_ALLGATHERBATCH_HD_STAGE_CORE_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct NHRRunCtx;

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

struct StageCopySlice {
    uint32_t remoteRank = 0;
    uint64_t localOffset = 0;
    uint64_t remoteOffset = 0;
    uint64_t size = 0;
};
struct LastTwoCopyTask {
    uint32_t peerRank = 0;
    bool isLocal = false;
    std::vector<StageCopySlice> slices;
};
struct PowerStepTask {
    uint32_t bit = 0;
    uint32_t remoteRank = 0;
    uint64_t totalBytes = 0;
    uint64_t localOffset = 0;
    uint64_t remoteOffset = 0;
    uint64_t size = 0;
};

class AllGatherHDStageCore {
public:
    AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, const WindowStageLayout &layout);
    HcclResult RunAsync();

private:
    HcclResult ValidateStageInput() const;
    HcclResult BuildStagePlan(HDStagePlan &plan) const;
    HcclResult ValidateStagePlan(const HDStagePlan &plan) const;
    HcclResult BuildNHRRunCtx(const HDStagePlan &plan, NHRRunCtx &runCtx) const;
    HcclResult ValidatePreparedNHRLayout(const HDStagePlan &plan, const NHRRunCtx &runCtx) const;
    HcclResult RunAllGatherStage(const HDStagePlan &plan) const;
    HcclResult RunPreCopy() const;
    HcclResult RunNHR(const char *pathTag, const NHRRunCtx &runCtx) const;
    HcclResult RunAllGatherNoPower(const HDStagePlan &plan) const;
    HcclResult RunAllGatherPower(const HDStagePlan &plan) const;
    HcclResult RunAllGatherFinal(const HDStagePlan &plan) const;
    HcclResult RunAllGatherLast(const HDStagePlan &plan) const;
    HcclResult RunAllGatherLastOne(const HDStagePlan &plan) const;
    HcclResult RunAllGatherLastTwo(const HDStagePlan &plan) const;
    HcclResult BuildFinalCopyTasks(uint32_t subgroupSize, std::vector<LastTwoCopyTask> &tasks) const;
    HcclResult ValidateFinalCopyTasks(uint32_t subgroupSize, const std::vector<LastTwoCopyTask> &tasks) const;
    HcclResult BuildLastTwoTasks(const HDStagePlan &plan, std::vector<LastTwoCopyTask> &tasks) const;
    HcclResult RunLastTwoLocalTask(const LastTwoCopyTask &task, const char *stageTag) const;
    HcclResult RunRemoteTaskOnMainThread(const LastTwoCopyTask &task, const char *stageTag) const;
    HcclResult RunPowerBit(const PowerStepTask &stepTask, const char *stageTag) const;
    HcclResult BuildPowerStepTask(const HDStagePlan &plan, uint32_t bit, PowerStepTask &stepTask) const;
    const ChannelResource *FindChannel(uint32_t remoteRank) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
    WindowStageLayout layout_ {};
};

}  // namespace ops_hccl_allgatherbatch

#endif


