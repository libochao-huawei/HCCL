#ifndef HCCL_ALLGATHERBATCH_NHR_CORE_H
#define HCCL_ALLGATHERBATCH_NHR_CORE_H
#include <vector>
#include "common.h"
namespace ops_hccl_allgatherbatch {
struct NHRSubgroupCtx {
    uint32_t subgroupRank = 0;
    uint32_t subgroupSize = 0;
    uint64_t baseOffset = 0;
    std::vector<uint32_t> subgroupRanks;
};
struct NHRStepInfo {
    uint32_t step = 0;
    uint32_t fromRank = 0;
    uint32_t toRank = 0;
    uint32_t sliceCount = 0;
    std::vector<uint32_t> txItemOrder;
    std::vector<uint32_t> rxItemOrder;
};
class AllGatherNHRCore {
public:
    AllGatherNHRCore(
        const OpParam &param,
        AlgResourceCtx &resCtx,
        uint64_t packedBytes,
        const NHRSubgroupCtx &subgroupCtx = NHRSubgroupCtx {});
    // NHR ???????????????????? step ??????? rank ??????
    HcclResult RunAsync();
private:
    bool HasSubgroup() const;
    uint32_t GetEffectiveRank() const;
    uint32_t GetEffectiveRankSize() const;
    bool IsRankInActiveView(uint32_t rank) const;
    HcclResult ValidateCommState() const;
    HcclResult ValidateChannelMetadata() const;
    HcclResult ValidateProtocolDistribution() const;
    HcclResult ValidateStepPlan(const std::vector<NHRStepInfo> &stepPlan) const;
    uint32_t CalcStepNum(uint32_t rankSize) const;
    HcclResult GetStepInfo(uint32_t step, uint32_t nSteps, NHRStepInfo &stepInfo) const;
    HcclResult BuildStepPlan(std::vector<NHRStepInfo> &stepPlan) const;
    const ChannelResource *FindChannel(uint32_t remoteRank) const;
    uint8_t *GetRankBuffer(uint32_t rank) const;
    bool IsCrossServerChannel(const ChannelResource &channel) const;
    bool MatchChannel(const ChannelResource &channel, bool crossServer, CommProtocol protocol) const;
    uint32_t CountChannelsByScope(bool crossServer) const;
    uint32_t CountChannelsByProtocol(bool crossServer, CommProtocol protocol) const;
    uint32_t CountRecognizedChannelsByScope(bool crossServer) const;
    HcclResult NotifyReadyToRank(uint32_t remoteRank, bool crossServer, CommProtocol protocol) const;
    HcclResult ReadFromRank(uint32_t remoteRank, bool crossServer, CommProtocol protocol) const;
    HcclResult RunProtocolStep(const NHRStepInfo &stepInfo, bool crossServer, CommProtocol protocol) const;
    HcclResult RunProtocol(bool crossServer, CommProtocol protocol, const std::vector<NHRStepInfo> &stepPlan) const;
    HcclResult RunScope(bool crossServer, const std::vector<NHRStepInfo> &stepPlan) const;
    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
    NHRSubgroupCtx subgroupCtx_;
};
}  // namespace ops_hccl_allgatherbatch
#endif
