#ifndef HCCL_ALLGATHERBATCH_NHR_CORE_H
#define HCCL_ALLGATHERBATCH_NHR_CORE_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct LocalSlice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct InterServerAlgoStep {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices = 0;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;
};

struct NHRRunCtx {
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t packedBytes = 0;
    uint64_t baseOffset = 0;
    uint8_t *inputBase = nullptr;
    uint8_t *outputBase = nullptr;
    bool keepOrder = true;
    std::vector<LocalSlice> slices;
    std::vector<uint32_t> subgroupRanks;
};

class AllGatherNHRCore {
public:
    AllGatherNHRCore(
        const OpParam &param,
        AlgResourceCtx &resCtx,
        uint64_t packedBytes,
        const NHRRunCtx &runCtx = NHRRunCtx {});

    HcclResult RunAsync();

private:
    void InitDefaultRunCtx();
    uint32_t GetEffectiveRank() const;
    uint32_t GetEffectiveRankSize() const;
    uint32_t GetSliceGroupSize() const;
    HcclResult ValidateCommState() const;
    HcclResult ValidateStepPlan(const std::vector<InterServerAlgoStep> &stepPlan) const;
    void GetRankMapping(uint32_t rankSize, bool keepOrder);
    void ReorderSequence(
        uint32_t start,
        uint32_t end,
        uint32_t len,
        std::vector<uint32_t> &tree,
        std::vector<uint32_t> &tmp) const;
    uint32_t GetStepNumInterServer(uint32_t rankSize) const;
    HcclResult GetStepInfo(uint32_t step, uint32_t nSteps, InterServerAlgoStep &stepInfo) const;
    HcclResult BuildStepPlan(std::vector<InterServerAlgoStep> &stepPlan) const;
    HcclResult RunAllGather(const std::vector<InterServerAlgoStep> &stepPlan) const;
    HcclResult BuildStepSlices(
        const InterServerAlgoStep &stepInfo,
        std::vector<LocalSlice> &txSlices,
        std::vector<LocalSlice> &rxSlices) const;
    void MergeSlices(std::vector<LocalSlice> &slices) const;
    const ChannelResource *FindChannelByGlobalRank(uint32_t remoteRank) const;
    const ChannelResource *FindChannelBySubgroupRank(uint32_t subgroupRank) const;
    HcclResult Tx(const ChannelResource &channel, const std::vector<LocalSlice> &txSlices) const;
    HcclResult Rx(const ChannelResource &channel, const std::vector<LocalSlice> &rxSlices) const;
    HcclResult SdmaRx(
        const ChannelResource *channelLeft,
        const ChannelResource *channelRight,
        const std::vector<LocalSlice> &rxSlices,
        const InterServerAlgoStep &stepInfo) const;
    HcclResult RdmaTxRx(
        const ChannelResource *channelLeft,
        const ChannelResource *channelRight,
        const std::vector<LocalSlice> &txSlices,
        const std::vector<LocalSlice> &rxSlices,
        const InterServerAlgoStep &stepInfo) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    uint64_t packedBytes_;
    NHRRunCtx runCtx_;
    std::vector<uint32_t> sliceMap_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
