#ifndef HCCL_ALLGATHERBATCH_ALL_GATHER_NHR_CORE_H
#define HCCL_ALLGATHERBATCH_ALL_GATHER_NHR_CORE_H

#include <vector>

#include "common.h"

namespace ops_hccl_allgatherbatch {

using InterServerAlgoStep = struct InterServerAlgoStepDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices = 0;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;
};

class AllGatherNHRCore {
public:
    AllGatherNHRCore(AlgResourceCtx &resCtx,
        ExecMem &execMem,
        u64 baseOffset,
        u64 totalSize,
        const std::vector<ChannelResource> &channels = {});

    HcclResult Prepare(bool needMerge);
    void SetInputPreparedInOutput(bool enabled);
    HcclResult RunAsync(const u32 rank, const u32 rankSize);

private:
    HcclResult RunAllGather(u32 rank, u32 rankSize,
        const std::vector<Slice> &outputSlices,
        const std::vector<ChannelResource> &links);
    HcclResult SdmaRx(const ChannelResource &linkLeft,
        const ChannelResource &linkRight,
        std::vector<Slice> &rxSlices);
    HcclResult RdmaTxRx(const ChannelResource &linkLeft,
        const ChannelResource &linkRight,
        InterServerAlgoStep &stepInfo,
        std::vector<Slice> &txSlices,
        std::vector<Slice> &rxSlices);
    HcclResult Tx(const ChannelResource &link, std::vector<Slice> &txSlices);
    HcclResult Rx(const ChannelResource &link, std::vector<Slice> &rxSlices);
    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize,
        InterServerAlgoStep &stepInfo);

    void MergeSlices(std::vector<Slice> &slices);

private:
    AlgResourceCtx &resCtx_;
    ExecMem &execMem_;
    u64 baseOffset_;
    u64 totalSize_;
    std::vector<ChannelResource> channels_;
    std::vector<u32> sliceMap_;
    bool isNeedMerge_ = false;
    bool inputPreparedInOutput_ = false;
};
}  // namespace ops_hccl_allgatherbatch
#endif
