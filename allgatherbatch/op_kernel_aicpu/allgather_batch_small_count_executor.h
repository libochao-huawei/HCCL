#ifndef HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H
#define HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H

#include <cstdint>

#include "common.h"

namespace ops_hccl_allgatherbatch {

struct WindowRange {
    uint32_t startDescIdx = 0;
    u64 startOffset = 0;
    uint32_t endDescIdx = 0;
    u64 endOffset = 0;
    u64 packedSize = 0;
};

struct WindowPart {
    uint32_t itemIdx = 0;
    u64 startOffset = 0;
    u64 sizeBytes = 0;
    u64 packedOffset = 0;
};

struct UnpackRankChunkPlan {
    std::vector<u32> beginRanks;
    std::vector<u32> endRanks;
    std::vector<bool> hasRankChunk;
    bool hasSubThreads = false;
    u64 quadThreadMinPackedBytes = 0;
};

class AllGatherBatchSmallCountExecutor {
public:
    AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx);
    HcclResult Orchestrate();

private:
    HcclResult RunLoop(std::vector<ChannelResource> &channels);
    HcclResult KernelRun(ExecMem &execMem, std::vector<ChannelResource> &channels);

    HcclResult BuildWindowPlan(const WindowRange &current, u64 maxWindowBytes,
        WindowRange &range, WindowRange &next, std::vector<WindowPart> &parts) const;
    HcclResult PackWindowToCCLIn(const std::vector<WindowPart> &parts, void *commInputPtr);
    HcclResult UnpackWindowFromCCLOut(const UnpackRankChunkPlan &unpackPlan,
        const std::vector<WindowPart> &parts, u64 packedSize, u8 *commOutputPtr);

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
