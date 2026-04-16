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

class AllGatherBatchSmallCountExecutor {
public:
    AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling);
    HcclResult Orchestrate();

private:
    HcclResult RunLoop(std::vector<ChannelResource> &channels);
    HcclResult KernelRun(ExecMem &execMem, std::vector<ChannelResource> &channels);

    HcclResult BuildWindowRange(const WindowRange &current, u64 maxWindowBytes,
        WindowRange &range, WindowRange &next) const;
    HcclResult PackWindowToCCLIn(const WindowRange &range, void *commInputPtr);
    HcclResult UnpackWindowFromCCLOut(const WindowRange &range, u8 *commOutputPtr);

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    BatchCallProfiling &profiling_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
