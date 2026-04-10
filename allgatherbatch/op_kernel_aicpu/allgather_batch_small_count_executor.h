#ifndef HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H
#define HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H

#include <cstdint>

#include "common.h"

namespace ops_hccl_allgatherbatch {

class AllGatherBatchSmallCountExecutor {
public:
    AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling);
    HcclResult Orchestrate();

private:
    HcclResult ValidateParam() const;
    HcclResult ValidateWindow(uint64_t windowOffset, uint64_t currentWindowBytes) const;
    HcclResult Pack(uint64_t windowOffset, uint64_t currentWindowBytes, const WindowStageLayout &layout) const;
    HcclResult Unpack(uint64_t windowOffset, uint64_t currentWindowBytes, const WindowStageLayout &layout) const;
    WindowStageLayout BuildStageLayout(uint64_t currentWindowBytes) const;
    uint64_t GetPerRankWindowCapacity() const;
    const BatchItemParam &GetInputItem() const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    BatchCallProfiling &profiling_;
};

}  // namespace ops_hccl_allgatherbatch

#endif

