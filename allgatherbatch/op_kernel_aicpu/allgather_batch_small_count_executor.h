#ifndef HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H
#define HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H

#include <cstdint>

#include "common.h"
#include "window_range.h"

namespace ops_hccl_allgatherbatch {

class AllGatherBatchSmallCountExecutor {
public:
    AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling);
    HcclResult Orchestrate();

private:
    HcclResult ValidateParam() const;
    HcclResult ValidateWindow(const WindowRange &window) const;
    HcclResult BuildFirstWindow(WindowRange &window) const;
    HcclResult BuildNextWindow(const WindowRange &current, WindowRange &next, bool &hasNext) const;
    HcclResult Pack(const WindowStageLayout &layout) const;
    HcclResult Unpack(const WindowStageLayout &layout) const;
    HcclResult AdvancePosition(uint32_t &itemIdx, uint64_t &offsetBytes) const;
    HcclResult LocateWindowEnd(uint32_t startItemIdx, uint64_t startOffsetBytes, uint64_t packedBytes,
        uint32_t &endItemIdx, uint64_t &endOffsetBytes) const;
    uint64_t CalcWindowCoveredBytes(const WindowRange &window) const;
    uint64_t GetPerRankWindowCapacity() const;
    uint8_t *GetStageRankWindowBase(const WindowStageLayout &layout, uint32_t rank) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    BatchCallProfiling &profiling_;
    WindowStageLayout stageLayout_ {};
};

}  // namespace ops_hccl_allgatherbatch

#endif
