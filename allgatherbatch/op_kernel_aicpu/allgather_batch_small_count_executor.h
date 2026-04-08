#ifndef HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H
#define HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H

#include <cstdint>

#include "common.h"
#include "window_range.h"

namespace ops_hccl_allgatherbatch {

class AllGatherBatchSmallCountExecutor {
public:
    AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx);

    // Device 侧主控入口：校验参数、建立窗口边界，再把控制流交给后续阶段的通信核心。
    HcclResult Orchestrate();

private:
    HcclResult ValidateParam() const;
    HcclResult ValidateWindow(const WindowRange &window) const;
    HcclResult BuildFirstWindow(WindowRange &window) const;
    HcclResult BuildNextWindow(const WindowRange &current, WindowRange &next, bool &hasNext) const;
    HcclResult Pack(const WindowRange &window) const;
    HcclResult Unpack(const WindowRange &window) const;
    HcclResult AdvancePosition(uint32_t &itemIdx, uint64_t &offsetBytes) const;
    HcclResult LocateWindowEnd(uint32_t startItemIdx, uint64_t startOffsetBytes, uint64_t packedBytes,
        uint32_t &endItemIdx, uint64_t &endOffsetBytes) const;
    uint64_t CalcWindowCoveredBytes(const WindowRange &window) const;
    uint64_t GetPerRankWindowCapacity() const;
    uint8_t *GetRankWindowBase(const WindowRange &window, uint32_t rank) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
