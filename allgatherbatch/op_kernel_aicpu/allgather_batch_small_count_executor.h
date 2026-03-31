#ifndef HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H
#define HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H

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
    HcclResult BuildFirstWindow(WindowRange &window) const;

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
};

}  // namespace ops_hccl_allgatherbatch

#endif
