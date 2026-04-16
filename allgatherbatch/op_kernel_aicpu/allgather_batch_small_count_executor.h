#ifndef HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H
#define HCCL_ALLGATHERBATCH_SMALL_COUNT_EXECUTOR_H

#include <cstdint>

#include "common.h"

namespace ops_hccl_allgatherbatch {

constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;
constexpr int HCCL_DEVICE_NUM_TWO = 2; // 平均device num小于等于此数值时，无法通过HCCS链路类型接口判定当前硬件环境
constexpr int HCCL_DEVICE_NUM_FOUR = 4; // 平均device num等于此数值时，需校验server内device选取合法性

class AllGatherBatchSmallCountExecutor {
public:
    AllGatherBatchSmallCountExecutor(const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling);
    HcclResult Orchestrate();

private:
    HcclResult RunLoop(std::vector<ChannelResource> &channels);
    u64 CalcLoopMaxCount(u64 cclBuffSize, u32 unitSize);
    HcclResult KernelRun(ExecMem &execMem, std::vector<ChannelResource> &channels);

    const OpParam &param_;
    AlgResourceCtx &resCtx_;
    BatchCallProfiling &profiling_;
    bool useCCLBuffer{true};
};

}  // namespace ops_hccl_allgatherbatch

#endif
