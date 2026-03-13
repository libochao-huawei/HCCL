#include "kernel_operator.h"
#include "sync_interface.h"
#include "llt_common.h"
#include "task_stub.h"
#include "aiv_task_stub.h"
#include "aiv_task_queue_stub.h"
#include "rank_info_recorder.h"


using namespace checker;
using namespace hccl;

namespace AscendC {

template __aicore__ void SyncFunc<HardEvent::MTE3_MTE2>();
template __aicore__ void SyncFunc<HardEvent::S_MTE2>();
template __aicore__ void SyncFunc<HardEvent::MTE3_S>();
template __aicore__ void SyncFunc<HardEvent::MTE2_S>();
template __aicore__ void SyncFunc<HardEvent::S_MTE3>();

template<HardEvent event> __aicore__ void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

// 在gm上设置同步信号的值
__aicore__ void SetSignalValue(__gm__ int32_t* gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> setValueTask(new TaskStubSetValue(value));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_S, setValueTask);

    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(gmSignalAddr, UB_FLAG_PAD_COUNT);
    localTensor.SetValue(0, value);
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(globalTensor, localTensor, UB_FLAG_PAD_COUNT, true);

    std::shared_ptr<TaskStub> sendSyncTask(new TaskStubSendSync(gmSignalAddr, value));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE3, sendSyncTask);
}

// 随路原子更新gm上同步信号的值，为加法，如果要减去某个值，输入值为负数即可
__aicore__ void AddSignalValue(__gm__ int32_t* gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> setValueTask(new TaskStubSetValue(value));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_S, setValueTask);

    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(gmSignalAddr, UB_FLAG_PAD_COUNT);
    for (uint32_t i = 0; i < UB_FLAG_PAD_COUNT; i++) {
        localTensor.SetValue(i, value);
    }
    //SetAtomicAdd<int32_t>();
    PipeBarrier<PIPE_ALL>();
    DataCopy(globalTensor, localTensor, UB_FLAG_PAD_COUNT, true);
    //SetAtomicNone();

    std::shared_ptr<TaskStub> sendSyncReduceTask(new TaskStubSendSyncReduce(gmSignalAddr, value));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE3, sendSyncReduceTask);
}

// 等待同步信号变成某个预期的值
__aicore__ void WaitSignalValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t expectedValue)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> recvSyncTask(new TaskStubRecvSync(gmSignalAddr, expectedValue));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE2, recvSyncTask);

    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(gmSignalAddr, UB_FLAG_PAD_COUNT);
    DataCopy(localTensor, globalTensor, UB_FLAG_PAD_COUNT, true);
    SyncFunc<HardEvent::MTE2_S>();

    std::shared_ptr<TaskStub> compValueTask(new TaskStubCompValue(expectedValue));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_S, compValueTask);
}

// 等待同步信号大于等于某个预期的值
__aicore__ void WaitSignalGEValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> recvSyncTask(new TaskStubRecvSync(gmSignalAddr, value));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE2, recvSyncTask);

    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(gmSignalAddr, UB_FLAG_PAD_COUNT);
    DataCopy(localTensor, globalTensor, UB_FLAG_PAD_COUNT, true);
    SyncFunc<HardEvent::MTE2_S>();

    std::shared_ptr<TaskStub> compValueTask(new TaskStubCompValue(value));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_S, compValueTask);
}

__aicore__ void SetFlagBatchValue(__gm__ int32_t *ctrlFlagGM, TQue<QuePosition::VECOUT, 1> &batchQue, int32_t setValue, int32_t count)
{
    GlobalTensor<int32_t> globalBatchSet;
    globalBatchSet.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT * count);
    LocalTensor<int32_t> localBatchSet = batchQue.AllocTensor<int32_t>();

    for (uint32_t i = 0; i < count; i++) {
        localBatchSet.SetValue(i * UB_FLAG_PAD_COUNT, setValue);
    }

    SyncFunc<HardEvent::S_MTE3>();

    DataCopy(globalBatchSet, localBatchSet, UB_FLAG_PAD_COUNT * count, true);

    batchQue.FreeTensor(localBatchSet);
}

// CountWait有依赖，只需编译通过，无需打桩，实际LLT走GetSignalValueWithExpected
__aicore__ int32_t GetSignalValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor)
{
    return 0;
}

// 算法分析器无法对GetSignalValue进行打桩，需使用如下函数进行替换
__aicore__ int32_t GetSignalValueWithExpected(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t expectedValue)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> recvSyncTask(new TaskStubRecvSync(gmSignalAddr, expectedValue));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE2, recvSyncTask);

    GlobalTensor<int32_t> globalTensor;
    globalTensor.SetGlobalBuffer(gmSignalAddr, UB_FLAG_PAD_COUNT);
    DataCopy(localTensor, globalTensor, UB_FLAG_PAD_COUNT, true);
    SyncFunc<HardEvent::MTE2_S>();

    std::shared_ptr<TaskStub> compValueTask(new TaskStubCompValue(expectedValue));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_S, compValueTask);
    return expectedValue;
}

}