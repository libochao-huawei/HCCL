#include "aiv_sync_stub.h"
#include "llt_common.h"
#include "task_stub.h"
#include "aiv_task_stub.h"
#include "aiv_task_queue_stub.h"
#include "rank_info_recorder.h"


using namespace checker;
using namespace hccl;

namespace AscendC {
void wait_flag(pipe_t w, pipe_t dst, event_t event, bool isGenFromAlloc)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> task(new TaskStubWaitFlag(w, dst, 0, (BlockId)block_idx, isGenFromAlloc));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, dst, task);
}

void set_flag(pipe_t w, pipe_t dst, event_t event, bool isGenFromFree)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> task(new TaskStubSetFlag(w, dst, 0, (BlockId)block_idx, isGenFromFree));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, w, task);
}

void pipe_barrier(pipe_t pipe)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    std::shared_ptr<TaskStub> task(new TaskStubPipeBarrier(pipe));

    if (pipe == PIPE_ALL) {
        AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_S, task);
        AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE2, task);
        AivTaskQueueStub::AppendAivTask(curRank, block_idx, PIPE_MTE3, task);
    } else {
        AivTaskQueueStub::AppendAivTask(curRank, block_idx, pipe, task);
    }
}

void GetSrcDstPipe(HardEvent event, pipe_t *src, pipe_t *dst)
{
    switch (event) {
        case HardEvent::MTE3_MTE2 : {
            *src = PIPE_MTE3;
            *dst = PIPE_MTE2;
            break;
        }
        case HardEvent::MTE3_S : {
            *src = PIPE_MTE3;
            *dst = PIPE_S;
            break;
        }
        case HardEvent::S_MTE2 : {
            *src = PIPE_S;
            *dst = PIPE_MTE2;
            break;
        }
        case HardEvent::S_MTE3 : {
            *src = PIPE_S;
            *dst = PIPE_MTE3;
            break;
        }
        case HardEvent::MTE2_S : {
            *src = PIPE_MTE2;
            *dst = PIPE_S;
            break;
        }
        case HardEvent::MTE2_MTE3 : {
            *src = PIPE_MTE2;
            *dst = PIPE_MTE3;
            break;
        }
        default: {
            HCCL_ERROR("[AIV][Checker][GetSrcDstPipe] event[%d] is invalid", event);
            return;
        }
    }
}

template <HardEvent event>
__aicore__ void SetFlag(int32_t eventID)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();

    pipe_t src, dst;
    GetSrcDstPipe(event, &src, &dst);

    std::shared_ptr<TaskStub> task(new TaskStubSetFlag(src, dst, 0, (BlockId)block_idx));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, src, task);
}

template <HardEvent event>
__aicore__ void WaitFlag(int32_t eventID)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();

    pipe_t src, dst;
    GetSrcDstPipe(event, &src, &dst);

    std::shared_ptr<TaskStub> task(new TaskStubWaitFlag(src, dst, 0, (BlockId)block_idx));
    AivTaskQueueStub::AppendAivTask(curRank, block_idx, dst, task);
}

template void SetFlag<HardEvent::MTE3_MTE2>(int32_t eventID);
template void SetFlag<HardEvent::S_MTE2>(int32_t eventID);
template void SetFlag<HardEvent::MTE3_S>(int32_t eventID);
template void SetFlag<HardEvent::S_MTE3>(int32_t eventID);
template void SetFlag<HardEvent::MTE2_S>(int32_t eventID);

template void WaitFlag<HardEvent::MTE3_MTE2>(int32_t eventID);
template void WaitFlag<HardEvent::S_MTE2>(int32_t eventID);
template void WaitFlag<HardEvent::MTE3_S>(int32_t eventID);
template void WaitFlag<HardEvent::S_MTE3>(int32_t eventID);
template void WaitFlag<HardEvent::MTE2_S>(int32_t eventID);

__aicore__ void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores)
{
    return;
}
}