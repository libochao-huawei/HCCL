#ifndef AIV_SYNC_STUB_H
#define AIV_SYNC_STUB_H
#include "aiv_base_stub.h"
#include "aiv_memory_stub.h"
 
namespace AscendC {
// Event Id
typedef enum {
  EVENT_ID0 = 0,
  EVENT_ID1,
  EVENT_ID2,
  EVENT_ID3,
} event_t;
 
// 同步相关接口打桩实现
void wait_flag(pipe_t w, pipe_t dst, event_t event, bool isGenFromAlloc = false);
 
void set_flag(pipe_t w, pipe_t dst, event_t event, bool isGenFromFree = false);
 
void pipe_barrier(pipe_t pipe);
 
template <pipe_t pipe> __aicore__ void PipeBarrier()
{
    pipe_barrier(pipe);
}
 
template <HardEvent event> __aicore__ void SetFlag(int32_t eventID);
 
template <HardEvent event> __aicore__ void WaitFlag(int32_t eventID);

__aicore__ void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0);
}
 
#endif