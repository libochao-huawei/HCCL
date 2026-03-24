#ifndef OPS_HCCL_P2P_AIV_KERNEL_BASE_H
#define OPS_HCCL_P2P_AIV_KERNEL_BASE_H

#include "kernel_operator.h"
#include "../inc/kernel_types.h"

using namespace AscendC;

namespace ops_hccl_p2p_aiv {

class P2pAivKernelBase {
public:
    __aicore__ inline void Init(const P2pAivKernelParam &param)
    {
        param_ = param;
        localSync_ = reinterpret_cast<volatile __gm__ P2pAivSyncState *>(param.localSyncAddr);
        remoteSync_ = reinterpret_cast<volatile __gm__ P2pAivSyncState *>(param.remoteSyncAddr);
    }

protected:
    template <typename T>
    __aicore__ inline void CopyGmToGm(__gm__ T *dst, __gm__ T *src, uint64_t count)
    {
        for (uint64_t idx = 0; idx < count; ++idx) {
            dst[idx] = src[idx];
        }
    }

    __aicore__ inline void WaitReady()
    {
        while (remoteSync_->ready != kP2pAivFlagReadyValue) {
        }
    }

    __aicore__ inline void WaitDone()
    {
        while (remoteSync_->done != kP2pAivFlagDoneValue) {
        }
    }

    __aicore__ inline void RecordReady()
    {
        localSync_->ready = kP2pAivFlagReadyValue;
    }

    __aicore__ inline void RecordDone()
    {
        localSync_->done = kP2pAivFlagDoneValue;
    }

protected:
    P2pAivKernelParam param_ {};
    volatile __gm__ P2pAivSyncState *localSync_ = nullptr;
    volatile __gm__ P2pAivSyncState *remoteSync_ = nullptr;
};

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_KERNEL_BASE_H
