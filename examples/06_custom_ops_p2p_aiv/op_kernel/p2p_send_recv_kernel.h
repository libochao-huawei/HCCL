#ifndef OPS_HCCL_P2P_AIV_SEND_RECV_KERNEL_H
#define OPS_HCCL_P2P_AIV_SEND_RECV_KERNEL_H

#include "p2p_aiv_kernel_base.h"

namespace ops_hccl_p2p_aiv {

template <typename T>
class P2pSendRecvKernel : public P2pAivKernelBase {
public:
    __aicore__ inline void Run(const P2pAivKernelParam &param)
    {
        Init(param);
        if (param.taskType == kP2pAivTaskSend) {
            RunSend();
            return;
        }
        if (param.taskType == kP2pAivTaskRecv) {
            RunRecv();
        }
    }

private:
    __aicore__ inline void RunSend()
    {
        const uint64_t count = param_.lenBytes / sizeof(T);
        CopyGmToGm(reinterpret_cast<__gm__ T *>(param_.localBufferAddr), reinterpret_cast<__gm__ T *>(param_.inputAddr), count);
        RecordReady();
        WaitDone();
    }

    __aicore__ inline void RunRecv()
    {
        WaitReady();
        const uint64_t count = param_.lenBytes / sizeof(T);
        CopyGmToGm(reinterpret_cast<__gm__ T *>(param_.outputAddr), reinterpret_cast<__gm__ T *>(param_.remoteBufferAddr), count);
        RecordDone();
    }
};

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_SEND_RECV_KERNEL_H
