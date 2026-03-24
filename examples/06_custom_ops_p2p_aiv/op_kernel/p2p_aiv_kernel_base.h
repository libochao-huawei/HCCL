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
    }

protected:
    template <typename T>
    __aicore__ inline void CopyGmToGm(__gm__ T *dst, __gm__ T *src, uint64_t count)
    {
        for (uint64_t idx = 0; idx < count; ++idx) {
            dst[idx] = src[idx];
        }
    }

    __aicore__ inline volatile __gm__ int32_t *GetLocalFlag(uint64_t offset)
    {
        return reinterpret_cast<volatile __gm__ int32_t *>(param_.localCommInfoAddr + offset);
    }

    __aicore__ inline volatile __gm__ int32_t *GetRemoteFlag(uint64_t offset)
    {
        return reinterpret_cast<volatile __gm__ int32_t *>(param_.remoteCommInfoAddr + offset);
    }

    __aicore__ inline void WaitFlag(uint64_t offset)
    {
        volatile __gm__ int32_t *flag = GetRemoteFlag(offset);
        while (*flag != static_cast<int32_t>(param_.tag)) {
        }
    }

    __aicore__ inline void RecordFlag(uint64_t offset)
    {
        volatile __gm__ int32_t *flag = GetLocalFlag(offset);
        *flag = static_cast<int32_t>(param_.tag);
    }

protected:
    P2pAivKernelParam param_ {};
};

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_KERNEL_BASE_H
