#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_LAUNCH_KERNEL_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_LAUNCH_KERNEL_H

#include <string>
#include "common.h"

namespace ops_hccl_allgather_2in2out {

extern thread_local aclrtNotify g_notifies[kControlNotifyNum];

// 创建 Host/AICPU 控制 notify。
// 资源准备阶段需要把 notify id 写入 device context，真正 launch 前也需要确保句柄已就绪。
HcclResult EnsureHostControlNotifiesCreated();
HcclResult LaunchKernel(OpParam &param, aclrtStream stream);

} // namespace ops_hccl_allgather_2in2out

#endif
