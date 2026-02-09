/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_LAUNCH_KERNEL_H
#define OPS_HCCL_P2P_LAUNCH_KERNEL_H

#include <string>

// 调用方式配置选项
enum class KernelLaunchMode {
    ACL_API,    // 使用传统 ACL API 方式
    BINARY      // 使用二进制 <<<>>> 调用方式
};

namespace ops_hccl_p2p {

// 声明 ASC 内核函数
extern __global__ __aicpu__ uint32_t HcclLaunchP2PAicpuKernelAsc(void *args);

HcclResult LaunchKernel(OpParam &param, aclrtStream stream, KernelLaunchMode mode = KernelLaunchMode::ACL_API);

extern thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];

}

#endif
