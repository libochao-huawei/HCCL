/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAUNCH_KERNEL_BATCH_H
#define LAUNCH_KERNEL_BATCH_H

#include "common.h"
#include "ccu_kernel.h"

namespace ops_hccl_allgather_batch {

struct CcuContextData {
    bool initialized = false;
    CcuKernelHandle kernelHandle = 0;
};

HcclResult InitCcuContext(HcclComm comm, const char *engineCtxTag, const OpParam &param, CcuContextData *&ctx);
HcclResult LaunchKernel(HcclComm comm, const OpParam &param, const CcuContextData &ctx, aclrtStream stream);
}

#endif
