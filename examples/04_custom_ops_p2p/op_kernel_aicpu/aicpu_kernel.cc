/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <hccl/hcomm_primitives.h>

#include "log.h"
#include "common.h"
#include "exec_op.h"

using namespace ops_hccl_p2p;

extern "C" unsigned int HcclLaunchP2PAicpuKernel(OpParam *param)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    CHK_RET(HcommAcquireComm(param->commName));
    CHK_RET(HcommBatchModeStart(param->tag));

    // 主thread等待Host stream的通知
    CHK_RET(HcommThreadNotifyWaitOnThread(param->resCtx->aicpuThread, 0, CUSTOM_TIMEOUT));

    // 执行算法编排
    CHK_RET(ExecOp(*param, param->resCtx));

    // 主thread通知Host stream
    CHK_RET(HcommThreadNotifyRecordOnThread(param->resCtx->aicpuThread, param->resCtx->aicpuThreadOnCpu, 0));

    CHK_RET(HcommBatchModeEnd(param->tag));
    CHK_RET(HcommReleaseComm(param->commName));

    HCCL_INFO("%s success, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    return 0;
}
