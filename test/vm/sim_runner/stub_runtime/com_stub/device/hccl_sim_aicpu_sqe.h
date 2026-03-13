/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim aicpu sqe header
 */

#ifndef HCCL_SIM_AICPU_SQE_H
#define HCCL_SIM_AICPU_SQE_H
#include "ascend_hal.h"

void CopyA3SqBufferStub(uint32_t devId, struct halSqCqConfigInfo *info);
void CopyA5SqBufferStub(uint32_t devId, struct halSqCqConfigInfo *info);

#endif // HCCL_SIM_AICPU_SQE_H