/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <vector>
#include <atomic>
#include <chrono>
#include <ctime>
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

uint32_t aclrtGetTaskIdFromExceptionInfo(const aclrtExceptionInfo *info)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

uint32_t aclrtGetStreamIdFromExceptionInfo(const aclrtExceptionInfo *info)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

uint32_t aclrtGetThreadIdFromExceptionInfo(const aclrtExceptionInfo *info)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

uint32_t aclrtGetDeviceIdFromExceptionInfo(const aclrtExceptionInfo *info)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

uint32_t aclrtGetErrorCodeFromExceptionInfo(const aclrtExceptionInfo *info)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtPeekAtLastError(aclrtLastErrLevel level)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetLastError(aclrtLastErrLevel level)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetMemUceInfo(int32_t deviceId, aclrtMemUceInfo *memUceInfoArray, size_t arraySize, size_t *retSize)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtMemUceRepair(int32_t deviceId, aclrtMemUceInfo *memUceInfoArray, size_t arraySize)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtDeviceTaskAbort(int32_t deviceId, uint32_t timeout)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclRecoverAllHcclTasks(int32_t deviceId)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetErrorVerbose(int32_t deviceId, aclrtErrorInfo *errorInfo)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtRepairError(int32_t deviceId, const aclrtErrorInfo *errorInfo)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}
#ifdef __cplusplus
}
#endif  // __cplusplus