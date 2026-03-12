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

aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtProcessReport(int32_t timeout)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtProcessHostFunc(int32_t timeout)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeHostFunc(uint64_t hostFuncThreadId, aclrtStream exeStream)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetOpTimeoutInterval(uint64_t *interval)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOut(uint32_t timeout)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout,  uint64_t *actualTimeout)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOutWithMs(uint32_t timeout)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetThreadLastTaskId(uint32_t *taskId)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}

aclError aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void *args)
{
    //TODO
    HCCL_VM_TRACE("[{}] not supported", __func__);
    return ACL_SUCCESS;
}
#ifdef __cplusplus
}
#endif  // __cplusplus