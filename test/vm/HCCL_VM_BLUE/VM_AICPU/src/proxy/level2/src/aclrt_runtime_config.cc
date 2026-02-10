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
#include <unistd.h>
#include <iostream>
#include "runtime/base.h"
#include "runtime/event.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
#include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "sim_runner_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t *value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtResetDeviceResLimit(int32_t deviceId)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t *value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtResetStreamResLimit(aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtUseStreamResInCurrentThread(aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtUnuseStreamResInCurrentThread(aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t *value)
{
    /*
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        printf("[ERROR][aclrtGetResInCurrentThread] can not find current context:%lu\n", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto dev = RunnerDB::GetById<sim::Device>(currCtx->device_id);
    if (!dev.has_value()) {
        printf("[ERROR][aclrtGetResInCurrentThread] can not find current device id:%lu\n", currCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (type == ACL_RT_DEV_RES_CUBE_CORE) {
        *value = GetCubeCoreCount(dev->logic_id);
    } else if (type == ACL_RT_DEV_RES_VECTOR_CORE) {
        *value = GetVectorCoreCount(dev->logic_id);
    }
    */
    return ACL_SUCCESS;
}

aclError aclrtGetOpTimeoutInterval(uint64_t *interval)
{
    *interval = 0;
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus