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
#include <iostream>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
// #include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccl_common_macro.h"
#include "sim_runner_ops.h"
#include "hccl_vm_log.h"
#include <securec.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId)
{
    auto runner = sim::GetCurrRunnerTls();
    auto ret = RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device& d) {
        return d.logic_id  == (uint32_t)deviceId;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find device: {:d}", __func__, deviceId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    // 找不到context，则默认新增一个context
    sim::Context ctx{};
    ctx.device_id = ret.first.id;
    ctx.run_id = runner.id;
    auto currCtxId = RunnerDB::Add<sim::Context>(ctx);

    // 新建context，默认新建一个stream
    sim::Stream stream{};
    stream.ctx_id = currCtxId;
    stream.is_primary_default = 1;
    stream.activated = 1;
    RunnerDB::Add<sim::Stream>(stream);
    *context = (aclrtContext)currCtxId;
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext context)
{
    uint64_t ctxId = (uint64_t)(uintptr_t)context;

    auto ret = RunnerDB::GetOneByPred<sim::Stream>([ctxId](const sim::Stream& stm) {
        return stm.ctx_id  == ctxId && stm.is_primary_default == 1;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("[{}] cannot find stream by ctx: {:d}", __func__, ctxId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    RunnerDB::Delete<sim::Stream>(ret.first.id);
    RunnerDB::Delete<sim::Context>(ctxId);
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    uint64_t ctxId = (uint64_t)(uintptr_t)context;
    sim::SetCurrCtxTls(ctxId);
    return ACL_SUCCESS;
}

aclError aclrtGetCurrentContext(aclrtContext *context)
{
    auto runner = sim::GetCurrRunnerTls();
    *context = (aclrtContext)runner.current_ctx_id;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetCurrentDefaultStream(aclrtStream *stream)
{
    auto runner = sim::GetCurrRunnerTls();
    auto curCtxId = runner.current_ctx_id;
    auto stm = RunnerDB::GetOneByPred<sim::Stream>([curCtxId](const sim::Stream& stm) {
        return stm.ctx_id  == curCtxId && stm.is_primary_default == 1;
    });
    if (!stm.second) {
        HCCL_VM_ERROR("[{}] cannot find stream by ctx: {:d}", __func__, curCtxId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    *stream = (aclrtStream)stm.first.id;
    return ACL_SUCCESS;
}

aclError aclrtGetPrimaryCtxState(int32_t deviceId, uint32_t *flags, int32_t *active)
{
    auto runner = sim::GetCurrRunnerTls();
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get currContext: {:d}", __func__, runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    *active = (currCtx->is_default == 1) ? 1 : 0;
    return ACL_SUCCESS;
}

aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
     // TODO
    return ACL_SUCCESS;
}

aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t *value)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtCtxGetFloatOverflowAddr(void **overflowAddr)
{
    // TODO
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus