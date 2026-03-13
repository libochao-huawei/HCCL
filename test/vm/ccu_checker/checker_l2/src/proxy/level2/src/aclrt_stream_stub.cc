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
#include "runtime/base.h"
#include "hccl_proxy_pub.h"
#include "hccl_sim_world_pub.h"
#include "task_status_cache.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include <securec.h>
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)
{
    auto runner = sim::GetCurrRunnerTls();
    if (runner.id == 0 || runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("[aclrtCreateStream] invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not get CurrContext:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& currCtxId = currCtx->id;

    sim::Stream streamTmp{};
    streamTmp.ctx_id = currCtxId;
    streamTmp.activated = 1;
    streamTmp.priority = priority;
    streamTmp.user_tag = flag;
    auto res = RunnerDB::Add<sim::Stream>(streamTmp);

    *stream = (aclrtStream)res;
    HCCL_VM_DEBUG("[aclstub][aclrtCreateStreamWithConfig]stream: {:d}", res);
    sim::SetLastStreamIdTls(res);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    return aclrtCreateStreamWithConfig(stream, 0, 0);
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtDestroyStream]stream: {:d}", streamId);
    RunnerDB::Delete<sim::Stream>(streamId);
    return ACL_SUCCESS;
}

aclError aclrtDestroyStreamForce(aclrtStream stream)
{
    return aclrtDestroyStream(stream);
}

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream)
{
    uint64_t activStreamId = (uint64_t)(uintptr_t)activeStream;
    HCCL_VM_DEBUG("[aclstub][aclrtActiveStream]stream: {:d}", activStreamId);
    auto res = RunnerDB::Update<sim::Stream>(activStreamId, [](sim::Stream &stm) { stm.activated = 1; });
    if (!res) {
        HCCL_VM_ERROR("can not get stream:{:d}", activStreamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtSetStreamFailureMode]stream: {:d}, mode: {:d}", streamId, mode);
    RunnerDB::Update<sim::Stream>(streamId, [streamId, mode](sim::Stream &stm) { stm.failure_mode = mode; });

    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    // GetMode();
    std::string mode = "checker";
    const int WAIT_COUNTDOWN = 10;    // 等待20s
    HCCL_VM_DEBUG("[aclstub][aclrtSynchronizeStream]streamId: {:d}", streamId);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    return aclrtSynchronizeStreamWithTimeout(stream, 0);
}

aclError aclrtStreamAbort(aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtStreamAbort]stream: {:d}", streamId);
    RunnerDB::Update<sim::Stream>(streamId, [streamId](sim::Stream &stm) { stm.activated = 0; });

    return ACL_SUCCESS;
}

aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus *status)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtStreamQuery]stream: {:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("can not get stream:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    *status = (aclrtStreamStatus)res->task_complete_status;
    return ACL_SUCCESS;
}

aclError aclrtGetStreamAvailableNum(uint32_t *streamCount)
{
    auto runner = sim::GetCurrRunnerTls();
    if (runner.id == 0 || runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("[aclrtGetStreamAvailableNum] invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not get CurrContext:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& devId = currCtx->device_id;

    auto device = RunnerDB::GetById<sim::Device>(devId);
    if (!device.has_value()) {
        HCCL_VM_ERROR("can not get device:{:d}", devId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto currCtxs = RunnerDB::GetByPred<sim::Context>([devId](const sim::Context& ctx) {
        return ctx.device_id  == devId;
    });

    auto currStreams = RunnerDB::GetByPred<sim::Stream>([currCtxs](const sim::Stream& stm) {
        for (auto& ctx : currCtxs) {
            if (ctx.id == stm.ctx_id) {
                return true;
            }
        }
        return false;
    });

    if (memcmp(device->soc_version, "A3", strlen("A3") == 0)) {
        *streamCount = 1984 - currStreams.size();
    }

    return ACL_SUCCESS;
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId)
{
    *streamId = (uint32_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtStreamGetId]streamId: {:d}", *streamId);
    return ACL_SUCCESS;
}

aclError aclrtSetStreamOverflowSwitch(aclrtStream stream, uint32_t flag)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtSetStreamOverflowSwitch]stream: {:d}, flag: {:d}", streamId, flag);
    RunnerDB::Update<sim::Stream>(streamId, [streamId, flag](sim::Stream &stm) { stm.overflow_switch = flag; });
    return ACL_SUCCESS;
}

aclError aclrtGetStreamOverflowSwitch(aclrtStream stream, uint32_t *flag)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtGetStreamOverflowSwitch]stream: {:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("can not get stream:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }
    *flag = res->overflow_switch;
    return ACL_SUCCESS;
}

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtSetStreamAttribute]stream: {:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("can not get stream:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    RunnerDB::Update<sim::Stream>(streamId, [streamId, stmAttrType, value](sim::Stream &stm) {
        if (stmAttrType == ACL_STREAM_ATTR_FAILURE_MODE) {
            stm.failure_mode = value->failureMode;
        } else if (stmAttrType == ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK ) {
            stm.overflow_switch = value->overflowSwitch;
        } else if (stmAttrType == ACL_STREAM_ATTR_USER_CUSTOM_TAG) {
            stm.user_tag = value->userCustomTag;
        }
    });

    return ACL_SUCCESS;
}

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[aclstub][aclrtGetStreamAttribute]stream: {:d}", streamId);
    auto res = RunnerDB::GetById<sim::Stream>(streamId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("can not get stream:{:d}", streamId);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (stmAttrType == ACL_STREAM_ATTR_FAILURE_MODE) {
        value->failureMode = res->failure_mode;
    } else if (stmAttrType == ACL_STREAM_ATTR_FLOAT_OVERFLOW_CHECK ) {
        value->overflowSwitch = res->overflow_switch;
    } else if (stmAttrType == ACL_STREAM_ATTR_USER_CUSTOM_TAG) {
        value->userCustomTag = res->user_tag;
    }
    return ACL_SUCCESS;
}

aclError aclrtStreamStop(aclrtStream stream)
{
    // TODO
    return ACL_SUCCESS;
}

//////////////////////////////////////////////////////////////
rtError_t rtStreamSynchronize(rtStream_t stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    HCCL_VM_DEBUG("[rtstub][rtStreamSynchronize]streamId: {:d}", streamId);
    return aclrtSynchronizeStream((aclrtStream)stream);
}

rtError_t rtStreamCreateWithFlags(rtStream_t *stm, int32_t priority, uint32_t flags)
{
    return aclrtCreateStreamWithConfig((aclrtStream *)stm, priority, flags);
}

rtError_t rtStreamGetSqid(const rtStream_t stream, uint32_t *sqId)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    *sqId = 0;
    HCCL_VM_DEBUG("[rtstub][aclrtSynchronizeStream]streamId: {:d}, sqId:0", streamId);
    return ACL_SUCCESS;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId)
{
    HCCL_VM_DEBUG("[rtstub][rtGetTaskIdAndStreamID]streamId: sqId:0");

    *streamId = (uint32_t)sim::GetLastStreamIdTls();
    *taskId = (uint32_t)sim::GetLastTaskIdTls();
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus