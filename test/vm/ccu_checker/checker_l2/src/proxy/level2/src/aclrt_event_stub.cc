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
// #include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

aclError aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    auto runner = sim::GetCurrRunnerTls();
    if (runner.id == 0 || runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("[{}] invalid param", __func__);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get CurrContext:{:d}", __func__, runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    
    auto& crrCtxId = currCtx->id;

    auto now = std::chrono::system_clock::now();
    std::time_t timeNow = std::chrono::system_clock::to_time_t(now);

    sim::Event tmp{};
    tmp.create_ctx_id = crrCtxId;
    tmp.event_flag = flag;
    tmp.status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    tmp.created_time = static_cast<uint64_t>(timeNow);
    auto res = RunnerDB::Add<sim::Event>(tmp);

    *event = (aclrtEvent)res;
    HCCL_VM_INFO("[{}] event: {:d}", __func__, res);
    return ACL_SUCCESS; 
}

aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flag)
{
    return aclrtCreateEventWithFlag(event, flag);
}

aclError aclrtCreateEvent(aclrtEvent *event)
{
    return aclrtCreateEventWithFlag(event, 0);
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    uint64_t eventId = (uint32_t)(uintptr_t)event;
    HCCL_VM_INFO("[{}] event: {:d}", __func__, eventId);
    RunnerDB::Delete<sim::Event>(eventId);
    return ACL_SUCCESS;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    uint64_t streamIdx = (uint64_t)(uintptr_t)stream;
    uint64_t eventIdx = (uint32_t)(uintptr_t)event;

    auto currEvent = RunnerDB::GetById<sim::Event>(eventIdx);
    if (!currEvent.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get Event: {:d}", __func__, eventIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto currStm = RunnerDB::GetById<sim::Stream>(streamIdx);
    if (!currStm.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get stream:{:d}", __func__, streamIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskCid taskCid{0, curRank, 0};
    sim::Task task{};
    task.stream_id  = streamIdx;
    task.cid        = 0;
    task.type       = (uint8_t)HccLTaskMetaType::EVENT_RECORD;

    auto taskId = RunnerDB::Add<sim::Task>(task);
    RunnerDB::Update<sim::Event>(eventIdx, [](sim::Event &evt) { evt.status = ACL_EVENT_RECORDED_STATUS_NOT_READY;});

    return ACL_SUCCESS;
}

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream)
{
    uint64_t eventIdx = (uint32_t)(uintptr_t)event;
    auto currEvent = RunnerDB::GetById<sim::Event>(eventIdx);
    if (!currEvent.has_value()) {
        HCCL_VM_ERROR("[{}] can not get Event:{:d}", __func__, eventIdx);
        return ACL_ERROR_INVALID_PARAM;
    }
    RunnerDB::Update<sim::Event>(eventIdx, [](sim::Event &evt) { evt.status = ACL_EVENT_RECORDED_STATUS_COMPLETE;});
    return ACL_SUCCESS;
}

aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status)
{
    uint64_t eventId = (uint32_t)(uintptr_t)event;
    auto res = RunnerDB::GetById<sim::Event>(eventId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("[{}] can not get event:{:d}", __func__, eventId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *status = (aclrtEventRecordedStatus)(res->status);
    return ACL_SUCCESS;
}

aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *status)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeEvent(aclrtEvent event)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent)
{
    // TODO
    *ms = 1;
    return ACL_SUCCESS;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtSetOpWaitTimeout(uint32_t timeout)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtEventGetTimestamp(aclrtEvent event, uint64_t *timestamp)
{
    // TODO
    return ACL_SUCCESS;
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId)
{
    *eventId = (uint32_t)(uintptr_t)event;
    return ACL_SUCCESS;
}

aclError aclrtGetEventAvailNum(uint32_t *eventCount)
{
    auto runner = sim::GetCurrRunnerTls();
    if (runner.id == 0 || runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("[{}] invalid param", __func__);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not get CurrContext:{:d}", __func__, runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    
    auto& devId = currCtx->device_id;

    auto device = RunnerDB::GetById<sim::Device>(devId);
    if (!device.has_value()) {
        HCCL_VM_ERROR("[{}] can not get device:{:d}", __func__, devId);
        return ACL_ERROR_INVALID_PARAM;
    }

    if (memcmp(device->soc_version, "A3", strlen("A3") == 0)) {
        *eventCount = 65535;// todo �ȼ�ʵ������Ҫ����EventSyncTask�ж�����
    }

    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus