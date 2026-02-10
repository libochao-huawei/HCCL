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
// #include "hccl_vm.h"
#include "task_ventilator.h"
#include "sim_runner_ops.h"
#include "hccl_vm_log.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

aclError aclrtCreateNotify(aclrtNotify *notify, uint64_t flag)
{
    auto runner = sim::GetCurrRunnerTls();
    if (runner.id == 0 || runner.current_ctx_id == 0) {
        HCCL_VM_ERROR("[aclrtCreateNotify] invalid param");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = RunnerDB::GetById<sim::Context>(runner.current_ctx_id);
    if (!currCtx.has_value()) {
        HCCL_VM_ERROR("can not get CurrContext: {:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto& currCtxId = currCtx->id;

    sim::Notify tmp{};
    tmp.create_ctx_id = currCtxId;
    auto res = RunnerDB::Add<sim::Notify>(tmp);

    *notify = (aclrtNotify)res;
    HCCL_VM_DEBUG("[aclstub][aclrtCreateNotify] notify: {:d}", res);
    return ACL_SUCCESS;
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("[aclstub][aclrtDestroyNotify] notify: {:d}", notifyId);
    RunnerDB::Delete<sim::Notify>(notifyId);
    return ACL_SUCCESS;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId)
{
    *notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("[aclstub][aclrtGetNotifyId] notifyId: {:d}", *notifyId);
    return ACL_SUCCESS;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[{}] InsertTaskToCollection fail", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    sim::Task task{};
    task.stream_id  = streamId;
    task.cid        = taskCid.value;
    task.type       = (uint8_t)HccLTaskMetaType::NOTIFY_RECORD;

    auto taskId = RunnerDB::Add<sim::Task>(task);

    // 下发cid
    //TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);
    // 记录状态
    //TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);
    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    // reset notify
    auto res = RunnerDB::Update<sim::Notify>(notifyId, [](sim::Notify &notify) { notify.value = 0; });
    if (!res) {
        HCCL_VM_ERROR("can not get notify: {:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[{}] InsertTaskToCollection fail", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    sim::Task task{};
    task.stream_id  = streamId;
    task.cid        = taskCid.value;
    task.type       =  (uint8_t)HccLTaskMetaType::NOTIFY_WAIT;

    auto taskId = RunnerDB::Add<sim::Task>(task);

    // 下发cid
    //TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);
    // 记录状态
    //TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);
    return ACL_SUCCESS;
}

aclError aclrtNotifyBatchReset(aclrtNotify *notifies, size_t num)
{
    for (int i = 0; i < num; i++) {
        uint64_t notifyId = (uint32_t)(uintptr_t)notifies[i];

        // reset notify
        auto res = RunnerDB::Update<sim::Notify>(notifyId, [](sim::Notify &notify) { notify.value = 0; });
        if (!res) {
            HCCL_VM_ERROR("can not get notify: {:d}", notifyId);
            return ACL_ERROR_INVALID_PARAM;
        }
    }
    return ACL_SUCCESS;
}

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char *key, size_t len, uint64_t flags)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;

    std::string notifyKeyStr = std::to_string(notifyId);

    sim::IpcNotify tmp{};
    tmp.notify_id = notifyId;
    // tmp.flag = (uint8_t)flags;
    memcpy(tmp.name_or_key, notifyKeyStr.data(), notifyKeyStr.length());
    memcpy(key, notifyKeyStr.data(), notifyKeyStr.length());
    tmp.create_pid = getpid();
    auto res = RunnerDB::Add<sim::IpcNotify>(tmp);
    HCCL_VM_DEBUG("[aclstub][aclrtNotifyGetExportKey] notify: {:d}", notifyId);
    return ACL_SUCCESS;
}

aclError aclrtNotifySetImportPid(aclrtNotify notify, int32_t *pid, size_t num)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    HCCL_VM_DEBUG("[aclstub][aclrtNotifySetImportPid] notify: {:d}", notifyId);

    auto ipcNotify = RunnerDB::GetOneByPred<sim::IpcNotify>([notifyId](const sim::IpcNotify& ipc) {
        return ipc.notify_id  == notifyId;
    });
    if (!ipcNotify.second) {
        HCCL_VM_ERROR("can not get notify in ipc notify: {:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    for (int i = 0; i < num; i++) {
        sim::IpcNotifyVistorList vistor;
        vistor.ipc_id = ipcNotify.first.id;
        vistor.visitor_pid = pid[i];
        RunnerDB::Add<sim::IpcNotifyVistorList>(vistor);
    }

    return ACL_SUCCESS;
}

aclError aclrtNotifyImportByKey(aclrtNotify *notify, const char *key, uint64_t flags)
{
    auto ipcNotify = RunnerDB::GetOneByPred<sim::IpcNotify>([key](const sim::IpcNotify& ipc) {
        return memcmp(key, ipc.name_or_key, strlen(key)) == 0;
    });
    if (!ipcNotify.second) {
        HCCL_VM_ERROR("can not get notify in ipc notify key: {}", key);
        return ACL_ERROR_INVALID_PARAM;
    }
    *notify = (aclrtNotify)ipcNotify.first.notify_id;
    HCCL_VM_DEBUG("[aclstub][aclrtNotifyImportByKey] notify: {:d}", ipcNotify.first.notify_id);
    return ACL_SUCCESS;
}

////////////////////////////rt 接口/////////////////////////////////
rtError_t rtsNotifyCreate(rtNotify_t *notify, uint64_t flag)
{
    return aclrtCreateNotify(notify, flag);
}

rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm)
{
    return aclrtWaitAndResetNotify(notify, stm, 0);
}

rtError_t  rtGetNotifyAddress(rtNotify_t notify, uint64_t * const notifyAddres)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    *notifyAddres = notifyId;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfo(rtNotify_t notify, uint32_t *phyDevId, uint32_t *tsId)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    // reset notify
    auto res = RunnerDB::GetById<sim::Notify>(notifyId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("can not get notify: {:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto createCtx = RunnerDB::GetById<sim::Context>(res->create_ctx_id);
    if (!createCtx.has_value()) {
        HCCL_VM_ERROR("can not get create ctx: {:d}", res->create_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(createCtx->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("can not get device id: {:d}", createCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    *phyDevId = devRes->physical_id;
    *tsId = 0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo *notifyInfo)
{
    uint64_t notifyId = (uint32_t)(uintptr_t)notify;
    // reset notify
    auto res = RunnerDB::GetById<sim::Notify>(notifyId);
    if (!res.has_value()) {
        HCCL_VM_ERROR("can not get notify: {:d}", notifyId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto createCtx = RunnerDB::GetById<sim::Context>(res->create_ctx_id);
    if (!createCtx.has_value()) {
        HCCL_VM_ERROR("can not get create ctx: {:d}", res->create_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = RunnerDB::GetById<sim::Device>(createCtx->device_id);
    if (!devRes.has_value()) {
        HCCL_VM_ERROR("can not get device id: {:d}", createCtx->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    notifyInfo->phyId   = devRes->physical_id;
    notifyInfo->tsId    = 0;
    notifyInfo->idType  = 0;
    notifyInfo->shrId   = (uint32_t)notifyId;
    notifyInfo->flag    =  0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t* devAddrOffset)
{
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif  // __cplusplus