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
#include "hccl_vm.h"
#include "task_ventilator.h"
#include <securec.h>

thread_local std::pair<uint32_t, ShmNpuPos> curr_dev{UINT32_MAX, {}}; // first: deviceId

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
aclError aclrtFreeHost(void *hostPtr)
{
    free(hostPtr);
    printf("[aclstub][aclrtFreeHost]hostPtr: %p\n", hostPtr);
    return ACL_SUCCESS;
}

aclError aclrtMallocHost(void **hostPtr, size_t size)
{
    *hostPtr = malloc(size);
    if (*hostPtr == nullptr) {
        printf("[aclstub][aclrtMallocHost]malloc Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    memset(*hostPtr, 0, size);
    printf("[aclstub][aclrtMallocHost]hostPtr: %p\n", hostPtr);
    return ACL_SUCCESS;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    uint32_t mode = SHMManager::GetHcclVmMode();
    if (mode == HcclSim::HcclVmMode::CHECKER) {
        // checker
        HcclSim::HcclVmResult ret = MockAllocNpuMemory(curr_dev.first, size, devPtr);
        printf("[aclstub][aclrtMalloc][Checker] devPtr: %p, size: %lu\n", *devPtr, size);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[aclstub][aclrtMalloc][Checker] MockAllocNpuMemory Failed.\n");
            return ACL_ERROR_INTERNAL_ERROR;
        }
        return ACL_SUCCESS;
    } else {
        //runner
        HcclSim::HcclVmResult ret = AllocNpuMemory(curr_dev.second, size, devPtr);
        printf("[aclstub][aclrtMalloc] devPtr: %p\n", *devPtr);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[aclstub][aclrtMalloc]AllocNpuMemory Failed.\n");
            return ACL_ERROR_INTERNAL_ERROR;
        }
        return ACL_SUCCESS;
    }
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    // GetMode();
    uint32_t mode = SHMManager::GetHcclVmMode();
    printf("[GetHcclVmMode] mode: %u\n", mode);
    if (mode == HcclSim::HcclVmMode::CHECKER) {
        // checker
        printf("[aclstub][aclrtMemcpy][Checker] dst: %p, dstMax: %lu, src: %p, count: %lu\n", dst, destMax, src, count);
        return ACL_SUCCESS;
    } else {
        // runner
        int32_t ret = memcpy_s(dst, destMax, src, count);
        if (ret != 0) {
            printf("[aclstub][aclrtMemcpy] memcpy_s failed.\n");
            return ACL_ERROR_INTERNAL_ERROR;
        }
        printf("[aclstub][aclrtMemcpy] dst: %p, dstMax: %lu, src: %p, count: %lu\n", dst, destMax, src, count);
        return ACL_SUCCESS;
    }
}

const char *aclrtGetSocName()
{
    printf("[aclstub][aclrtGetSocName] Ascend910B\n");
    return "Ascend910_9391";
}

aclError aclrtGetDevicesTopo(uint32_t devId, uint32_t otherDevId, uint64_t *value)
{
    printf("[%s] not support.\n", __func__);
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    HcclSim::HcclVmResult ret = AllocStream(curr_dev.second, stream);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[aclstub][aclrtCreateStream]AllocStream Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    ShmSimStream* simStream = static_cast<ShmSimStream*>(*stream);
    printf("[aclstub][aclrtCreateStream]stream: %s\n", simStream->streamId.ToString().c_str());
    return ACL_SUCCESS;
}

int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI)
{   
    *modelRI = &rtModelFake;
    return ACL_SUCCESS;
}

HcclResult hrtGetDeviceIndexByPhyId(uint32_t devicePhyId, uint32_t &deviceLogicId)
{
    printf("[%s] not support.\n", __func__);
    return HcclResult::HCCL_SUCCESS;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId)
{
    ShmSimNotify* simNotify = static_cast<ShmSimNotify*>(notify);
    uint64_t simNotifyId = simNotify->notifyId.value;
    *notifyId = static_cast<uint32_t>(simNotifyId);
    printf("[aclstub][aclrtGetNotifyId]notifyId: %u\n", (*notifyId));
    return ACL_SUCCESS;
}

aclError aclrtCreateNotify(aclrtNotify *notify, uint64_t flag)
{
    HcclSim::HcclVmResult ret = AllocNotify(curr_dev.second, notify);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[aclstub][aclrtCreateNotify]AllocNotify Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    ShmSimNotify* simNotify = static_cast<ShmSimNotify*>(*notify);
    printf("[aclstub][aclrtCreateNotify]notify: %s.\n", simNotify->notifyId.ToString().c_str());
    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)  // todo
{
    ShmSimNotify* simNotify = static_cast<ShmSimNotify*>(notify);
    ShmSimStream* simStream = static_cast<ShmSimStream*>(stream);
    printf("[aclstub][aclrtWaitAndResetNotify]notify: %s, stream: %s\n",
        simNotify->notifyId.ToString().c_str(), simStream->streamId.ToString().c_str());

    ShmNpuPos npuPos = simStream->streamId.GetNpuPos();
    uint32_t curRank = 0;
    auto ret = GetRankIdByNpuPos(npuPos, &curRank);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] GetRankIdByNpuPos fail \n", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = simStream->streamId.value;
    taskMetaData.taskData.notify.notifyId = simNotify->notifyId.value;
    taskMetaData.taskData.notify.notifyCount = 1; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(simStream->streamId.value, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(simStream->streamId.value, taskCid);

    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
    aclrtBinHandle *binHandle)
{
    printf("[%s] not support.\n", __func__);
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceCount(uint32_t *count)
{
    HcclSim::HcclVmResult ret = GetNpuNum(count);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[aclstub][aclrtGetDeviceCount]GetNpuNum Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    printf("[aclstub][aclrtGetDeviceCount]count: %d\n", *count);
    return ACL_SUCCESS;
}

aclError aclrtSetDevice(int32_t deviceId)
{
    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByIndex(deviceId, &simNpu);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[aclstub][aclrtSetDevice]GetNpuByIndex Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    curr_dev.first = deviceId;
    curr_dev.second = simNpu->npuPos;
    printf("[aclstub][aclrtSetDevice]deviceId: %d, npuPos: %s\n", curr_dev.first, curr_dev.second.ToString().c_str());
    return ACL_SUCCESS;
}

aclError aclrtGetDevice(int32_t* device)
{
    *device = curr_dev.first;
    printf("[aclstub][aclrtGetDevice]device: %d\n", curr_dev.first);
    return ACL_SUCCESS;
}

aclError aclrtResetDevice(int32_t deviceId)
{
    ShmSimNpu* simNpu = nullptr;
    auto ret = GetNpuByIndex(deviceId, &simNpu);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[aclstub][aclrtSetDevice]GetNpuByIndex Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    curr_dev.first = deviceId;
    curr_dev.second = simNpu->npuPos;
    printf("[aclstub][aclrtResetDevice]deviceId: %d, npuPos: %s\n", curr_dev.first, curr_dev.second.ToString().c_str());
    return ACL_SUCCESS;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    ShmSimNotify* simNotify = static_cast<ShmSimNotify*>(notify);
    ShmSimStream* simStream = static_cast<ShmSimStream*>(stream);
    printf("[aclstub][aclrtRecordNotify]notify: %s, stream: %s\n",
        simNotify->notifyId.ToString().c_str(), simStream->streamId.ToString().c_str());

    ShmNpuPos npuPos = simStream->streamId.GetNpuPos();
    uint32_t curRank = 0;
    auto ret = GetRankIdByNpuPos(npuPos, &curRank);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] GetRankIdByNpuPos fail \n", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = simStream->streamId.value;
    taskMetaData.taskData.notify.notifyId = simNotify->notifyId.value;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(simStream->streamId.value, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(simStream->streamId.value, taskCid);

    return ACL_SUCCESS;
}

aclError aclrtCreateEvent(aclrtEvent *event)
{
    return ACL_SUCCESS;
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    return ACL_SUCCESS;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent)
{
    *ms = 1;
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    HcclSim::HcclVmResult ret = ReleaseStream(stream);
    printf("[aclstub][aclrtDestroyStream]stream: %p\n", stream);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[aclstub][aclrtDestroyStream]ReleaseStream Failed.\n");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    return ACL_SUCCESS;
}

aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode)
{
    printf("[aclstub][aclrtSetStreamFailureMode]stream: %p, mode: %lu\n", stream, mode);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    // GetMode();
    std::string mode = "checker";
    const int WAIT_COUNTDOWN = 10;    // 等待20s
    ShmSimStream* streamObj = static_cast<ShmSimStream*>(stream);
    int cnt = WAIT_COUNTDOWN;
    while (!TaskStatusCache::GetInstance().IsStreamFinish(streamObj->streamId.value)) {
        sleep(1);
        cnt--;  // TODO checker模式下不会执行，hccl_test无法结束
        if (cnt <= 0) {
            uint32_t mode = SHMManager::GetHcclVmMode();
            if (mode == HcclSim::HcclVmMode::CHECKER) {
                // checker
                printf("[aclstub][aclrtSynchronizeStream][Checker] Wait Timeout.\n");
                break;
            } else {
                // runner
                printf("[aclstub][aclrtSynchronizeStream] Wait Timeout. There might be something wrong. You can manually stop the program.\n");
            }
        }
    }
    printf("[aclstub][aclrtSynchronizeStream]streamId: %lu\n", streamObj->streamId.value);
    return ACL_SUCCESS;
}

aclError aclInit(const char *configPath)
{
    printf("[aclstub][aclInit]Success\n");
    return ACL_SUCCESS;
}

aclError aclFinalize()
{
    printf("[aclstub][aclFinalize]Success\n");
    return ACL_SUCCESS;
}

bool IsStreamCapture(aclrtStream stream)
{
    printf("[aclstub][IsStreamCapture]\n");
    return false;
}

#ifdef __cplusplus
}
#endif  // __cplusplus