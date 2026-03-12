/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "hccl/hccl_types.h"
#include "hccl/hcomm_primitives.h"
#include "sim_communicator.h"
#include "sim_thread.h"
#include "sim_channel.h"
#include "task_ventilator.h"
#include "task_status_cache.h"
#include "hccl_shm_pub.h"
#include "hccl_common_defs.h"

using namespace std;
using namespace HcclProxy;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout)
{
    // timeout 暂时未使用
    static_cast<void>(timeout);

    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();
    uint64_t notifyId = threadPtr->GetNotifyIdByIndex(notifyIdx);

    auto streamPtr = threadPtr->GetStream();
    uint64_t streamId = (streamPtr->streamId).value;

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notifyId;
    taskMetaData.taskData.notify.notifyCount = 1; // notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    uint64_t streamId = (streamPtr->streamId).value;

    auto dstThreadPtr = reinterpret_cast<SimHcclThread*>(dstThread);
    uint64_t dstNotifyId = dstThreadPtr->GetNotifyIdByIndex(dstNotifyIdx);

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = dstNotifyId;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
{
    if (dst == nullptr) {
        printf("[ERROR] [%s] dst is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (src == nullptr) {
        printf("[ERROR] [%s] src is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    if (SHMManager::GetHcclVmMode() == HcclSim::HcclVmMode::RUNNER) {
        // runner
        auto ret = GetProxyBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetProxyBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // checker
        auto ret = GetMockBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] [Checker] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetMockBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] [Checker] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.transMem.srcRankId = curRank;
    taskMetaData.taskData.transMem.srcOffset = srcOffset;
    taskMetaData.taskData.transMem.dstRankId = curRank;
    taskMetaData.taskData.transMem.dstOffset = dstOffset;
    taskMetaData.taskData.transMem.len       = len;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
{
    if (dst == nullptr) {
        printf("[ERROR] [%s] dst is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (src == nullptr) {
        printf("[ERROR] [%s] src is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    auto channelPtr = reinterpret_cast<SimChannel*>(channel);
    uint32_t locRank = channelPtr->GetLocRankId();
    uint32_t rmtRank = channelPtr->GetRmtRankId();
    uint8_t protocol = static_cast<uint8_t>(channelPtr->GetProtocol());

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    if (SHMManager::GetHcclVmMode() == HcclSim::HcclVmMode::RUNNER) {
        // runner
        auto ret = GetProxyBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetProxyBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // checker
        auto ret = GetMockBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetMockBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.transMem.srcRankId = locRank;
    taskMetaData.taskData.transMem.srcOffset = srcOffset;
    taskMetaData.taskData.transMem.dstRankId = rmtRank;
    taskMetaData.taskData.transMem.dstOffset = dstOffset;
    taskMetaData.taskData.transMem.len       = len;
    taskMetaData.taskData.transMem.protocol  = protocol;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
{
    if (dst == nullptr) {
        printf("[ERROR] [%s] dst is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (src == nullptr) {
        printf("[ERROR] [%s] src is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    auto channelPtr = reinterpret_cast<SimChannel*>(channel);
    uint32_t locRank = channelPtr->GetLocRankId();
    uint32_t rmtRank = channelPtr->GetRmtRankId();
    uint8_t protocol = static_cast<uint8_t>(channelPtr->GetProtocol());

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    if (SHMManager::GetHcclVmMode() == HcclSim::HcclVmMode::RUNNER) {
        // runner
        auto ret = GetProxyBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetProxyBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // checker
        auto ret = GetMockBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetMockBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.transMem.srcRankId = rmtRank;
    taskMetaData.taskData.transMem.srcOffset = srcOffset;
    taskMetaData.taskData.transMem.dstRankId = locRank;
    taskMetaData.taskData.transMem.dstOffset = dstOffset;
    taskMetaData.taskData.transMem.len       = len;
    taskMetaData.taskData.transMem.protocol  = protocol;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    auto channelPtr = reinterpret_cast<SimChannel*>(channel);
    uint64_t remoteNotifyId = channelPtr->GetRmtNotifyIdByIndex(remoteNotifyIdx);
    uint32_t locRankId = channelPtr->GetLocRankId();
    uint32_t rmtRankId = channelPtr->GetRmtRankId();
    uint8_t protocol = static_cast<uint8_t>(channelPtr->GetProtocol());

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = remoteNotifyId;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = locRankId;
    taskMetaData.taskData.notify.dstRankId = rmtRankId;
    taskMetaData.taskData.notify.protocol = protocol;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommChannelNotifyWaitOnThread(
    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout)
{
    // timeout 暂时未使用
    static_cast<void>(timeout);

    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    auto channelPtr = reinterpret_cast<SimChannel*>(channel);
    uint64_t localNotifyId = channelPtr->GetLocNotifyIdByIndex(localNotifyIdx);
    uint32_t locRankId = channelPtr->GetLocRankId();
    uint32_t rmtRankId = channelPtr->GetRmtRankId();
    uint8_t protocol = static_cast<uint8_t>(channelPtr->GetProtocol());

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = localNotifyId;
    taskMetaData.taskData.notify.notifyCount = 1; //notify value
    taskMetaData.taskData.notify.srcRankId = rmtRankId;
    taskMetaData.taskData.notify.dstRankId = locRankId;
    taskMetaData.taskData.notify.protocol = protocol;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (dst == nullptr) {
        printf("[ERROR] [%s] dst is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (src == nullptr) {
        printf("[ERROR] [%s] src is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    if (SHMManager::GetHcclVmMode() == HcclSim::HcclVmMode::RUNNER) {
        // runner
        auto ret = GetProxyBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetProxyBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // checker
        auto ret = GetMockBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetMockBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.reduce.srcRankId = curRank;
    taskMetaData.taskData.reduce.dstRankId = curRank;
    taskMetaData.taskData.reduce.srcOffset = srcOffset;
    taskMetaData.taskData.reduce.dstOffset = dstOffset;
    taskMetaData.taskData.reduce.dataType  = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.dataCount = count;
    taskMetaData.taskData.reduce.reduceOp  = static_cast<uint8_t>(reduceOp);

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (dst == nullptr) {
        printf("[ERROR] [%s] dst is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (src == nullptr) {
        printf("[ERROR] [%s] src is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    auto channelPtr = reinterpret_cast<SimChannel*>(channel);
    uint32_t locRank = channelPtr->GetLocRankId();
    uint32_t rmtRank = channelPtr->GetRmtRankId();
    uint8_t protocol = static_cast<uint8_t>(channelPtr->GetProtocol());

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;

    if (SHMManager::GetHcclVmMode() == HcclSim::HcclVmMode::RUNNER) {
        // runner
        auto ret = GetProxyBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetProxyBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // checker
        auto ret = GetMockBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetMockBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.reduce.srcRankId = locRank;
    taskMetaData.taskData.reduce.dstRankId = rmtRank;
    taskMetaData.taskData.reduce.srcOffset = srcOffset;
    taskMetaData.taskData.reduce.dstOffset = dstOffset;
    taskMetaData.taskData.reduce.dataType  = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.dataCount = count;
    taskMetaData.taskData.reduce.reduceOp  = static_cast<uint8_t>(reduceOp);
    taskMetaData.taskData.reduce.protocol = protocol;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count,
    HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (dst == nullptr) {
        printf("[ERROR] [%s] dst is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (src == nullptr) {
        printf("[ERROR] [%s] src is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    auto channelPtr = reinterpret_cast<SimChannel*>(channel);
    uint32_t locRank = channelPtr->GetLocRankId();
    uint32_t rmtRank = channelPtr->GetRmtRankId();
    uint8_t protocol = static_cast<uint8_t>(channelPtr->GetProtocol());

    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    
    if (SHMManager::GetHcclVmMode() == HcclSim::HcclVmMode::RUNNER) {
        // runner
        auto ret = GetProxyBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetProxyBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // checker
        auto ret = GetMockBufferMemOffset(src, &srcOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] src GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }

        ret = GetMockBufferMemOffset(dst, &dstOffset);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] dst GetProxyBufferMemOffset fail", __func__);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.reduce.srcRankId = rmtRank;
    taskMetaData.taskData.reduce.dstRankId = locRank;
    taskMetaData.taskData.reduce.srcOffset = srcOffset;
    taskMetaData.taskData.reduce.dstOffset = dstOffset;
    taskMetaData.taskData.reduce.dataType  = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.dataCount = count;
    taskMetaData.taskData.reduce.reduceOp  = static_cast<uint8_t>(reduceOp);
    taskMetaData.taskData.reduce.protocol = protocol;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx)
{
    printf("[%s] not support.\n", __func__);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommInterOpNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
{
    printf("[INFO] [%s] start", __func__);
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    ShmNpuResId notify; // 重新组装notify
    notify.value = streamId;
    notify.field.resId = dstNotifyId;

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notify.value;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;
    // taskMetaData.taskData.notify.protocol = protocol; 不需要

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

int32_t HcommInterOpNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    // timeout 暂时未使用
    static_cast<void>(timeOut);
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    if (streamPtr == nullptr) {
        printf("[ERROR] [%s] streamPtr is NULL", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint64_t streamId = (streamPtr->streamId).value;

    ShmNpuResId notify; // 重新组装notify
    notify.value = streamId;
    notify.field.resId = notifyId;

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId   = 1;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notify.value;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;
    // taskMetaData.taskData.notify.protocol = protocol; 不需要

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

#ifndef HCOMM_PRIMITIVES_H_MODIFIED
HcclResult HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
#else
int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
#endif
{
    printf("[%s] raw notify: %lu\n", __func__, dstNotifyId);
    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    uint64_t streamId = (streamPtr->streamId).value;

    ShmNpuResId notify; // 重新组装notify
    notify.value = streamId;
    notify.field.resId = dstNotifyId;

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notify.value;
    taskMetaData.taskData.notify.notifyCount = 0; //notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

#ifndef HCOMM_PRIMITIVES_H_MODIFIED
HcclResult HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
#else
int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
#endif
{
    printf("[%s] raw notify: %lu\n", __func__, notifyId);
    // timeout 暂时未使用
    static_cast<void>(timeOut);

    auto threadPtr = reinterpret_cast<SimHcclThread*>(thread);
    uint32_t curRank = threadPtr->GetCurRank();

    auto streamPtr = threadPtr->GetStream();
    uint64_t streamId = (streamPtr->streamId).value;

    ShmNpuResId notify; // 重新组装notify
    notify.value = streamId;
    notify.field.resId = notifyId;

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    taskMetaData.taskData.notify.notifyId = notify.value;
    taskMetaData.taskData.notify.notifyCount = 1; // notify value
    taskMetaData.taskData.notify.srcRankId = curRank;
    taskMetaData.taskData.notify.dstRankId = curRank;

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[ERROR] [%s] InsertTaskToCollection fail \n", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 下发cid
    HcclTaskCid taskCid{0, curRank, index};
    TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);

    // 记录状态
    TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);

    return HcclResult::HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus