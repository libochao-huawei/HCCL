/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exec_op.h"
#include <map>
#include <algorithm>
#include "log.h"

namespace ops_hccl_allgather {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t UB_MAX_DATA_SIZE = 256*1024*1024;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;
constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t NOTIFY_IDX_FIN_ACK = 2;

HcclResult ExecOp(OpParam &param, AlgResourceCtx &resCtx)
{
    uint32_t dataTypeSize = SIZE_TABLE[param.dataType];
    uint64_t size = param.count * dataTypeSize;
    uint64_t count = param.count;
    std::map<uint32_t, ChannelInfo> channelMap;

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, size)));
        return HCCL_SUCCESS;
    }

    for (auto channel : resCtx.channels) {
        channelMap[channel.remoteRank] = channel;
    }
    uint32_t cclBuffMultiplier = param.rankSize;
    uint64_t cclBufferSize = resCtx.cclMem.size;
    uint64_t cclBuffBound = cclBufferSize / cclBuffMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    uint64_t maxDataSizePerLoop = std::min(UB_MAX_DATA_SIZE, cclBuffBound);
    uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    uint64_t loopCount = count / maxDataCountPerLoop;
    void *cclBuffAddr = resCtx.cclMem.addr;
    uint64_t processedDataCount = 0;

    for (uint64_t loop = 0; loop < loopCount; loop++) {
        uint64_t sliceCount = std::min(maxDataCountPerLoop, count - loop * maxDataCountPerLoop);
        uint64_t sliceSize = sliceCount * dataTypeSize;
        uint64_t cclBuffOffset = sliceSize * param.myRank;
        uint64_t inputOffset = processedDataCount * dataTypeSize;
        // 本地拷贝到hcclbuf
        void *curCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(cclBuffAddr) + cclBuffOffset);
        void *curInputAddr = static_cast<void *>(static_cast<uint8_t *>(param.inputPtr) + inputOffset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], curCclBuffAddr, curInputAddr, sliceSize)));

        // 前同步
        for (uint32_t i = 1; i < param.rankSize; i++) {
            // 主thread通知其他thread开始工作
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[i], 0)));
            // 其他thread等待主thread通知
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[i], 0, CUSTOM_TIMEOUT)));
        }

        // 交换数据
        for (uint32_t i = 0; i < param.rankSize; i++) {
            // thread通知对端数据准备完成，可以开始传输
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resCtx.threads[i], channelMap[i].channel, NOTIFY_IDX_ACK)));
            // thread等待对端确认数据准备完成
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resCtx.threads[i], channelMap[i].channel, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            uint32_t remoteRank = channelMap[i].remoteRank;
            void *remoteCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(channelMap[i].remoteCclMem.addr) + sliceSize * remoteRank);
            
            // 将本地数写到对端cclBuff
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resCtx.threads[i], channelMap[i].channel, remoteCclBuffAddr, curCclBuffAddr, sliceSize)));

            // 告诉对端执行完成, 同时等待对端完成
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resCtx.threads[i], channelMap[i].channel, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[i], channelMap[i].channel, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        // 后同步, 从cclbuff拷贝到outputPtr
        for (uint32_t rankId = 0; rankId < param.rankSize; rankId++) {
            uint64_t cclBuffOffset = sliceSize * rankId;
            uint64_t outputOffset = processedDataCount * dataTypeSize + rankId * size;
            void *curCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(cclBuffAddr) + cclBuffOffset);
            void *curOutputAddr = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + outputOffset);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], curOutputAddr, curCclBuffAddr, sliceSize)));
        }
        processedDataCount += sliceCount;

    }
    return HCCL_SUCCESS;
}
}
