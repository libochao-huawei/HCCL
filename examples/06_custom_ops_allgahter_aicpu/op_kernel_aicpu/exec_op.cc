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

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    printf("[DEBUG] OpParam: dataType=%u, count=%lu, rankSize=%u, myRank=%u, inputPtr=%p, outputPtr=%p\n",
           static_cast<uint32_t>(param.dataType), param.count, param.rankSize, param.myRank,
           param.inputPtr, param.outputPtr);
    fflush(stdout);
    
    uint32_t dataTypeSize = SIZE_TABLE[param.dataType];
    uint64_t dataSize = param.count * dataTypeSize;
    uint64_t count = param.count;

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }
    uint32_t cclBuffMultiplier = param.rankSize;
    uint64_t cclBufferSize = resCtx.cclMem.size;
    uint64_t cclBuffBound = cclBufferSize / cclBuffMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    uint64_t maxDataSizePerLoop = std::min(UB_MAX_DATA_SIZE, cclBuffBound);
    uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    HCCL_INFO("[HcclAllGatherCustom] maxDataSizePerLoop: %d, maxDataCountPerLoop: %d, loopCount: %d", maxDataSizePerLoop, maxDataCountPerLoop, loopCount);
    uint64_t loopCount = count / maxDataCountPerLoop + static_cast<uint64_t>(count % maxDataCountPerLoop != 0);
    void *cclBuffAddr = resCtx.cclMem.addr;
    uint64_t processedDataCount = 0;

    for (uint64_t loop = 0; loop < loopCount; loop++) {
        uint64_t sliceCount = std::min(maxDataCountPerLoop, count - loop * maxDataCountPerLoop);
        uint64_t sliceSize = sliceCount * dataTypeSize;
        uint64_t cclBuffOffset = sliceSize * param.myRank;
        uint64_t inputBaseOffset = processedDataCount * dataTypeSize;
        uint64_t inputOffset = processedDataCount * dataTypeSize;
        printf("[DEBUG] loop=%lu, sliceCount=%lu, sliceSize=%lu, cclBuffOffset=%lu, inputOffset=%lu\n", 
               loop, sliceCount, sliceSize, cclBuffOffset, inputOffset);
        fflush(stdout);
        // 本地拷贝到hcclbuf
        void *curCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(cclBuffAddr) + cclBuffOffset);
        void *curInputAddr = static_cast<void *>(static_cast<uint8_t *>(param.inputPtr) + inputOffset);
        printf("[DEBUG] curCclBuffAddr=%p, curInputAddr=%p\n", curCclBuffAddr, curInputAddr);
        fflush(stdout);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], curCclBuffAddr, curInputAddr, sliceSize)));

        // 前同步
        // 主thread通知其他thread开始工作
        for (uint32_t i = 1; i < resCtx.threads.size(); i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[i], 0)));
        }
        // 其他thread等待主thread通知
        for (uint32_t i = 1; i < resCtx.threads.size(); i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[i], 0, CUSTOM_TIMEOUT)));
        }
        // 交换数据
        for (uint32_t i = 0; i < resCtx.threads.size(); i++) {
            uint32_t remoteRank = resCtx.channels[i].remoteRank;
            ChannelHandle remoteChannelHandle = resCtx.channels[i].handle;
            void *remoteCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(resCtx.channels[i].remoteCclMem.addr) + sliceSize * remoteRank);
            printf("[DEBUG] i=%u, thread=%p, remoteRank=%u, channelHandle=%lu, remoteCclBuffAddr=%p, curCclBuffAddr=%p, sliceSize=%lu\n",
                   i, resCtx.threads[i], remoteRank, remoteChannelHandle, remoteCclBuffAddr, curCclBuffAddr, sliceSize);
            fflush(stdout);
            // thread通知对端数据准备完成，可以开始传输
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_ACK)));
            // thread等待对端确认数据准备完成
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            
            // 将本地数写到对端cclBuff
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(resCtx.threads[i], remoteChannelHandle, remoteCclBuffAddr, curCclBuffAddr, sliceSize)));

            // 告诉对端执行完成, 同时等待对端完成
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(resCtx.threads[i], remoteChannelHandle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        // 后同步
        // 主thread等待其他thread
        for (uint32_t i = 1; i < resCtx.threads.size(); i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[i], i, CUSTOM_TIMEOUT)));
        }
        // 其他thread通知主thread
        for (uint32_t i = 1; i < resCtx.threads.size(); i++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[i], resCtx.threads[0], i - 1)));
        }
        // 从cclbuff拷贝到outputPtr
        for (uint32_t rankId = 0; rankId < param.rankSize; rankId++) {
            if (rankId == param.myRank) {
                continue;
            }
            uint64_t cclBuffOffset = sliceSize * rankId;
            uint64_t outputOffset = processedDataCount * dataTypeSize + rankId * dataSize;
            void *curCclBuffAddr = static_cast<void *>(static_cast<uint8_t *>(cclBuffAddr) + cclBuffOffset);
            void *curOutputAddr = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + outputOffset);
            printf("[DEBUG] output copy: rankId=%u, cclBuffOffset=%lu, outputOffset=%lu, curCclBuffAddr=%p, curOutputAddr=%p, sliceSize=%lu\n",
                   rankId, cclBuffOffset, outputOffset, curCclBuffAddr, curOutputAddr, sliceSize);
            fflush(stdout);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], curOutputAddr, curCclBuffAddr, sliceSize)));
        }
        processedDataCount += sliceCount;

    }
    return HCCL_SUCCESS;
}
}
