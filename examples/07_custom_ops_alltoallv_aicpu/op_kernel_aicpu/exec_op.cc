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
#include "log.h"
#include <algorithm>

namespace ops_hccl_alltoallv_aicpu {

HcclResult ExecAlltoAllV(OpParam &param, AlgResourceCtx* resCtx)
{
    uint32_t rankSize = param.rankSize;
    uint32_t rank = param.rank;
    uint64_t dataTypeSize = SIZE_TABLE[param.dataType];
    
    ThreadHandle mainThread = resCtx->mainThread;
    
    uint64_t cclBufferSize = resCtx->cclBuffer.size;
    uint64_t cclBufferCountPerRank = cclBufferSize / dataTypeSize / rankSize;
    
    uint64_t maxSendOrRecvDataCount = 0;
    for (uint32_t i = 0; i < rankSize; i++) {
        maxSendOrRecvDataCount = std::max(maxSendOrRecvDataCount, param.all2AllVDataDes.sendCounts[i]);
        maxSendOrRecvDataCount = std::max(maxSendOrRecvDataCount, param.all2AllVDataDes.recvCounts[i]);
    }
    
    uint64_t maxDataCountPerLoop = cclBufferCountPerRank;
    uint64_t loopTimes = maxSendOrRecvDataCount / maxDataCountPerLoop +
        static_cast<uint64_t>(maxSendOrRecvDataCount % maxDataCountPerLoop != 0);
    
    uint64_t processedDataCount = 0;
    
    for (uint64_t loop = 0; loop < loopTimes; loop++) {
        uint64_t currDataCount = (loop == loopTimes - 1) ? 
            maxSendOrRecvDataCount - processedDataCount : maxDataCountPerLoop;
        
        HCCL_INFO("[ExecAlltoAllV] loop %llu, currDataCount %llu", loop, currDataCount);
        
        for (uint32_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
            if (remoteRank == rank) {
                uint64_t sendCount = param.all2AllVDataDes.sendCounts[rank];
                uint64_t sendDispl = param.all2AllVDataDes.sdispls[rank];
                uint64_t recvCount = param.all2AllVDataDes.recvCounts[rank];
                uint64_t recvDispl = param.all2AllVDataDes.rdispls[rank];
                
                uint64_t curSendCount = 0, curRecvCount = 0;
                uint64_t curSendDispl = sendDispl, curRecvDispl = recvDispl;
                
                if (sendCount > processedDataCount) {
                    curSendCount = std::min(currDataCount, sendCount - processedDataCount);
                    curSendDispl = sendDispl + processedDataCount;
                }
                if (recvCount > processedDataCount) {
                    curRecvCount = std::min(currDataCount, recvCount - processedDataCount);
                    curRecvDispl = recvDispl + processedDataCount;
                }
                
                if (curSendCount > 0 && curRecvCount > 0) {
                    uint64_t size = curSendCount * dataTypeSize;
                    void* srcAddr = static_cast<char*>(param.inputPtr) + curSendDispl * dataTypeSize;
                    void* dstAddr = static_cast<char*>(param.outputPtr) + curRecvDispl * dataTypeSize;
                    CHK_RET(HcommLocalCopyOnThread(mainThread, dstAddr, srcAddr, size));
                }
                continue;
            }
            
            uint64_t sendCount = param.all2AllVDataDes.sendCounts[remoteRank];
            uint64_t sendDispl = param.all2AllVDataDes.sdispls[remoteRank];
            uint64_t recvCount = param.all2AllVDataDes.recvCounts[remoteRank];
            uint64_t recvDispl = param.all2AllVDataDes.rdispls[remoteRank];
            
            uint64_t curSendCount = 0, curRecvCount = 0;
            uint64_t curSendDispl = sendDispl, curRecvDispl = recvDispl;
            
            if (sendCount > processedDataCount) {
                curSendCount = std::min(currDataCount, sendCount - processedDataCount);
                curSendDispl = sendDispl + processedDataCount;
            }
            if (recvCount > processedDataCount) {
                curRecvCount = std::min(currDataCount, recvCount - processedDataCount);
                curRecvDispl = recvDispl + processedDataCount;
            }
            
            uint32_t channelIdx = (remoteRank < rank) ? remoteRank : remoteRank - 1;
            if (channelIdx >= resCtx->channelHandles.size()) {
                HCCL_ERROR("[ExecAlltoAllV] Invalid channel index %u", channelIdx);
                return HCCL_E_INTERNAL;
            }
            
            ChannelHandle channel = resCtx->channelHandles[channelIdx];
            
            if (curSendCount > 0) {
                uint64_t sendSize = curSendCount * dataTypeSize;
                void* srcAddr = static_cast<char*>(param.inputPtr) + curSendDispl * dataTypeSize;
                void* dstAddr = static_cast<char*>(resCtx->cclBuffer.addr) + 
                    rank * cclBufferCountPerRank * dataTypeSize;
                
                CHK_RET(HcommWriteOnThread(mainThread, channel, dstAddr, srcAddr, sendSize));
            }
            
            if (curRecvCount > 0) {
                uint64_t recvSize = curRecvCount * dataTypeSize;
                void* srcAddr = static_cast<char*>(resCtx->cclBuffer.addr) + 
                    remoteRank * cclBufferCountPerRank * dataTypeSize;
                void* dstAddr = static_cast<char*>(param.outputPtr) + curRecvDispl * dataTypeSize;
                
                CHK_RET(HcommLocalCopyOnThread(mainThread, dstAddr, srcAddr, recvSize));
            }
        }
        
        processedDataCount += currDataCount;
    }
    
    HCCL_INFO("[ExecAlltoAllV] AlltoAllV execution completed");
    return HCCL_SUCCESS;
}

}