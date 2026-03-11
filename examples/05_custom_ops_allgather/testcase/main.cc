/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cmath>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_allgather.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo rootInfo;
    uint32_t deviceId;
    uint32_t rankId;
    uint32_t rankSize;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t deviceId = ctx->deviceId;
    uint32_t rankId = ctx->rankId;
    uint32_t rankSize = ctx->rankSize;
    
    uint64_t count = 1024; // Elements per rank
    size_t sendBytes = count * sizeof(float);
    size_t recvBytes = count * rankSize * sizeof(float);

    // Set device
    ACLCHECK(aclrtSetDevice(deviceId));

    // Init Comm
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(rankSize, &ctx->rootInfo, rankId, &hcclComm));

    // Create Stream
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // Alloc Buffers
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // Init Send Data
    std::vector<float> hostSend(count, (float)rankId);
    ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf, recvBytes, 0, recvBytes));

    // Run AllGather
    HCCLCHECK(HcclAllGatherCustom(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, hcclComm, stream));

    // Sync
    ACLCHECK(aclrtSynchronizeStream(stream));

    // Verify
    std::vector<float> hostRecv(count * rankSize);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    for (uint32_t r = 0; r < rankSize; r++) {
        for (uint64_t i = 0; i < count; i++) {
            float val = hostRecv[r * count + i];
            if (std::abs(val - (float)r) > 1e-5) {
                printf("[Rank %u] Error at rank %u offset %llu: expected %f, got %f\n", rankId, r, i, (float)r, val);
                pass = false;
                break;
            }
        }
        if (!pass) break;
    }

    if (pass) {
        printf("[Rank %u] AllGather Success!\n", rankId);
    } else {
        printf("[Rank %u] AllGather Failed!\n", rankId);
    }

    // Cleanup
    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    // ACLCHECK(aclrtResetDevice(deviceId)); // Avoid resetting device in thread if reused
    return 0;
}

int main()
{
    // Init ACL
    ACLCHECK(aclInit(NULL));
    
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    printf("Found %u devices\n", devCount);
    
    if (devCount < 2) {
        printf("Need at least 2 devices for test\n");
        return 0;
    }
    
    // Use first 2 devices
    uint32_t rankSize = 2;
    
    ACLCHECK(aclrtSetDevice(0));
    HcclRootInfo rootInfo;
    HCCLCHECK(HcclGetRootInfo(&rootInfo));
    
    std::vector<std::thread> threads;
    std::vector<ThreadContext> ctxs(rankSize);
    
    for (uint32_t i = 0; i < rankSize; i++) {
        ctxs[i].rootInfo = rootInfo;
        ctxs[i].deviceId = i;
        ctxs[i].rankId = i;
        ctxs[i].rankSize = rankSize;
        threads.emplace_back(Sample, &ctxs[i]);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    ACLCHECK(aclFinalize());
    return 0;
}
