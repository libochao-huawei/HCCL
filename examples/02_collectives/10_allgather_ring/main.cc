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
#include <vector>
#include <thread>
#include <chrono>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define ACLCHECK(ret)                                                                           \
    do {                                                                                        \
        if (ret != ACL_SUCCESS) {                                                               \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

#define HCCLCHECK(ret)                                                                           \
    do {                                                                                         \
        if (ret != HCCL_SUCCESS) {                                                               \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                          \
        }                                                                                        \
    } while (0)

extern "C" HcclResult HcclAllGatherRing(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream);

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t sendCount = 1U;
    uint64_t recvCount = ctx->devCount;
    size_t sendSize = sendCount * sizeof(float);
    size_t recvSize = recvCount * sizeof(float);

    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));

    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    float *tmpHostBuf = static_cast<float *>(hostBuf);
    for (uint64_t i = 0; i < sendCount; ++i) {
        tmpHostBuf[i] = static_cast<float>(device);
    }
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    HCCLCHECK(HcclAllGatherRing(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    ACLCHECK(aclrtSynchronizeStream(stream));

    std::this_thread::sleep_for(std::chrono::seconds(device));
    void *resultBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&resultBuf, recvSize));
    ACLCHECK(aclrtMemcpy(resultBuf, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float *tmpResultBuf = static_cast<float *>(resultBuf);
    std::cout << "rankId: " << device << ", output: [";
    for (uint32_t i = 0; i < recvCount; ++i) {
        std::cout << " " << tmpResultBuf[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuf));

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    return 0;
}

int main()
{
    ACLCHECK(aclInit(nullptr));
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());
    return 0;
}
