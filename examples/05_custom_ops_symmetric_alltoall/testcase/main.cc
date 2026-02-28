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
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "hccl_custom_symmetric_alltoall.h"

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        aclError _acl_ret = ret;                                                               \
        if (_acl_ret != ACL_SUCCESS) {                                                        \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _acl_ret); \
            return _acl_ret;                                                                   \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        HcclResult _hccl_ret = ret;                                                           \
        if (_hccl_ret != HCCL_SUCCESS) {                                                       \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _hccl_ret); \
            return _hccl_ret;                                                                  \
        }                                                                                      \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

struct SymmetricMemHandle {
    void* vaPtr;
    aclrtDrvMemHandle phyHandle;
    CommSymWindow symWin;
};

void* AllocSymmetricMem(HcclComm comm, uint64_t size, CommSymWindow* winHandle, int32_t deviceId)
{
    void* vaPtr = nullptr;
    aclrtDrvMemHandle phyHandle = nullptr;

    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed, ret=%d\n", ret);
        return nullptr;
    }

    size_t granularity = 0;
    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = deviceId;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
    if (ret != ACL_SUCCESS || granularity == 0) {
        printf("GetAllocationGranularity failed, ret=%d\n", ret);
        return nullptr;
    }
    size_t allocSize = (size + granularity - 1) / granularity * granularity;

    ret = aclrtReserveMemAddress(&vaPtr, allocSize, 0, nullptr, 1);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtReserveMemAddress failed, ret=%d\n", deviceId, ret);
        return nullptr;
    }

    ret = aclrtMallocPhysical(&phyHandle, allocSize, &prop, 0);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtMallocPhysical failed, ret=%d\n", deviceId, ret);
        aclrtReleaseMemAddress(vaPtr);
        return nullptr;
    }

    ret = aclrtMapMem(vaPtr, allocSize, 0, phyHandle, 0);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtMapMem failed, ret=%d\n", deviceId, ret);
        aclrtFreePhysical(phyHandle);
        aclrtReleaseMemAddress(vaPtr);
        return nullptr;
    }

    HcclResult hcclRet = HcclCommSymWinRegister(comm, vaPtr, allocSize, winHandle, HCCL_WIN_COLL_SYMMETRIC);
    if (hcclRet != HCCL_SUCCESS) {
        printf("Device %d: HcclCommSymWinRegister failed, ret=%d, vaPtr=%p, size=%lu\n", 
               deviceId, hcclRet, vaPtr, static_cast<unsigned long>(allocSize));
        aclrtUnmapMem(vaPtr);
        aclrtFreePhysical(phyHandle);
        aclrtReleaseMemAddress(vaPtr);
        return nullptr;
    }

    printf("[Device %d] AllocSymmetricMem: vaPtr=%p, winHandle=%p, size=%lu\n",
           deviceId, vaPtr, *winHandle, static_cast<unsigned long>(allocSize));

    return vaPtr;
}

int FreeSymmetricMem(void* vaPtr, CommSymWindow winHandle)
{
    if (vaPtr == nullptr || winHandle == nullptr) {
        return 0;
    }

    HcclResult hcclRet = HcclCommSymWinDeregister(winHandle);
    if (hcclRet != HCCL_SUCCESS) {
        printf("HcclCommSymWinDeregister failed, ret=%d\n", hcclRet);
        return -1;
    }

    aclrtDrvMemHandle phyHandle;
    aclError ret = aclrtMemRetainAllocationHandle(vaPtr, &phyHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMemRetainAllocationHandle failed, ret=%d\n", ret);
        return -1;
    }
    ret = aclrtUnmapMem(vaPtr);
    if (ret != ACL_SUCCESS) {
        printf("aclrtUnmapMem failed, ret=%d\n", ret);
        return -1;
    }
    ret = aclrtFreePhysical(phyHandle);
    if (ret != ACL_SUCCESS) {
        printf("aclrtFreePhysical failed, ret=%d\n", ret);
        return -1;
    }
    ret = aclrtReleaseMemAddress(vaPtr);
    if (ret != ACL_SUCCESS) {
        printf("aclrtReleaseMemAddress failed, ret=%d\n", ret);
        return -1;
    }

    printf("FreeSymmetricMem: vaPtr=%p released\n", vaPtr);
    return 0;
}

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    uint32_t device = ctx->device;
    uint32_t devCount = ctx->devCount;

    uint64_t count = devCount * 8;
    uint64_t dataSize = count * sizeof(float);

    aclError ret = aclrtSetDevice(static_cast<int32_t>(device));
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtSetDevice failed, ret=%d\n", device, ret);
        return -1;
    }

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = device;

    HcclComm hcclComm;
    HcclResult hcclRet = HcclCommInitRootInfoConfig(ctx->devCount, ctx->rootInfo, device, &config, &hcclComm);
    if (hcclRet != HCCL_SUCCESS) {
        printf("Device %d: HcclCommInitRootInfoConfig failed, ret=%d\n", device, hcclRet);
        return -1;
    }
    printf("Device %d: HcclCommInitRootInfoConfig success, hcclComm=%p\n", device, hcclComm);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    ret = aclrtMalloc(&sendBuf, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtMalloc sendBuf failed, ret=%d\n", device, ret);
        return -1;
    }
    ret = aclrtMalloc(&recvBuf, dataSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtMalloc recvBuf failed, ret=%d\n", device, ret);
        aclrtFree(sendBuf);
        return -1;
    }

    float* sendData = static_cast<float*>(sendBuf);
    for (uint64_t i = 0; i < count; i++) {
        sendData[i] = static_cast<float>(device);
    }

    printf("Device %d: Initialized sendBuf with rank %u\n", device, device);

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtCreateStream failed, ret=%d\n", device, ret);
        return -1;
    }

    hcclRet = HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
    if (hcclRet != HCCL_SUCCESS) {
        printf("Device %d: HcclAllReduce failed, ret=%d\n", device, hcclRet);
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtSynchronizeStream failed, ret=%d\n", device, ret);
        return -1;
    }

    float* recvData = static_cast<float*>(recvBuf);
    printf("Device %d: recvBuf[0]=%f, recvBuf[1]=%f, recvBuf[2]=%f, recvBuf[3]=%f\n",
           device, recvData[0], recvData[1], recvData[2], recvData[3]);

    float expected = 0.0f;
    for (uint32_t i = 0; i < devCount; i++) {
        expected += static_cast<float>(i);
    }

    bool pass = true;
    for (uint64_t i = 0; i < count; i++) {
        if (recvData[i] != expected) {
            printf("Device %d: Verification failed at index %lu, expected %f, got %f\n",
                   device, static_cast<unsigned long>(i), expected, recvData[i]);
            pass = false;
            break;
        }
    }
    if (pass) {
        printf("Device %d: Verification PASSED!\n", device);
    }

    aclrtFree(recvBuf);
    aclrtFree(sendBuf);

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ret = aclrtDestroyStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtDestroyStream failed, ret=%d\n", device, ret);
        return -1;
    }
    ret = aclrtResetDevice(device);
    if (ret != ACL_SUCCESS) {
        printf("Device %d: aclrtResetDevice failed, ret=%d\n", device, ret);
        return -1;
    }

    return 0;
}

int main()
{
    ACLCHECK(aclInit(NULL));

    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    if (devCount < 2) {
        std::cout << "This example requires at least 2 devices" << std::endl;
        return -1;
    }

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

    std::cout << "All devices completed successfully!" << std::endl;
    return 0;
}
