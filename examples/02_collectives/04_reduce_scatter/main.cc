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
#include <cmath>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

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
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
    uint64_t strideCount;
    bool testPass;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t sendCount = ctx->devCount;
    uint64_t recvCount = sendCount;
    size_t sendSize = static_cast<size_t>(ctx->devCount) * sendCount * sizeof(float);
    size_t recvSize = recvCount * sizeof(float);

    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // 申请集合通信操作的 Device 内存
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // 申请 Host 内存用于存放输入数据，并按逻辑 2D 布局初始化
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    float *tmpHostBuff = static_cast<float *>(hostBuf);
    // sendBuf[row][col] = row * sendCount + col
    for (uint32_t row = 0; row < ctx->devCount; ++row) {
        for (uint32_t col = 0; col < sendCount; ++col) {
            tmpHostBuff[row * sendCount + col] = static_cast<float>(row * sendCount + col);
        }
    }
    // 将 Host 侧输入数据拷贝到 Device 侧
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 释放 Host 侧内存
    ACLCHECK(aclrtFreeHost(hostBuf));

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 执行 ReduceScatter，将所有 rank 的 sendBuf 相加后，再把结果按照 rank_id 顺序均匀分散到各个 rank 的 recvBuf
    HCCLCHECK(HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx->strideCount, hcclComm, stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(ctx->device));
    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, recvSize));
    ACLCHECK(aclrtMemcpy(resultBuff, recvSize, recvBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float *tmpResBuff = static_cast<float *>(resultBuff);

    // 计算预期值并验证
    bool pass = true;
    for (uint32_t recvIdx = 0; recvIdx < recvCount; ++recvIdx) {
        float expectedVal = 0.0f;
        for (uint32_t r = 0; r < ctx->devCount; ++r) {
            expectedVal += static_cast<float>(r * sendCount + recvIdx);
        }
        float actualVal = tmpResBuff[recvIdx];
        if (fabsf(actualVal - expectedVal) > 1e-6f) {
            printf("[FAIL] rankId=%u, recvIdx=%u, actual=%.1f, expected=%.1f\n",
                   ctx->device, recvIdx, actualVal, expectedVal);
            pass = false;
        } else {
            printf("[PASS] rankId=%u, recvIdx=%u, value=%.1f\n",
                   ctx->device, recvIdx, actualVal);
        }
    }
    ctx->testPass = pass;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    ACLCHECK(aclrtFree(sendBuf));          // 释放 Device 侧内存
    ACLCHECK(aclrtFree(recvBuf));          // 释放 Device 侧内存
    ACLCHECK(aclrtDestroyStream(stream));  // 销毁任务流
    return 0;
}

int main()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成 Root 节点信息，各线程使用同一份 RootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 测试用例配置
    const uint64_t sendCountBase = devCount;
    struct TestCaseConfig { uint64_t strideCount; const char* desc; };
    TestCaseConfig testCases[] = {
        { 0,                       "stride=0 (continuous)" },
        { sendCountBase,           "stride=sendCount" },
        { sendCountBase * 2,       "stride=2*sendCount" },
    };

    for (const auto& tc : testCases) {
        std::cout << "\n=== Test: strideCount=" << tc.strideCount
                  << " (" << tc.desc << ") ===" << std::endl;

        std::vector<std::thread> threads(devCount);
        std::vector<ThreadContext> args(devCount);
        for (uint32_t i = 0; i < devCount; i++) {
            args[i].rootInfo = rootInfo;
            args[i].device = i;
            args[i].devCount = devCount;
            args[i].strideCount = tc.strideCount;
            args[i].testPass = true;
            threads[i] = std::thread(Sample, (void *)&args[i]);
        }

        bool allPass = true;
        for (uint32_t i = 0; i < devCount; i++) {
            threads[i].join();
            if (!args[i].testPass) {
                allPass = false;
            }
        }
        if (allPass) {
            std::cout << "=== All tests PASSED ===" << std::endl;
        } else {
            std::cout << "=== Some tests FAILED ===" << std::endl;
        }
    }

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // 释放 Host 内存
    ACLCHECK(aclFinalize());               // 设备去初始化
    return 0;
}
