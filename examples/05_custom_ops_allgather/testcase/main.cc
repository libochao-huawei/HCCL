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
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>

#include "securec.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_allgather.h"

#define ACLCHECK(expr)                                                                         \
    do {                                                                                       \
        auto _ret = (expr); /* 执行一次并保存结果 */                                              \
        if (_ret != ACL_SUCCESS) {                                                             \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                       \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(expr)                                                                        \
    do {                                                                                       \
        auto _ret = (expr); /* 执行一次并保存结果 */                                              \
        if (_ret != HCCL_SUCCESS) {                                                            \
            printf("[ERROR] hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                       \
        }                                                                                      \
    } while (0)

// Helper for logging with timestamp and rank
void Log(int rank, const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);

    int ret = vsnprintf_s(buf, sizeof(buf), sizeof(buf) - 1, fmt, args);
    va_end(args);

    if (ret < 0) {
        printf("[ERROR] [Rank %d] Log formatting failed! ret: %d\n", rank, ret);
        return;
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);
    printf("[%ld.%06ld] [Rank %d] %s\n", tv.tv_sec, tv.tv_usec, rank, buf);
}

// 1. 初始化 MPI、ACL 和 HCCL 通信域
int InitEnv(int argc, char* argv[], int& rank, int& size, HcclComm& hcclComm) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Log(rank, "MPI Initialized. World Size: %d", size);

    ACLCHECK(aclInit(NULL));
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    if (devCount == 0) {
        Log(rank, "Error: No devices found");
        return -1;
    }

    int deviceId = rank % devCount;
    ACLCHECK(aclrtSetDevice(deviceId));
    Log(rank, "Device %d selected (Total devices: %u)", deviceId, devCount);

    HcclRootInfo rootInfo;
    if (rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
        Log(rank, "Root info generated");
    }
    MPI_Bcast(&rootInfo, sizeof(HcclRootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    HCCLCHECK(HcclCommInitRootInfo(size, &rootInfo, rank, &hcclComm));
    Log(rank, "HCCL Comm Initialized");
    return 0;
}

// 2. 准备内存和测试数据
int PrepareData(int rank, uint64_t count, size_t sendBytes, size_t recvBytes, 
                aclrtStream& stream, void*& sendBuf, void*& recvBuf) {
    ACLCHECK(aclrtCreateStream(&stream));
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<float> hostSend(count, (float)rank);
    ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf, recvBytes, 0, recvBytes));
    Log(rank, "Buffers allocated and initialized");
    return 0;
}

// 3. 校验 AllGather 结果
int VerifyResult(int rank, int size, uint64_t count, size_t recvBytes, void* recvBuf) {
    std::vector<float> hostRecv(count * size);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    for (int r = 0; r < size; r++) {
        for (uint64_t i = 0; i < count; i++) {
            float val = hostRecv[r * count + i];
            if (std::abs(val - (float)r) > 1e-5) {
                Log(rank, "Error at rank %d offset %llu: expected %f, got %f", r, i, (float)r, val);
                return -1;
            }
        }
    }
    return 0;
}

// 4. 统一清理资源
void Cleanup(HcclComm hcclComm, void* sendBuf, void* recvBuf, aclrtStream stream) {
    if (hcclComm) HcclCommDestroy(hcclComm);
    if (sendBuf) aclrtFree(sendBuf);
    if (recvBuf) aclrtFree(recvBuf);
    if (stream) aclrtDestroyStream(stream);
    aclFinalize();
    MPI_Finalize();
}

int main(int argc, char* argv[]) {
    int rank = 0, size = 0;
    HcclComm hcclComm = nullptr;
    
    if (InitEnv(argc, argv, rank, size, hcclComm) != 0) {
        MPI_Finalize();
        return -1;
    }

    uint64_t count = 1024;
    size_t sendBytes = count * sizeof(float);
    size_t recvBytes = count * size * sizeof(float);
    
    aclrtStream stream = nullptr;
    void *sendBuf = nullptr, *recvBuf = nullptr;

    if (PrepareData(rank, count, sendBytes, recvBytes, stream, sendBuf, recvBuf) == 0) {
        Log(rank, "Starting HcclAllGatherCustom...");
        
        // 使用 Lambda 包装执行逻辑，避免宏(HCCLCHECK)直接 return 导致绕过 Cleanup
        auto run_allgather = [&]() -> int {
            HCCLCHECK(HcclAllGatherCustom(sendBuf, recvBuf, sendBytes, HCCL_DATA_TYPE_FP32, hcclComm, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
            return 0;
        };

        if (run_allgather() == 0) {
            Log(rank, "HcclAllGatherCustom completed and synchronized");
            if (VerifyResult(rank, size, count, recvBytes, recvBuf) == 0) {
                Log(rank, "Test Passed!");
            } else {
                Log(rank, "Test Failed!");
            }
        }
    }

    // 确保任何情况下都会执行资源释放
    Cleanup(hcclComm, sendBuf, recvBuf, stream);
    return 0;
}