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
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    printf("[%ld.%06ld] [Rank %d] %s\n", tv.tv_sec, tv.tv_usec, rank, buf);
}

int main(int argc, char* argv[])
{
    // MPI Init
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Log(rank, "MPI Initialized. World Size: %d", size);

    // ACL Init
    ACLCHECK(aclInit(NULL));
    
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    
    if (devCount == 0) {
        Log(rank, "Error: No devices found");
        MPI_Finalize();
        return -1;
    }
    
    int deviceId = rank % devCount;
    ACLCHECK(aclrtSetDevice(deviceId));
    Log(rank, "Device %d selected (Total devices: %u)", deviceId, devCount);

    // HCCL Root Info Exchange
    HcclRootInfo rootInfo;
    if (rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
        Log(rank, "Root info generated");
    }
    MPI_Bcast(&rootInfo, sizeof(HcclRootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // HCCL Init
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(size, &rootInfo, rank, &hcclComm));
    Log(rank, "HCCL Comm Initialized");

    // Prepare Data
    uint64_t count = 1024;
    size_t sendBytes = count * sizeof(float);
    size_t recvBytes = count * size * sizeof(float);

    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<float> hostSend(count, (float)rank);
    ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf, recvBytes, 0, recvBytes));
    Log(rank, "Buffers allocated and initialized");

    // Run Custom AllGather
    Log(rank, "Starting HcclAllGatherCustom...");
    HCCLCHECK(HcclAllGatherCustom(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, hcclComm, stream));
    
    ACLCHECK(aclrtSynchronizeStream(stream));
    Log(rank, "HcclAllGatherCustom completed and synchronized");

    // Verify
    std::vector<float> hostRecv(count * size);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    for (int r = 0; r < size; r++) {
        for (uint64_t i = 0; i < count; i++) {
            float val = hostRecv[r * count + i];
            if (std::abs(val - (float)r) > 1e-5) {
                Log(rank, "Error at rank %d offset %llu: expected %f, got %f", r, i, (float)r, val);
                pass = false;
                break;
            }
        }
        if (!pass) break;
    }

    if (pass) {
        Log(rank, "Test Passed!");
    } else {
        Log(rank, "Test Failed!");
    }

    // Cleanup
    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtFree(sendBuf));
    ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclFinalize());
    
    MPI_Finalize();
    return 0;
}
