/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_allgather_batch.h"

#define ACLCHECK(expr)                                                                                          \
    do {                                                                                                        \
        auto _ret = (expr);                                                                                     \
        if (_ret != ACL_SUCCESS) {                                                                              \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, _ret);        \
            return _ret;                                                                                        \
        }                                                                                                       \
    } while (0)

#define HCCLCHECK(expr)                                                                                         \
    do {                                                                                                        \
        auto _ret = (expr);                                                                                     \
        if (_ret != HCCL_SUCCESS) {                                                                             \
            printf("[ERROR] hccl interface return err %s:%d, retcode: %d\n", __FILE__, __LINE__, _ret);       \
            return _ret;                                                                                        \
        }                                                                                                       \
    } while (0)

template<typename... Args>
void Log(int rank, const Args&... args)
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    std::ostringstream oss;
    oss << "[" << tv.tv_sec << "." << std::setfill('0') << std::setw(6) << tv.tv_usec << "] [Rank " << rank << "] ";
    (oss << ... << args);
    std::cout << oss.str() << std::endl;
}

int GetLocalRank()
{
    MPI_Comm localComm;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm) == MPI_SUCCESS) {
        int localRank = -1;
        if (MPI_Comm_rank(localComm, &localRank) == MPI_SUCCESS) {
            MPI_Comm_free(&localComm);
            return localRank;
        }
        MPI_Comm_free(&localComm);
    }

    static const char *const envNames[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID"
    };
    for (const char *envName : envNames) {
        const char *env = std::getenv(envName);
        if (env != nullptr && env[0] != '\0') {
            return std::atoi(env);
        }
    }
    return -1;
}

int InitEnv(int argc, char* argv[], int& rank, int& size, HcclComm& hcclComm)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ACLCHECK(aclInit(nullptr));
    uint32_t devCount = 0;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    if (devCount == 0) {
        Log(rank, "No Ascend device found");
        return -1;
    }

    const int localRank = GetLocalRank();
    const int deviceId = (localRank >= 0) ? localRank : rank;
    Log(rank, "devCount=", devCount, ", localRank=", localRank, ", deviceId=", deviceId);
    if (deviceId < 0 || static_cast<uint32_t>(deviceId) >= devCount) {
        Log(rank, "Selected device is out of range");
        return -1;
    }

    ACLCHECK(aclrtSetDevice(deviceId));

    HcclRootInfo rootInfo;
    if (rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
    }
    MPI_Bcast(&rootInfo, sizeof(HcclRootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    HCCLCHECK(HcclCommInitRootInfo(size, &rootInfo, rank, &hcclComm));
    return 0;
}

template<typename T>
int VerifyTyped(int rank, int size, uint64_t count, void *recvBuf)
{
    std::vector<T> hostRecv(count * size);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), hostRecv.size() * sizeof(T), recvBuf,
                         hostRecv.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int r = 0; r < size; r++) {
        for (uint64_t i = 0; i < count; i++) {
            T expected = static_cast<T>(r);
            T actual = hostRecv[r * count + i];
            if (actual != expected) {
                Log(rank, "Verify failed. rank=", r, " idx=", i, " expected=", static_cast<int64_t>(expected),
                    " actual=", static_cast<int64_t>(actual));
                return -1;
            }
        }
    }
    return 0;
}

template<>
int VerifyTyped<float>(int rank, int size, uint64_t count, void *recvBuf)
{
    std::vector<float> hostRecv(count * size);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), hostRecv.size() * sizeof(float), recvBuf,
                         hostRecv.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    for (int r = 0; r < size; r++) {
        for (uint64_t i = 0; i < count; i++) {
            float actual = hostRecv[r * count + i];
            if (std::abs(actual - static_cast<float>(r)) > 1e-5) {
                Log(rank, "Verify failed. rank=", r, " idx=", i, " expected=", static_cast<float>(r), " actual=", actual);
                return -1;
            }
        }
    }
    return 0;
}

void Cleanup(HcclComm hcclComm, std::vector<void*> &buffers, aclrtStream stream)
{
    if (hcclComm) {
        HcclCommDestroy(hcclComm);
    }
    for (void *ptr : buffers) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    if (stream) {
        aclrtDestroyStream(stream);
    }
    aclFinalize();
    MPI_Finalize();
}

int main(int argc, char* argv[])
{
    int rank = 0;
    int size = 0;
    HcclComm hcclComm = nullptr;
    if (InitEnv(argc, argv, rank, size, hcclComm) != 0) {
        MPI_Finalize();
        return -1;
    }

    const uint64_t count0 = 1024;
    const uint64_t count1 = 256;
    const size_t sendBytes0 = count0 * sizeof(float);
    const size_t recvBytes0 = count0 * size * sizeof(float);
    const size_t sendBytes1 = count1 * sizeof(int32_t);
    const size_t recvBytes1 = count1 * size * sizeof(int32_t);

    aclrtStream stream = nullptr;
    ACLCHECK(aclrtCreateStream(&stream));

    void *sendBuf0 = nullptr;
    void *recvBuf0 = nullptr;
    void *sendBuf1 = nullptr;
    void *recvBuf1 = nullptr;
    std::vector<void*> buffers = {sendBuf0, recvBuf0, sendBuf1, recvBuf1};

    ACLCHECK(aclrtMalloc(&sendBuf0, sendBytes0, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf0, recvBytes0, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&sendBuf1, sendBytes1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf1, recvBytes1, ACL_MEM_MALLOC_HUGE_FIRST));
    buffers = {sendBuf0, recvBuf0, sendBuf1, recvBuf1};

    std::vector<float> hostSend0(count0, static_cast<float>(rank));
    std::vector<int32_t> hostSend1(count1, rank);
    ACLCHECK(aclrtMemcpy(sendBuf0, sendBytes0, hostSend0.data(), sendBytes0, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(sendBuf1, sendBytes1, hostSend1.data(), sendBytes1, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf0, recvBytes0, 0, recvBytes0));
    ACLCHECK(aclrtMemset(recvBuf1, recvBytes1, 0, recvBytes1));

    HcclAllGatherItem items[2];
    items[0].sendBuf = sendBuf0;
    items[0].sendCount = count0;
    items[0].dataType = HCCL_DATA_TYPE_FP32;
    items[0].recvBuf = recvBuf0;
    items[1].sendBuf = sendBuf1;
    items[1].sendCount = count1;
    items[1].dataType = HCCL_DATA_TYPE_INT32;
    items[1].recvBuf = recvBuf1;

    Log(rank, "Starting HcclAllGatherBatch...");
    int rc = 0;
    auto run = [&]() -> int {
        HCCLCHECK(HcclAllGatherBatch(items, 2, hcclComm, stream));
        ACLCHECK(aclrtSynchronizeStream(stream));
        return 0;
    };
    rc = run();
    if (rc == 0) {
        rc = VerifyTyped<float>(rank, size, count0, recvBuf0);
    }
    if (rc == 0) {
        rc = VerifyTyped<int32_t>(rank, size, count1, recvBuf1);
    }

    if (rc == 0) {
        Log(rank, "Test Passed!");
    } else {
        Log(rank, "Test Failed!");
    }

    Cleanup(hcclComm, buffers, stream);
    return rc == 0 ? 0 : -1;
}
