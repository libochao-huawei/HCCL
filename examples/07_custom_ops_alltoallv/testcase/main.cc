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
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_alltoallv.h"

#define ACLCHECK(expr)                                                                         \
    do {                                                                                       \
        auto _ret = (expr);                                                                    \
        if (_ret != ACL_SUCCESS) {                                                             \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                       \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(expr)                                                                        \
    do {                                                                                       \
        auto _ret = (expr);                                                                    \
        if (_ret != HCCL_SUCCESS) {                                                            \
            printf("[ERROR] hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, _ret); \
            return _ret;                                                                       \
        }                                                                                      \
    } while (0)

inline void BuildLogString(std::ostringstream& oss) {}

template<typename T, typename... Args>
inline void BuildLogString(std::ostringstream& oss, const T& first, const Args&... args) {
    oss << first;
    BuildLogString(oss, args...);
}

template<typename... Args>
void Log(int rank, const Args&... args) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    std::ostringstream oss;

    oss << "[" << tv.tv_sec << "."
        << std::setfill('0') << std::setw(6) << tv.tv_usec
        << "] [Rank " << rank << "] ";

    BuildLogString(oss, args...);

    std::cout << oss.str() << std::endl;
}

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

int PrepareData(int rank, int size, uint64_t count, size_t sendBytes, size_t recvBytes,
                aclrtStream& stream, void*& sendBuf, void*& recvBuf) {
    ACLCHECK(aclrtCreateStream(&stream));
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<float> hostSend(sendBytes / sizeof(float), (float)rank);
    ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf, recvBytes, 0, recvBytes));
    Log(rank, "Buffers allocated and initialized");
    return 0;
}

int VerifyResult(int rank, int size, uint64_t count, size_t recvBytes, void* recvBuf) {
    std::vector<float> hostRecv(recvBytes / sizeof(float));
    ACLCHECK(aclrtMemcpy(hostRecv.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    uint64_t offset = 0;
    for (int r = 0; r < size; r++) {
        for (uint64_t i = 0; i < count; i++) {
            float expected = (float)r;
            float val = hostRecv[offset++];
            if (std::abs(val - expected) > 1e-5) {
                Log(rank, "Error at offset %lu: expected %f, got %f", offset - 1, expected, val);
                return -1;
            }
        }
    }
    return 0;
}

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

    const uint64_t countPerRank = 256;
    const size_t sendBytes = countPerRank * size * sizeof(float);
    const size_t recvBytes = countPerRank * size * sizeof(float);

    std::vector<uint64_t> sendCounts(size, countPerRank);
    std::vector<uint64_t> recvCounts(size, countPerRank);
    std::vector<uint64_t> sdispls(size, 0);
    std::vector<uint64_t> rdispls(size, 0);

    for (int i = 0; i < size; i++) {
        sdispls[i] = i * countPerRank;
        rdispls[i] = i * countPerRank;
    }

    aclrtStream stream = nullptr;
    void *sendBuf = nullptr, *recvBuf = nullptr;

    if (PrepareData(rank, size, countPerRank, sendBytes, recvBytes, stream, sendBuf, recvBuf) == 0) {
        Log(rank, "Starting HcclAllToAllVCustom...");

        auto run_alltoallv = [&]() -> int {
            HCCLCHECK(HcclAllToAllVCustom(sendBuf, recvBuf, sendCounts.data(), recvCounts.data(),
                                          sdispls.data(), rdispls.data(),
                                          HCCL_DATA_TYPE_FP32, hcclComm, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
            return 0;
        };

        if (run_alltoallv() == 0) {
            Log(rank, "HcclAllToAllVCustom completed and synchronized");
            if (VerifyResult(rank, size, countPerRank * size, recvBytes, recvBuf) == 0) {
                Log(rank, "Test Passed!");
            } else {
                Log(rank, "Test Failed!");
            }
        }
    }

    Cleanup(hcclComm, sendBuf, recvBuf, stream);
    return 0;
}