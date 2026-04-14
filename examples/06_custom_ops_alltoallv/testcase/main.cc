/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN AS IS BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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

int PrepareData(int rank, int size, uint64_t totalCount, size_t sendBytes, size_t recvBytes, 
                aclrtStream& stream, void*& sendBuf, void*& recvBuf,
                void*& sendCounts, void*& sdispls, void*& recvCounts, void*& rdispls) {
    ACLCHECK(aclrtCreateStream(&stream));
    ACLCHECK(aclrtMalloc(&sendBuf, sendBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    
    ACLCHECK(aclrtMalloc(&sendCounts, size * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&sdispls, size * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&recvCounts, size * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&rdispls, size * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<uint64_t> hSendCounts(size);
    std::vector<uint64_t> hSdispls(size);
    std::vector<uint64_t> hRecvCounts(size);
    std::vector<uint64_t> hRdispls(size);
    
    for (int i = 0; i < size; i++) {
        hSendCounts[i] = totalCount / size;
        hSdispls[i] = i * (totalCount / size);
        hRecvCounts[i] = totalCount / size;
        hRdispls[i] = i * (totalCount / size);
    }
    
    ACLCHECK(aclrtMemcpy(sendCounts, size * sizeof(uint64_t), hSendCounts.data(), 
                         size * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(sdispls, size * sizeof(uint64_t), hSdispls.data(), 
                         size * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(recvCounts, size * sizeof(uint64_t), hRecvCounts.data(), 
                         size * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(rdispls, size * sizeof(uint64_t), hRdispls.data(), 
                         size * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<float> hostSend(totalCount, (float)rank);
    ACLCHECK(aclrtMemcpy(sendBuf, sendBytes, hostSend.data(), sendBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemset(recvBuf, recvBytes, 0, recvBytes));
    Log(rank, "Buffers allocated and initialized");
    return 0;
}

int VerifyResult(int rank, int size, uint64_t totalCount, size_t recvBytes, void* recvBuf) {
    std::vector<float> hostRecv(totalCount);
    ACLCHECK(aclrtMemcpy(hostRecv.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    uint64_t perRankCount = totalCount / size;
    for (int r = 0; r < size; r++) {
        for (uint64_t i = 0; i < perRankCount; i++) {
            float val = hostRecv[r * perRankCount + i];
            if (std::abs(val - (float)r) > 1e-5) {
                Log(rank, "Error at rank %d offset %llu: expected %f, got %f", r, i, (float)r, val);
                return -1;
            }
        }
    }
    return 0;
}

void Cleanup(HcclComm hcclComm, void* sendBuf, void* recvBuf, void* sendCounts, void* sdispls, 
             void* recvCounts, void* rdispls, aclrtStream stream) {
    if (hcclComm) HcclCommDestroy(hcclComm);
    if (sendBuf) aclrtFree(sendBuf);
    if (recvBuf) aclrtFree(recvBuf);
    if (sendCounts) aclrtFree(sendCounts);
    if (sdispls) aclrtFree(sdispls);
    if (recvCounts) aclrtFree(recvCounts);
    if (rdispls) aclrtFree(rdispls);
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

    uint64_t totalCount = 1024;
    size_t sendBytes = totalCount * sizeof(float);
    size_t recvBytes = totalCount * sizeof(float);
    
    aclrtStream stream = nullptr;
    void *sendBuf = nullptr, *recvBuf = nullptr;
    void *sendCounts = nullptr, *sdispls = nullptr, *recvCounts = nullptr, *rdispls = nullptr;

    if (PrepareData(rank, size, totalCount, sendBytes, recvBytes, stream, 
                    sendBuf, recvBuf, sendCounts, sdispls, recvCounts, rdispls) == 0) {
        Log(rank, "Starting HcclAlltoAllVCustom...");
        
        auto run_alltoallv = [&]() -> int {
            HCCLCHECK(HcclAlltoAllVCustom(sendBuf, sendCounts, sdispls, recvBuf, 
                                          recvCounts, rdispls, HCCL_DATA_TYPE_FP32, 
                                          hcclComm, stream));
            ACLCHECK(aclrtSynchronizeStream(stream));
            return 0;
        };

        if (run_alltoallv() == 0) {
            Log(rank, "HcclAlltoAllVCustom completed and synchronized");
            if (VerifyResult(rank, size, totalCount, recvBytes, recvBuf) == 0) {
                Log(rank, "Test Passed!");
            } else {
                Log(rank, "Test Failed!");
            }
        }
    }

    Cleanup(hcclComm, sendBuf, recvBuf, sendCounts, sdispls, recvCounts, rdispls, stream);
    return 0;
}