/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl_custom_p2p_aiv.h"

#define ACLCHECK_RET(expr) do { auto _ret = (expr); if (_ret != ACL_SUCCESS) { std::cerr << "acl error at " << __FILE__ << ":" << __LINE__ << ", ret=" << _ret << std::endl; return static_cast<int>(_ret); } } while (0)
#define HCCLCHECK_RET(expr) do { auto _ret = (expr); if (_ret != HCCL_SUCCESS) { std::cerr << "hccl error at " << __FILE__ << ":" << __LINE__ << ", ret=" << _ret << std::endl; return static_cast<int>(_ret); } } while (0)

namespace {

constexpr uint32_t kRankSize = 2;
constexpr uint64_t kElemCount = 1024;

struct ThreadContext {
    HcclRootInfo *rootInfo = nullptr;
    uint32_t rank = 0;
    std::atomic<int> *globalRet = nullptr;
};

int VerifyResult(void *deviceBuf)
{
    std::vector<float> hostRecv(kElemCount, 0.0f);
    ACLCHECK_RET(aclrtMemcpy(hostRecv.data(), hostRecv.size() * sizeof(float), deviceBuf,
        hostRecv.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint64_t i = 0; i < kElemCount; ++i) {
        const float expected = static_cast<float>(i + 1);
        if (hostRecv[i] != expected) {
            std::cerr << "verify failed at index=" << i << ", expected=" << expected
                      << ", actual=" << hostRecv[i] << std::endl;
            return -1;
        }
    }
    return 0;
}

int RunRank(ThreadContext *ctx)
{
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    aclrtStream stream = nullptr;
    HcclComm comm = nullptr;

    const uint32_t rank = ctx->rank;
    const size_t bytes = kElemCount * sizeof(float);

    ACLCHECK_RET(aclrtSetDevice(static_cast<int32_t>(rank)));
    HCCLCHECK_RET(HcclCommInitRootInfo(kRankSize, ctx->rootInfo, rank, &comm));
    ACLCHECK_RET(aclrtCreateStream(&stream));

    if (rank == 0) {
        std::vector<float> hostSend(kElemCount, 0.0f);
        for (uint64_t i = 0; i < kElemCount; ++i) {
            hostSend[i] = static_cast<float>(i + 1);
        }
        ACLCHECK_RET(aclrtMalloc(&sendBuf, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACLCHECK_RET(aclrtMemcpy(sendBuf, bytes, hostSend.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        HCCLCHECK_RET(HcclSendCustomAiv(sendBuf, kElemCount, HCCL_DATA_TYPE_FP32, 1, comm, stream));
        ACLCHECK_RET(aclrtSynchronizeStream(stream));
    } else {
        ACLCHECK_RET(aclrtMalloc(&recvBuf, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACLCHECK_RET(aclrtMemset(recvBuf, bytes, 0, bytes));
        HCCLCHECK_RET(HcclRecvCustomAiv(recvBuf, kElemCount, HCCL_DATA_TYPE_FP32, 0, comm, stream));
        ACLCHECK_RET(aclrtSynchronizeStream(stream));
        if (VerifyResult(recvBuf) != 0) {
            return -1;
        }
        std::cout << "rank " << rank << " verify passed" << std::endl;
    }

    if (comm != nullptr) {
        HCCLCHECK_RET(HcclCommDestroy(comm));
    }
    if (sendBuf != nullptr) {
        ACLCHECK_RET(aclrtFree(sendBuf));
    }
    if (recvBuf != nullptr) {
        ACLCHECK_RET(aclrtFree(recvBuf));
    }
    if (stream != nullptr) {
        ACLCHECK_RET(aclrtDestroyStream(stream));
    }
    ACLCHECK_RET(aclrtResetDevice(static_cast<int32_t>(rank)));
    return 0;
}

} // namespace

int main()
{
    ACLCHECK_RET(aclInit(nullptr));

    uint32_t devCount = 0;
    ACLCHECK_RET(aclrtGetDeviceCount(&devCount));
    if (devCount < kRankSize) {
        std::cerr << "need at least 2 devices, actual=" << devCount << std::endl;
        aclFinalize();
        return -1;
    }

    ACLCHECK_RET(aclrtSetDevice(0));

    void *rootInfoBuf = nullptr;
    ACLCHECK_RET(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    auto *rootInfo = static_cast<HcclRootInfo *>(rootInfoBuf);
    HCCLCHECK_RET(HcclGetRootInfo(rootInfo));

    std::atomic<int> globalRet {0};
    ThreadContext ctx0 {rootInfo, 0, &globalRet};
    ThreadContext ctx1 {rootInfo, 1, &globalRet};

    std::thread sender([&]() {
        int ret = RunRank(&ctx0);
        if (ret != 0) {
            globalRet.store(ret);
        }
    });
    std::thread receiver([&]() {
        int ret = RunRank(&ctx1);
        if (ret != 0) {
            globalRet.store(ret);
        }
    });

    sender.join();
    receiver.join();

    ACLCHECK_RET(aclrtFreeHost(rootInfoBuf));
    ACLCHECK_RET(aclFinalize());

    if (globalRet.load() != 0) {
        std::cerr << "Test Failed" << std::endl;
        return globalRet.load();
    }

    std::cout << "Test Passed" << std::endl;
    return 0;
}
