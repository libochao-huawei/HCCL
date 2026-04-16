/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "barrier_dpu_op.h"
#include "op_common_ops.h"
#include "sal.h"
#include "env_config.h"
#include "adapter_acl.h"
#include <acl/acl_rt.h>
#include <algorithm>
#include <cstring>

using namespace std;
using namespace ops_hccl;

namespace {

constexpr uint32_t kBarrierTokenSize = 1U;
constexpr HcclDataType kBarrierDataType = HCCL_DATA_TYPE_UINT32;

HcclResult GetBarrierMemory(u32 rankSize, void** sendBuf, void** recvBuf)
{
    static thread_local void* g_sendBuf = nullptr;
    static thread_local void* g_recvBuf = nullptr;
    static thread_local u32 g_rankSize = 0;
    static thread_local bool g_isFirstBarrier = true;

    if (g_isFirstBarrier || g_rankSize != rankSize) {
        if (g_sendBuf != nullptr) {
            (void)aclrtFree(g_sendBuf);
            g_sendBuf = nullptr;
        }
        if (g_recvBuf != nullptr) {
            (void)aclrtFree(g_recvBuf);
            g_recvBuf = nullptr;
        }

        u32 tokenSize = sizeof(u32);
        u32 sendSize = tokenSize;
        u32 recvSize = tokenSize * rankSize;

        aclError aclRet = aclrtMalloc(&g_sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[GetBarrierMemory] Failed to allocate sendBuf, ret=%d", aclRet);
            return HCCL_E_RUNTIME;
        }

        aclRet = aclrtMalloc(&g_recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[GetBarrierMemory] Failed to allocate recvBuf, ret=%d", aclRet);
            (void)aclrtFree(g_sendBuf);
            g_sendBuf = nullptr;
            return HCCL_E_RUNTIME;
        }

        (void)memset(g_sendBuf, 0, sendSize);
        (void)memset(g_recvBuf, 0, recvSize);

        g_rankSize = rankSize;
        g_isFirstBarrier = false;
        HCCL_INFO("[GetBarrierMemory] Allocated barrier memory, rankSize=%u, sendBuf=%p, recvBuf=%p",
                  rankSize, g_sendBuf, g_recvBuf);
    }

    *sendBuf = g_sendBuf;
    *recvBuf = g_recvBuf;
    return HCCL_SUCCESS;
}

HcclResult BarrierSingleStep(HcclComm comm, aclrtStream stream)
{
    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    u32 rankId = 0;
    CHK_RET(HcclGetRankId(comm, &rankId));
    HCCL_DEBUG("[BarrierSingleStep] rankId[%u], rankSize[%u]", rankId, rankSize);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    CHK_RET(GetBarrierMemory(rankSize, &sendBuf, &recvBuf));

    u32 token = rankId;
    (void)memcpy(sendBuf, &token, sizeof(u32));

    HcclResult ret = HcclAllGather(sendBuf, recvBuf, kBarrierTokenSize,
                                   kBarrierDataType, comm, stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[BarrierSingleStep] HcclAllGather failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret));
        return ret;
    }

    ret = aclrtStreamSynchronize(stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[BarrierSingleStep] Stream synchronize failed, ret[0x%016llx]", HCCL_ERROR_CODE(ret));
        return ret;
    }

    return HCCL_SUCCESS;
}

}  // anonymous namespace

extern "C" {

HcclResult HcclBarrierDPU(HcclComm intraComm, HcclComm interComm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HCCL_INFO("[HcclBarrierDPU] Start, hierarchical barrier");

    CHK_RET(CheckBarrierDPUInputPara(intraComm, stream));
    CHK_RET(CheckBarrierDPUInputPara(interComm, stream));

    u32 intraRankSize = 0;
    CHK_RET(HcclGetRankSize(intraComm, &intraRankSize));
    HCCL_INFO("[HcclBarrierDPU] Intra-pod barrier, rankSize[%u]", intraRankSize);

    if (intraRankSize > 1) {
        HcclResult ret = BarrierSingleStep(intraComm, stream);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclBarrierDPU] Intra-pod barrier failed, ret[0x%016llx]",
                       HCCL_ERROR_CODE(ret));
            return ret;
        }
    }

    u32 interRankSize = 0;
    CHK_RET(HcclGetRankSize(interComm, &interRankSize));
    HCCL_INFO("[HcclBarrierDPU] Inter-pod barrier, rankSize[%u]", interRankSize);

    if (interRankSize > 1) {
        HcclResult ret = BarrierSingleStep(interComm, stream);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclBarrierDPU] Inter-pod barrier failed, ret[0x%016llx]",
                       HCCL_ERROR_CODE(ret));
            return ret;
        }
    }

    HCCL_INFO("[HcclBarrierDPU] Success, take time[%lld]us",
              DURATION_US(TIME_NOW() - startut).count());
    return HCCL_SUCCESS;
}

}  // extern "C"

namespace ops_hccl {

HcclResult CheckBarrierDPUInputPara(const HcclComm comm, const aclrtStream stream)
{
    RPT_INPUT_ERR(stream == nullptr, "EI0003",
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclBarrierDPU", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    RPT_INPUT_ERR(comm == nullptr, "EI0003",
        std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclBarrierDPU", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl