/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_custom_allgather_batch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common.h"
#include "launch_kernel.h"

using namespace ops_hccl_allgather_batch;

namespace {

std::mutex g_ctxMutex;
std::unordered_map<std::string, CcuContext> g_ctxByKey;

std::string BuildContextKey(const char *commName, uint32_t itemCount)
{
    return std::string(commName) + "_batch_" + std::to_string(itemCount);
}

HcclResult CheckItemValid(const HcclAllGatherItem &item, uint32_t index)
{
    CHK_PTR_NULL(item.sendBuf);
    CHK_PTR_NULL(item.recvBuf);
    CHK_PRT_RET(item.sendCount == 0,
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] sendCount is 0", index),
                HCCL_E_PARA);
    CHK_PRT_RET(!IsSupportedDataType(item.dataType),
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] dataType=%d unsupported", index, item.dataType),
                HCCL_E_PARA);

    const uint32_t elemSize = GetDataTypeSize(item.dataType);
    CHK_PRT_RET(item.sendCount > UINT64_MAX / elemSize,
                HCCL_ERROR("[HcclAllGatherBatch] item[%u] sendCount overflow, count=%lu elemSize=%u",
                           index, item.sendCount, elemSize),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

} // namespace

extern "C" HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(items);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(itemCount == 0 || itemCount > MAX_ITEM_COUNT,
                HCCL_ERROR("[HcclAllGatherBatch] itemCount=%u out of range", itemCount),
                HCCL_E_PARA);

    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    const int ret = sprintf_s(param.tag, sizeof(param.tag), "AllGatherBatch_%s_CCU_Custom", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[HcclAllGatherBatch] sprintf_s tag failed"), HCCL_E_INTERNAL);

    CHK_RET(HcclGetRankId(comm, &param.rank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize > MAX_RANK_SIZE,
                HCCL_ERROR("[HcclAllGatherBatch] rankSize=%u exceeds %u", param.rankSize, MAX_RANK_SIZE),
                HCCL_E_NOT_SUPPORT);

    param.itemCount = itemCount;
    for (uint32_t i = 0; i < itemCount; ++i) {
        CHK_RET(CheckItemValid(items[i], i));
        param.items[i] = items[i];
    }

    const std::string ctxKey = BuildContextKey(param.commName, itemCount);
    {
        std::lock_guard<std::mutex> guard(g_ctxMutex);
        CcuContext &ctx = g_ctxByKey[ctxKey];
        CHK_RET(InitCcuContext(comm, param, ctx));
        CHK_RET(LaunchKernel(comm, param, ctx, stream));
    }
    return HCCL_SUCCESS;
}
