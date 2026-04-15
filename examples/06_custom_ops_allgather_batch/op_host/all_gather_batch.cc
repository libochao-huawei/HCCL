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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common.h"
#include "host_utils.h"
#include "launch_kernel.h"

using namespace ops_hccl_allgather_batch;

namespace {

std::mutex g_ctxMutex;
std::unordered_map<std::string, std::shared_ptr<std::mutex>> g_ctxInitMutexByKey;

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
    const int ret = sprintf_s(param.tag, sizeof(param.tag), "AllGatherBatch_%s_Custom", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[HcclAllGatherBatch] sprintf_s tag failed"), HCCL_E_INTERNAL);

    CHK_RET(HcclGetRankId(comm, &param.rank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize > MAX_RANK_SIZE,
                HCCL_ERROR("[HcclAllGatherBatch] rankSize=%u exceeds %u", param.rankSize, MAX_RANK_SIZE),
                HCCL_E_NOT_SUPPORT);

    param.itemCount = itemCount;
    for (uint32_t i = 0; i < itemCount; ++i) {
        CHK_RET(CheckItemValid(items[i], i, param.rank, param.rankSize));
        param.items[i] = items[i];
    }

    const std::string ctxKey = BuildContextKey(
        param.tag, reinterpret_cast<uintptr_t>(comm), itemCount);
    const std::string engineCtxTag = BuildEngineCtxTag(param.tag, itemCount);
    std::shared_ptr<std::mutex> initMutex;
    {
        std::lock_guard<std::mutex> guard(g_ctxMutex);
        std::shared_ptr<std::mutex> &ctxMutex = g_ctxInitMutexByKey[ctxKey];
        if (ctxMutex == nullptr) {
            ctxMutex = std::make_shared<std::mutex>();
        }
        initMutex = ctxMutex;
    }
    CcuContextData *ctx = nullptr;
    {
        std::lock_guard<std::mutex> guard(*initMutex);
        CHK_RET(InitCcuContext(comm, engineCtxTag.c_str(), param, ctx));
    }
    CHK_RET(LaunchKernel(comm, param, *ctx, stream));
    return HCCL_SUCCESS;
}
