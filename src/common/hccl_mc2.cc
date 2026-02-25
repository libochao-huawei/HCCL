/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "hccl_mc2.h"
#include "log.h"
#include "sal.h"
#include "alg_env_config.h"
#include "hccl_inner.h"

HcclResult HcclKfcAllocOpArgs(void **opArgs)
{
    CHK_PTR_NULL(opArgs);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcAllocOpArgs, start malloc opArgs in %p", opArgs);
    }
    HcclOpArgs *opArgsMem = (HcclOpArgs *)malloc(sizeof(HcclOpArgs));
    if (opArgsMem == nullptr) {
        HCCL_ERROR("[HcclKfcAllocOpArgs] malloc HcclOpArgs mem failed, please check.");
        return HCCL_E_INTERNAL;
    }
    opArgsMem->Init();
    *opArgs = opArgsMem;
    HCCL_RUN_INFO("[HcclKfcAllocOpArgs] malloc HcclOpArgs success, please fill mem[%p->%p] in it.", opArgs, *opArgs);
    return HCCL_SUCCESS;
}

HcclResult HcclKfcFreeOpArgs(void *opArgs)
{
    CHK_PTR_NULL(opArgs);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcFreeOpArgs, start free opArgs in %p", opArgs);
    }
    free(opArgs);
    opArgs = nullptr;
    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetSrcDataType(void *opArgs, uint8_t srcDataType)
{
    CHK_PTR_NULL(opArgs);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcOpArgsSetSrcDataType, opArgs[%p] set srcDataType[%u]", opArgs, srcDataType);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->srcDataType = static_cast<HcclDataType>(srcDataType);
    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetDstDataType(void *opArgs, uint8_t dstDataType)
{
    CHK_PTR_NULL(opArgs);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcOpArgsSetDstDataType, opArgs[%p] set dstDataType[%u]", opArgs, dstDataType);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->dstDataType = static_cast<HcclDataType>(dstDataType);
    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetReduceType(void *opArgs, uint32_t reduceType)
{
    CHK_PTR_NULL(opArgs);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcOpArgsSetReduceType, opArgs[%p] set reduceType[%u]", opArgs, reduceType);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->reduceType = static_cast<HcclReduceOp>(reduceType);
    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetCount(void *opArgs, uint64_t count)
{
    CHK_PTR_NULL(opArgs);
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[%s] count[%llu] is invalid (bigger than MAX count[%lu])", __func__, count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcOpArgsSetCount, opArgs[%p] set count[%lu]", opArgs, count);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->count = count;
    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetAlgConfig(void *opArgs, char *algConfig)
{
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(algConfig);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcOpArgsSetAlgConfig, opArgs[%p] set algConfig[%s]", opArgs, algConfig);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    s32 ret = strcpy_s(opArgsPtr->algConfig, ALG_CONFIG_SIZE, algConfig);
    if (ret != EOK) {
        HCCL_ERROR("[%s] strcpy_s algConfig failed, ret[%d]", __func__, ret);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetCommEngine(void *opArgs, uint8_t commEngine)
{
    CHK_PTR_NULL(opArgs);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcOpArgsSetCommEngine, opArgs[%p] set commEngine[%u]", opArgs, commEngine);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->commEngine = static_cast<CommEngine>(commEngine);
    return HCCL_SUCCESS;
}

HcclResult HcclCreateOpResCtx(HcclComm comm, uint8_t opType, void *opArgs, void **opResCtx)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(opResCtx);

    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcCreateOpResCtx, opType[%u], opArgs[%p] opResCtx[%p]", opType, opArgs, opResCtx);
    }
    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);

    CHK_RET(HcclCreateOpResCtxInner(comm, opType, opArgsPtr->srcDataType, opArgsPtr->dstDataType,
            opArgsPtr->reduceType, opArgsPtr->count, &opArgsPtr->algConfig, opArgsPtr->commEngine, opResCtx));

    return HCCL_SUCCESS;
}