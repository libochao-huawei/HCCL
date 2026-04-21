/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RES_DL_H
#define HCCL_RES_DL_H

#include "dlsym_common.h"
#include "hccl_res.h"

#if CANN_VERSION_NUM >= 90000000
#include "hccl_res_expt.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcclGetRemoteIpcHcclBuf, HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size);
DECL_WEAK_FUNC(int32_t, HcclTaskRegister, HcclComm comm, const char* msgTag, Callback cb);
DECL_WEAK_FUNC(int32_t, HcclTaskUnRegister, HcclComm comm, const char* msgTag);
DECL_WEAK_FUNC(HcclResult, HcclDevMemAcquire, HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated);
DECL_WEAK_FUNC(HcclResult, HcclThreadExportToCommEngine, HcclComm comm, uint32_t threadNum,
    const ThreadHandle* threads, CommEngine dstCommEngine, ThreadHandle* exportedThreads);
DECL_WEAK_FUNC(HcclResult, HcclChannelGetRemoteMems, HcclComm comm, ChannelHandle channel,
    uint32_t* memNum, CommMem** remoteMems, char*** memTags);
DECL_WEAK_FUNC(HcclResult, HcclCommMemReg, HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle);
DECL_WEAK_FUNC(HcclResult, HcclEngineCtxDestroy, HcclComm comm, const char* ctxTag, CommEngine engine);

DECL_SUPPORT_FLAG(HcclThreadExportToCommEngine);
// 动态库管理接口（大驼峰命名）
void HcclResDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RES_DL_H
