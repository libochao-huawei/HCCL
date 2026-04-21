/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DEVICE_PROFILING_DL_H
#define HCOMM_DEVICE_PROFILING_DL_H

#include "dlsym_common.h"
#include "hccl_res_dl.h"

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcommProfilingReportMainStreamAndFirstTask, ThreadHandle thread);
DECL_WEAK_FUNC(HcclResult, HcommProfilingReportMainStreamAndLastTask, ThreadHandle thread);
DECL_WEAK_FUNC(HcclResult, HcommProfilingReportDeviceHcclOpInfo, HcomProInfoTmp profInfo);
DECL_WEAK_FUNC(HcclResult, HcommProfilingInit, ThreadHandle* threads, uint32_t threadNum);
DECL_WEAK_FUNC(HcclResult, HcommProfilingEnd, ThreadHandle* threads, uint32_t threadNum);
DECL_WEAK_FUNC(HcclResult, HcommProfilingReportDeviceOp, const char* groupname);
DECL_WEAK_FUNC(HcclResult, HcommProfilingReportKernelStartTask, uint64_t thread, const char* groupname);
DECL_WEAK_FUNC(HcclResult, HcommProfilingReportKernelEndTask, uint64_t thread, const char* groupname);
DECL_WEAK_FUNC(HcclResult, HcclDfxRegOpInfoByCommId, char* commId, void* hcclDfxOpInfo);
DECL_SUPPORT_FLAG(HcommProfilingReportMainStreamAndFirstTask);
DECL_SUPPORT_FLAG(HcommProfilingReportMainStreamAndLastTask);
DECL_SUPPORT_FLAG(HcommProfilingReportDeviceHcclOpInfo);
DECL_SUPPORT_FLAG(HcommProfilingInit);
DECL_SUPPORT_FLAG(HcommProfilingEnd);
DECL_SUPPORT_FLAG(HcommProfilingReportDeviceOp);
DECL_SUPPORT_FLAG(HcommProfilingReportKernelStartTask);
DECL_SUPPORT_FLAG(HcommProfilingReportKernelEndTask);

// 动态库管理接口
void HcommDeviceProfilingDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DEVICE_PROFILING_DL_H