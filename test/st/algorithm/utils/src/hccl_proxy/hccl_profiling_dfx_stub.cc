/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_dl_stub.h"
#include "hcomm_diag_dl.h"
#include "hccl_common.h"
#include "hcomm_host_profiling_dl.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static uint64_t HcommGetProfilingSysCycleTimeStub()
{
    return 0;
}

static HcclResult HcommProfilingRegThreadStub(HcomProInfoTmp profInfo, ThreadHandle* threads)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingUnRegThreadStub(HcomProInfoTmp profInfo, ThreadHandle* threads)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingReportKernelStub(uint64_t beginTime, const char *profName)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingReportOpStub(HcomProInfoTmp profInfo)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommRegOpInfoStub(const char* commId, void* opInfo, size_t size)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommRegOpTaskExceptionStub(const char* commId, HcommGetOpInfoCallback callback)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcclReportAicpuKernelStub(HcclComm comm, uint64_t beginTime, char *kernelName)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcclDfxRegOpInfoStub(HcclComm comm, void* dfxOpInfo)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcclProfilingReportOpStub(HcclComm comm, uint64_t beginTime)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingReportDeviceOpStub(const char* groupname)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}
static HcclResult HcommProfilingReportKernelStartTaskStub(uint64_t thread, const char* groupname)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingReportKernelEndTaskStub(uint64_t thread, const char* groupname)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingReportMainStreamAndFirstTaskStub(ThreadHandle thread)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingReportMainStreamAndLastTaskStub(ThreadHandle thread)
{
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult HcommProfilingEndStub(ThreadHandle* threads, uint32_t threadNum) {
    HCCL_WARNING("[%s] not support.", __func__);
    return HCCL_SUCCESS;
}

bool HcommIsSupportHcommRegOpInfo(void)
{
    return false;
}

bool HcommIsSupportHcommRegOpTaskException(void)
{
    return false;
}

// device侧会重定义，这里单独声明
HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle) = nullptr;
HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle) = nullptr;
HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfoTmp) = nullptr;
HcclResult (*hcommProfilingInitPtr)(ThreadHandle*, uint32_t) = nullptr;
HcclResult (*hcommProfilingEndPtr)(ThreadHandle*, uint32_t) = nullptr;
HcclResult (*hcommProfilingReportDeviceOpPtr)(const char* groupname) = nullptr;
HcclResult (*hcommProfilingReportKernelStartTaskPtr)(uint64_t thread, const char* groupname) = nullptr;
HcclResult (*hcommProfilingReportKernelEndTaskPtr)(uint64_t thread, const char* groupname) = nullptr;
HcclResult (*hcommRegOpInfoPtr)(const char*, void*, size_t) = nullptr;
HcclResult (*hcommRegOpTaskExceptionPtr)(const char*, HcommGetOpInfoCallback) = nullptr;

void InitHcclProfilingDfxDlStubFunc()
{
    hcommProfilingRegThreadPtr = HcommProfilingRegThreadStub;
    hcommProfilingUnRegThreadPtr = HcommProfilingUnRegThreadStub;
    hcommProfilingReportKernelPtr = HcommProfilingReportKernelStub;
    hcommProfilingReportOpPtr = HcommProfilingReportOpStub;
    hcommProfilingReportKernelEndTaskPtr = HcommProfilingReportKernelEndTaskStub;
    hcommProfilingReportKernelStartTaskPtr = HcommProfilingReportKernelStartTaskStub;
    hcommProfilingEndPtr = HcommProfilingEndStub;
    hcclProfilingReportOpPtr = HcclProfilingReportOpStub;
    hcommProfilingReportDeviceOpPtr = HcommProfilingReportDeviceOpStub;
    hcommGetProfilingSysCycleTimePtr = HcommGetProfilingSysCycleTimeStub;

    hcclDfxRegOpInfoPtr = HcclDfxRegOpInfoStub;
    hcclReportAicpuKernelPtr = HcclReportAicpuKernelStub;
    hcommRegOpInfoPtr = HcommRegOpInfoStub;
    hcommRegOpTaskExceptionPtr = HcommRegOpTaskExceptionStub;
}

#ifdef __cplusplus
}
#endif  // __cplusplus