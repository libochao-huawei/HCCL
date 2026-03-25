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

#include "hccl_res_dl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HcomProInfoTmp {
#define MAX_LENGTH 128
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
}HcomProInfoTmp;

HcclResult __attribute__((weak)) HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread);
HcclResult __attribute__((weak)) HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread);
HcclResult __attribute__((weak)) HcommProfilingReportDeviceHcclOpInfo(HcomProInfoTmp profInfo);
HcclResult __attribute__((weak)) HcommProfilingInit(ThreadHandle* threads, uint32_t threadNum);
HcclResult __attribute__((weak)) HcommProfilingEnd(ThreadHandle* threads, uint32_t threadNum);
HcclResult __attribute__((weak)) HcommProfilingReportDeviceOp(const char* groupname);
HcclResult __attribute__((weak)) HcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname);
HcclResult __attribute__((weak)) HcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname);

// 查询函数声明
bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void);
bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void);
bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void);
bool HcommIsSupportHcommProfilingInit(void);
bool HcommIsSupportHcommProfilingEnd(void);
bool HcommIsSupportHcommProfilingReportDeviceOp(void);
bool HcommIsSupportHcommProfilingReportKernelStartTask(void);
bool HcommIsSupportHcommProfilingReportKernelEndTask(void);

// 动态库管理接口
void HcommDeviceProfilingDlInit(void* libHcommHandle);
void HcommDeviceProfilingDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DEVICE_PROFILING_DL_H