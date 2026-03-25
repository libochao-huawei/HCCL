/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HOST_PROFILING_DL_H
#define HCOMM_HOST_PROFILING_DL_H

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

HcclResult __attribute__((weak)) HcommProfilingRegThread(HcomProInfoTmp profInfo, ThreadHandle* threads);
HcclResult __attribute__((weak)) HcommProfilingUnRegThread(HcomProInfoTmp profInfo, ThreadHandle* threads);
HcclResult __attribute__((weak)) HcommProfilingReportKernel(uint64_t beginTime, const char* profName);
HcclResult __attribute__((weak)) HcommProfilingReportOp(HcomProInfoTmp profInfo);
uint64_t __attribute__((weak)) HcommGetProfilingSysCycleTime();
HcclResult __attribute__((weak)) HcclDfxRegOpInfo(HcclComm comm, void* dfxOpInfo);
HcclResult __attribute__((weak)) HcclProfilingReportOp(HcclComm comm, uint64_t beginTime);
HcclResult __attribute__((weak)) HcclReportAicpuKernel(HcclComm comm, uint64_t beginTime, char *kernelName);

// 查询函数声明
bool HcommIsSupportHcommProfilingRegThread(void);
bool HcommIsSupportHcommProfilingUnRegThread(void);
bool HcommIsSupportHcommProfilingReportKernel(void);
bool HcommIsSupportHcommProfilingReportOp(void);
bool HcommIsSupportHcommGetProfilingSysCycleTime(void);
bool HcommIsSupportHcclDfxRegOpInfo(void);
bool HcommIsSupportHcclProfilingReportOp(void);
bool HcommIsSupportHcclProfilingReportOp(void);

// 动态库管理接口
void HcommProfilingDlInit(void* libHcommHandle);
void HcommProfilingDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PROFILING_DL_H