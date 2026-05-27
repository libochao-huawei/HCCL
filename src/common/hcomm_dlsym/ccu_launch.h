/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Stub header for CANN 8.5 compatibility - provides minimal type declarations
 * needed by hcomm_dlsym layer. Full definitions come from hcomm SDK at runtime.
 */

#ifndef CCU_LAUNCH_H
#define CCU_LAUNCH_H

#include <stdint.h>
#include "ccu_res.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef THREAD_HANDLE_DEFINED
#define THREAD_HANDLE_DEFINED
typedef uint64_t ThreadHandle;
#endif

extern CcuResult HcommCcuKernelLaunch(ThreadHandle threadHandle, CcuKernelHandle kernelHandle,
    const void *taskArgs, uint32_t argSize);
extern CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle);
extern CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, const char *kernelFuncName,
    const void *kernelFunc, const void *kernelArg, CcuKernelHandle *kernelHandle);
extern CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle);

#ifdef __cplusplus
}
#endif

#endif // CCU_LAUNCH_H