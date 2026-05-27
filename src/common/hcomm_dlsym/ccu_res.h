/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Stub header for CANN 8.5 compatibility - provides minimal type declarations
 * needed by hcomm_dlsym layer. Full definitions come from hcomm SDK at runtime.
 */

#ifndef CCU_RES_H
#define CCU_RES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CCU_SUCCESS = 0,
    CCU_E_INTERNAL = 1,
    CCU_E_PARAM = 2,
    CCU_E_PARA = CCU_E_PARAM,
    CCU_E_MEMORY = 3,
    CCU_E_PTR = 4,
    CCU_E_TIMEOUT = 5,
    CCU_E_NOT_SUPPORT = 6,
    CCU_E_NOT_FOUND = 7,
    CCU_E_UNAVAIL = 8,
} CcuResult;

typedef uint64_t CcuInsHandle;
typedef uint64_t CcuKernelHandle;

#ifdef __cplusplus
}
#endif

#endif // CCU_RES_H