/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DLSYM_COMMON_H
#define DLSYM_COMMON_H

#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif

// 支持标志（静态，默认 false）
#define DEFINE_SUPPORT_FLAG(name) \
    static bool g_##name##Supported = false; \
    extern "C" bool HcommIsSupport##name(void) { \
        return g_##name##Supported; \
    }

#define DECL_SUPPORT_FLAG(name) \
    extern "C" bool HcommIsSupport##name##(void)

#define INIT_SUPPORT_FLAG(handle, name) \
    do { \
        void *ptr = (void *)dlsym(handle, "name"); \
        if (ptr == nullptr) { \
            g_##name##Supported = false; \
            HCCL_DEBUG("[HcclWrapper] %s not supported", "name"); \
        } else { \
            g_##name##Supported = true; \
        } \
    } while(0)

// 弱符号函数定义
#define DEFINE_WEAK_FUNC_WITH_HCCLRESULT(func_decl) \
    func_decl __attribute__((weak)); \
    func_decl \
    { \
        HCCL_ERROR("[HcclWrapper] %s not supported", __FUNCTION__); \
        return HCCL_E_NOT_SUPPORT; \
    }

#define DEFINE_WEAK_FUNC_WITH_INT32(func_decl) \
    func_decl __attribute__((weak)); \
    func_decl \
    { \
        HCCL_ERROR("[HcclWrapper] %s not supported", __FUNCTION__); \
        return -1; \
    }

// 弱符号函数声明
#define DECL_WEAK_FUNC(func_decl) \
    func_decl __attribute__((weak))

#ifdef __cplusplus
}
#endif

#endif // DLSYM_COMMON_H