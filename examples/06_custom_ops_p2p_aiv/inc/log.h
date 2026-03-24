/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_AIV_LOG_H
#define OPS_HCCL_P2P_AIV_LOG_H

#include <cstdio>
#include <hccl/hccl_types.h>

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_ERROR
#endif

typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_NONE = 4
} LogLevel;

#ifndef LIKELY
#define LIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 1)))
#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))
#endif

#ifdef HOST_COMPILE
#define HCCL_DEBUG(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_DEBUG) { printf("[DEBUG][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); } } while (0)
#define HCCL_INFO(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_INFO) { printf("[INFO][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); } } while (0)
#define HCCL_WARNING(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_WARNING) { printf("[WARN][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); } } while (0)
#define HCCL_ERROR(format, ...) do { if (LOG_LEVEL <= LOG_LEVEL_ERROR) { printf("[ERROR][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); } } while (0)
#else
#define HCCL_DEBUG(format, ...) do {} while (0)
#define HCCL_INFO(format, ...) do {} while (0)
#define HCCL_WARNING(format, ...) do {} while (0)
#define HCCL_ERROR(format, ...) do {} while (0)
#endif

#define CHK_PTR_NULL(ptr) do { if (UNLIKELY((ptr) == nullptr)) { HCCL_ERROR("ptr [%s] is nullptr", #ptr); return HCCL_E_PTR; } } while (0)
#define CHK_PRT_RET(result, exeLog, retCode) do { if (UNLIKELY(result)) { exeLog; return retCode; } } while (0)
#define CHK_RET(call) do { int32_t hcclRet = (call); if (UNLIKELY(hcclRet != HCCL_SUCCESS)) { HCCL_ERROR("hccl call failed, ret=%d", hcclRet); return static_cast<HcclResult>(hcclRet); } } while (0)
#define CHK_HCOMM(call) do { int32_t hcommRet = (call); if (UNLIKELY(hcommRet != 0)) { HCCL_ERROR("hcomm call failed, ret=%d", hcommRet); return HCCL_E_INTERNAL; } } while (0)
#define ACLCHECK(cmd) do { aclError aclRet = (cmd); if (UNLIKELY(aclRet != ACL_SUCCESS)) { HCCL_ERROR("acl call failed %s:%d, ret=%d", __FILE__, __LINE__, aclRet); return HCCL_E_RUNTIME; } } while (0)

#endif // OPS_HCCL_P2P_AIV_LOG_H
