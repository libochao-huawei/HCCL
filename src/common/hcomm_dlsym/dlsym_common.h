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

#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include "dlog_pub.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"

/*
 * 8.5.0 CANN 下 9.0.0-only 类型的完整桩定义。
 * 枚举值/结构体布局与 9.0.0 SDK 一致，使主源文件无需 #if 即可编译通过；
 * 运行时通过弱符号机制（DEFINE_WEAK_FUNC）返回错误，不会真正走到 9.0.0 逻辑。
 */
#if CANN_VERSION_NUM < 90000000
#ifdef __cplusplus
extern "C" {
#endif

typedef void *HcclCommSymWindow;
typedef void *HcclMemHandle;
typedef int32_t (Callback)(uint64_t, int32_t);

typedef enum {
    COMM_MEM_TYPE_INVALID = -1,
    COMM_MEM_TYPE_DEVICE = 0,
    COMM_MEM_TYPE_HOST = 1
} CommMemType;

typedef struct {
    CommMemType type;
    void *addr;
    uint64_t size;
} CommMem;

typedef enum {
    ENDPOINT_ATTR_INVALID = -1,
    ENDPOINT_ATTR_BW_COEFF = 0,
    ENDPOINT_ATTR_DIE_ID = 1,
    ENDPOINT_ATTR_LOCATION = 2
} EndpointAttr;

typedef uint32_t EndpointAttrBwCoeff;
typedef uint32_t EndpointAttrDieId;
typedef uint32_t EndpointAttrLocation;
typedef uint32_t EndpointAttrRdma;
typedef uint32_t EndpointAttrSdma;

typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID = -1,
    HCCL_OP_EXPANSION_MODE_AI_CPU = 0,
    HCCL_OP_EXPANSION_MODE_AIV = 1,
    HCCL_OP_EXPANSION_MODE_HOST = 2,
    HCCL_OP_EXPANSION_MODE_HOST_TS = 3,
    HCCL_OP_EXPANSION_CCU_MS = 4,
    HCCL_OP_EXPANSION_CCU_SCHED = 5,
    HCCL_OP_EXPANSION_AIV_ONLY = 6
} HcclOpExpansionMode;

typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255
} HcclCommStatus;

typedef enum {
    HCCL_CONFIG_TYPE_INVALID = -1,
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE = 0
} HcclConfigType;

typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;

extern HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType,
    uint32_t infoLen, void *info) __attribute__((weak));

#define HCOMM_ALG_TAG_LENGTH 288

struct HcclDfxOpInfo {
    CommAbiHeader header;
    uint64_t beginTime;
    uint64_t endTime;
    uint32_t opMode;
    uint32_t opType;
    uint32_t reduceOp;
    uint32_t dataType;
    uint32_t outputType;
    uint64_t dataCount;
    uint32_t root;
    char algTag[HCOMM_ALG_TAG_LENGTH];
    CommEngine engine;
    uint64_t cpuTsThread;
    uint32_t cpuWaitAicpuNotifyIdx;
    uint32_t cpuWaitAicpuNotifyId;
    int8_t reserve[128];
};

#define COMM_PROTOCOL_UBC_CTP ((CommProtocol)4)
#define COMM_PROTOCOL_UBC_TP  ((CommProtocol)5)
#define COMM_PROTOCOL_UB_MEM  ((CommProtocol)6)

#define COMM_TOPO_A2AXSERVER ((CommTopo)4)
#define COMM_TOPO_CUSTOM     ((CommTopo)5)

#define COMM_ADDR_TYPE_EID ((CommAddrType)3)
#define COMM_ADDR_EID_LEN 36

#ifdef __cplusplus
}
#endif
#endif /* CANN_VERSION_NUM < 90000000 */

#ifndef HCOMM_PRO_INFO_TMP_DEFINED
#define HCOMM_PRO_INFO_TMP_DEFINED
#ifdef __cplusplus
extern "C" {
#endif
#define HCOM_PRO_INFO_MAX_LENGTH 128
typedef struct HcomProInfoTmp {
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[HCOM_PRO_INFO_MAX_LENGTH];
    char commName[HCOM_PRO_INFO_MAX_LENGTH];
    char algType[HCOM_PRO_INFO_MAX_LENGTH];
    bool isCapture;
    bool isAiv;
    uint8_t reserved[HCOM_PRO_INFO_MAX_LENGTH];
} HcomProInfoTmp;
#ifdef __cplusplus
}
#endif
#endif /* HCOMM_PRO_INFO_TMP_DEFINED */

#ifdef __cplusplus
extern "C" {
#endif

#define HCCL_LOG_DEBUG DLOG_DEBUG
#define HCCL_LOG_INFO  DLOG_INFO
#define HCCL_LOG_WARN  DLOG_WARN
#define HCCL_LOG_ERROR DLOG_ERROR

#define LOG_FUNC(module, level, fmt, ...) do { \
    DlogRecord(module, level, fmt, ##__VA_ARGS__); \
} while (0)

#define HCCL_LOG_PRINT(moduleId, logType, format, ...) do { \
    LOG_FUNC(moduleId, logType, "[%s:%d] [%u]" format, __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
} while(0)

#define HCCL_RUN_LOG_PRINT(format, ...) do { \
    LOG_FUNC(HCCL_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u]" format, \
             __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
} while(0)

/* 预定义日志宏, 便于使用 */
#define HCCL_COMPAT_DEBUG(format, ...) do { \
    HCCL_LOG_PRINT(HCCL, HCCL_LOG_DEBUG, format, ##__VA_ARGS__); \
} while(0)

#define HCCL_COMPAT_ERROR(format, ...) do { \
    HCCL_LOG_PRINT(HCCL, HCCL_LOG_ERROR, format, ##__VA_ARGS__); \
} while(0)

#define DECL_WEAK_FUNC(type, func_name, ...) \
    type func_name(__VA_ARGS__) __attribute__((weak));

#define DEFINE_WEAK_FUNC(type, func_name, ...) \
    static bool g_##func_name##Supported = false; \
    extern "C" bool HcommIsSupport##func_name(void) { \
        return g_##func_name##Supported; \
    } \
    type func_name(__VA_ARGS__) __attribute__((weak)); \
    type func_name(__VA_ARGS__) \
    { \
        HCCL_COMPAT_ERROR("[HcclWrapper] %s not supported", __func__); \
        return (type)(-1); \
    }

#define DECL_SUPPORT_FLAG(func_name) \
    extern "C" bool HcommIsSupport##func_name(void)

#define INIT_SUPPORT_FLAG(handle, func_name) \
    do { \
        void *ptr = (void *)dlsym(handle, #func_name); \
        if (ptr == nullptr) { \
            g_##func_name##Supported = false; \
            HCCL_COMPAT_DEBUG("[HcclWrapper] %s not supported", #func_name); \
        } else { \
            g_##func_name##Supported = true; \
        } \
    } while(0)


#ifdef __cplusplus
}
#endif

#endif // DLSYM_COMMON_H