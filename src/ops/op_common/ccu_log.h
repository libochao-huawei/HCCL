/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

#ifndef CCU_LOG_H
#define CCU_LOG_H

#include "log.h"
#if CANN_VERSION_NUM < 90000000
typedef enum {
    CCU_SUCCESS = 0,               /**< success */
    CCU_E_PARA = 1,                /**< parameter error */
    CCU_E_PTR = 2,                 /**< empty pointer */
    CCU_E_MEMORY = 3,              /**< memory error */
    CCU_E_INTERNAL = 4,            /**< internal error */
    CCU_E_NOT_SUPPORT = 5,         /**< not support feature */
    CCU_E_NOT_FOUND = 6,           /**< not found specific resource */
    CCU_E_UNAVAIL = 7,             /**< resource unavailable */
    CCU_E_SYSCALL = 8,             /**< call system interface error */
    CCU_E_TIMEOUT = 9,             /**< timeout */
    CCU_E_OPEN_FILE_FAILURE = 10,  /**< open file fail */
    CCU_E_TCP_CONNECT = 11,        /**< tcp connect fail */
    CCU_E_ROCE_CONNECT = 12,       /**< roce connect fail */
    CCU_E_TCP_TRANSFER = 13,       /**< tcp transfer fail */
    CCU_E_ROCE_TRANSFER = 14,      /**< roce transfer fail */
    CCU_E_RUNTIME = 15,            /**< call runtime api fail */
    CCU_E_DRV = 16,                /**< call driver api fail */
    CCU_E_PROFILING = 17,          /**< call profiling api fail */
    CCU_E_CCE = 18,                /**< call cce api fail */
    CCU_E_NETWORK = 19,            /**< call network api fail */
    CCU_E_AGAIN = 20,              /**< try again */
    CCU_E_REMOTE = 21,             /**< error cqe */
    CCU_E_SUSPENDING = 22,         /**< error communicator suspending */
    CCU_E_OPRETRY_FAIL = 23,       /**< retry constraint */
    CCU_E_OOM = 24,                /**< out of memory */
    CCU_E_IN_STATUS = 1041,        /**< The error information is in the status. */

    /*
     * 以下错误码采用显式分段赋值，避免隐式自增导致的 ABI 漂移：
     *   - 1042..1099  driver / 其它通用错误，预留 ~58 个槽位
     *   - 1100..1199  资源不可用专属网段，CCU_CHK_RES_UNAVAIL 依赖
     *                 (ccuRet > CCU_E_RES_UNAVAIL_START && ccuRet < CCU_E_RES_UNAVAIL_END)
     *                 该谓词；新资源类型必须在 1101..1198 内追加，
     *                 严禁在该范围插入非"资源不可用"语义的错误码。
     *   - 1200..      其它分类错误
     * 任何新增错误码必须显式赋值，不得依赖隐式自增。
     */
    CCU_E_DRV_INIT_FAILED = 1042,
    CCU_E_DRV_BUSY        = 1043,

    /* === 资源不足类错误（必须落在 (START, END) 区间内）=== */
    CCU_E_RES_UNAVAIL_START = 1100,

    CCU_E_CHANNEL_CTX_UNAVAIL = 1101,
    CCU_E_JETTY_CTX_UNAVAIL   = 1102,
    CCU_E_WQEBB_UNAVAIL       = 1103,
    CCU_E_MS_UNAVAIL          = 1104,
    CCU_E_LOOP_UNAVAIL        = 1105,
    CCU_E_CKE_UNAVAIL         = 1106,
    CCU_E_XN_UNAVAIL          = 1107,
    CCU_E_GSA_UNAVAIL         = 1108,
    /* 新资源类型在此追加，下一可用值为 1109，严禁超过 1198 */

    CCU_E_RES_UNAVAIL_END = 1199,

    /* === 其它分类错误 === */
    CCU_E_TRANSLATE_FAILED    = 1200,
    CCU_E_ALREADY_BOUND       = 1201,
    CCU_E_LOOP_BODY_UNDEFINED = 1202,

    CCU_E_RESERVED = 0x7FFFFFFF    /**< reserved，固定哨兵值 */
} CcuResult;
#endif /* CANN_VERSION_NUM < 90000000 */

#define HCCL_TO_CCU_RET(hcclRet) static_cast<CcuResult>(hcclRet)
/* 检查函数返回值, 并返回指定错误码 */
#define CCU_CHK_RET(call)                                 \
    do {                                              \
        CcuResult ccuRet = HCCL_TO_CCU_RET(call);                        \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) {                    \
            HCCL_ERROR("[%s]call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet;                               \
        }                                             \
    } while (0)

HcclResult ConvertCcuToHccl(CcuResult ccuResult);

#endif //CCU_LOG_H
