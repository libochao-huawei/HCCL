//
// Created by x30067372 on 26-4-20.
//

#ifndef CCU_LOG_H
#define CCU_LOG_H

#include "log.h"

#define HCCL_TO_CCU_RET(hcclRet) static_cast<CcuResult>(hcclRet)
/* 检查函数返回值, 并返回指定错误码 */
#define CCU_CHK_RET(call)                                 \
    do {                                              \
        CcuResult ccuRet = HCCL_TO_CCU_RET(call);                        \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) {                    \
            if (ccuRet == CCU_E_AGAIN) {                \
                HCCL_WARNING("[%s]call trace: ccuRet -> %d", __func__, ccuRet); \
            } else {                                  \
                HCCL_ERROR("[%s]call trace: ccuRet -> %d", __func__, ccuRet); \
            }                                         \
            return ccuRet;                               \
        }                                             \
    } while (0)


HcclResult inline ConvertCcuToHccl(CcuResult ccuResult) {
    switch (ccuResult) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_MEMORY: return HCCL_E_MEMORY;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        case CCU_E_SYSCALL: return HCCL_E_SYSCALL;
        case CCU_E_TIMEOUT: return HCCL_E_TIMEOUT;
        case CCU_E_OPEN_FILE_FAILURE: return HCCL_E_OPEN_FILE_FAILURE;
        case CCU_E_TCP_CONNECT: return HCCL_E_TCP_CONNECT;
        case CCU_E_ROCE_CONNECT: return HCCL_E_ROCE_CONNECT;
        case CCU_E_TCP_TRANSFER: return HCCL_E_TCP_TRANSFER;
        case CCU_E_ROCE_TRANSFER: return HCCL_E_ROCE_TRANSFER;
        case CCU_E_RUNTIME: return HCCL_E_RUNTIME;
        case CCU_E_DRV: return HCCL_E_DRV;
        case CCU_E_PROFILING: return HCCL_E_PROFILING;
        case CCU_E_CCE: return HCCL_E_CCE;
        case CCU_E_NETWORK: return HCCL_E_NETWORK;
        case CCU_E_AGAIN: return HCCL_E_AGAIN;
        case CCU_E_REMOTE: return HCCL_E_REMOTE;
        case CCU_E_SUSPENDING: return HCCL_E_SUSPENDING;
        case CCU_E_OPRETRY_FAIL:
        case CCU_E_OOM:
        case CCU_E_IN_STATUS:
            return HCCL_E_INTERNAL;
        default:
            return HCCL_E_INTERNAL;
    }
}

#endif //CCU_LOG_H
