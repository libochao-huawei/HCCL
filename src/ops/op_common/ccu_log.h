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
            HCCL_ERROR("[%s]call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet;                               \
        }                                             \
    } while (0)


HcclResult inline ConvertCcuToHccl(CcuResult ccuResult) {
    switch (ccuResult) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;;
        
        default:
            return HCCL_E_INTERNAL;
    }
}

#endif //CCU_LOG_H
