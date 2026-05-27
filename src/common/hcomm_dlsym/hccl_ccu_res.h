/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Stub header for CANN 8.5 compatibility - provides minimal type declarations
 * needed by hcomm_dlsym layer. Full definitions come from hcomm SDK at runtime.
 */

#ifndef HCCL_CCU_RES_H
#define HCCL_CCU_RES_H

#include "hccl/hccl_types.h"
#include "ccu_res.h"

#ifdef __cplusplus
extern "C" {
#endif

extern HcclResult HcclCommQueryCcuIns(HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum);

#ifdef __cplusplus
}
#endif

#endif // HCCL_CCU_RES_H