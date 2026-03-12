/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <vector>
#include <atomic>
#include <iostream>
#include <securec.h>
#include "acl/acl_rt.h"
#include "acl/acl_base.h"
#include "dtype_common.h"
#include "rts_device.h"
#include "hccp_common.h"
#include "ascend_hal.h"
#include "hccl_vm_log.h"
#include "atrace_types.h"
#include "mem.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

HcclResult hrtGetDeviceType(DevType &devType)
{
    devType = DevType::DEV_TYPE_910_95;
    HCCL_VM_TRACE("[{}] Return DEV_TYPE_910_95", __func__);
    return HCCL_SUCCESS;
}

rtError_t rtOpenNetService(const rtNetServiceOpenArgs *args)
{
    // hccpThreadStatus = 1;
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    return RT_ERROR_NONE; 
}


rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void *devInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyCreateServer(rtCntNotify_t * const cntNotify, uint64_t flags)
{
    return 0;
}

rtError_t rtsCntNotifyGetId(rtCntNotify_t cntNotify, uint32_t *notifyId)
{
    return ACL_SUCCESS;
}

rtError_t rtGetDevResAddress(rtDevResInfo *const resInfo, rtDevResAddrInfo *const addrInfo)
{
    if (resInfo == nullptr || (resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_NOTIFY_RECORD && resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR)) {
        // 非NotifyRecord场景暂不处理
        return RT_ERROR_NONE;
    }
    // std::cout<<"[ERROR][rtGetDevResAddress] Not support...."<<std::endl;

    uint32_t notifyId = resInfo->resId;
    uint32_t len = 8;
    *(addrInfo->resAddress) = static_cast<uint64_t>(notifyId);
    *(addrInfo->len) = len;
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif  // __cplusplus