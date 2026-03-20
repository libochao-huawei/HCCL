/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlhcomm_function.h"
#include "log.h"

namespace ops_hccl {
DlHcommFunction &DlHcommFunction::GetInstance()
{
    static DlHcommFunction hcclDlHcommFunction;
    (void)hcclDlHcommFunction.DlHcommFunctionInit();
    return hcclDlHcommFunction;
}

DlHcommFunction::~DlHcommFunction()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

HcclResult DlHcommFunction::DlHcommFunctionInterInit()
{
    dlHcclThreadResGetInfo = (HcclResult(*)(HcclComm, ThreadHandle, void*, uint32_t, void**))dlsym(handle_,
        "HcclThreadResGetInfo");
    return HCCL_SUCCESS;
}

HcclResult DlHcommFunction::DlHcommFunctionInit()
{
    if (handle_.load(std::memory_order_acquire) != nullptr) {
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(handleMutex_);

    if (handle_.load(std::memory_order_relaxed) != nullptr) {
        return HCCL_SUCCESS;
    }
    // dlopen
    void* h = dlopen("libhcomm.so", RTLD_NOW);
    handle_.store(h, std::memory_order_relaxed);
    CHK_RET(DlHcommFunctionInterInit());
    return HCCL_SUCCESS;
}
}
