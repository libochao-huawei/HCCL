/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_ALLTOALLV_AICPU_LOAD_KERNEL_H
#define OPS_HCCL_ALLTOALLV_AICPU_LOAD_KERNEL_H

#include <string>
#include <hccl/hccl.h>
#include <acl/acl_rt.h>

namespace ops_hccl_alltoallv_aicpu {

HcclResult LoadAICPUKernel(void);
HcclResult GetKernelFilePath(std::string &binaryPath);
HcclResult LoadBinaryFromFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
    aclrtBinHandle &binHandle);

extern thread_local aclrtBinHandle g_binKernelHandle;

}

#endif