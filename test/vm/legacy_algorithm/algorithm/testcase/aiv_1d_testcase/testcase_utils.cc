/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "testcase_utils.h"
#include <stdlib.h>
 
void ClearHcclEnv()
{
    unsetenv("HCCL_HIGH_PERF_ENABLE");
    unsetenv("HCCL_DETERMINISTIC");
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    unsetenv("HCCL_ALGO");
    unsetenv("HCCL_BUFFSIZE");
    unsetenv("HCCL_OP_BASE_FFTS_MODE_ENABLE");
    unsetenv("HCCL_INTER_HCCS_DISABLE");
    unsetenv("HCCL_OP_EXPANSION_MODE");
    unsetenv("HCCL_CONCURRENT_ENABLE");
    unsetenv("HCCL_IODIE_NUM");
    unsetenv("HCCL_DETOUR");
    return;
}
 
std::vector<u64> GenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}