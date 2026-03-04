/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_resource_graph_mode.h"


HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode *opParam, ResResponseGraphMode *resResponse)
{
    // 资源初始化
    resResponse->opMemSize = 0;
    resResponse->streamNum = 0;
    resResponse->taskNum = 0;
    resResponse->aivCoreNum = 0;

    // aicpu引擎计算资源
    HcclCalcAicpuResOffline(resResponse);

    // 其他引擎补充在下面
    return HCCL_SUCCESS;
}

HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode *opParam, ResResponseGraphMode *resResponse)
{
    resResponse->opMemSize = 0;
    resResponse->streamNum = 0;
    resResponse->taskNum = 0;
    resResponse->aivCoreNum = 0;

    // aicpu引擎计算资源
    HcclCalcAicpuResOffline(resResponse);

    // 其他引擎补充在下面
    return HCCL_SUCCESS;
}

namespace hccl {
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode *resResponse)
{
    uint64_t aicpuOpMemSize = 0;
    uint64_t aicpuStreamNum = 0;
    uint64_t aicpuTaskNum = 3;

    resResponse->opMemSize = std::max(resResponse->opMemSize, aicpuOpMemSize);
    resResponse->streamNum = std::max(resResponse->streamNum, aicpuStreamNum);
    resResponse->taskNum = std::max(resResponse->taskNum, aicpuTaskNum);
    return HCCL_SUCCESS;
}
} // namespace hccl
