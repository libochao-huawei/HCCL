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
#include <cstddef>
#include <cstring>

HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode** opParam)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    *opParam = new OpParamGraphMode();
    if (*opParam == nullptr) {
        return HCCL_E_MEMORY;
    }
    // 初始化 opType
    memset((*opParam)->opType, 0, sizeof((*opParam)->opType));
    return HCCL_SUCCESS;
}

HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode opParam)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    delete opParam;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode opParam, const char* opType)
{
    if (opParam == nullptr || opType == nullptr) {
        return HCCL_E_PARA;
    }
    strncpy_s(opParam->opType, sizeof(opParam->opType), opType, sizeof(opParam->opType) - 1);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcOpResOnlineGraphMode(OpParamGraphMode opParam, uint64_t* opMemSize, uint64_t* streamNum, uint64_t* taskNum, uint64_t* aivCoreNum)
{
    if (opMemSize == nullptr || streamNum == nullptr || taskNum == nullptr || aivCoreNum == nullptr) {
        return HCCL_E_PARA;
    }
    // 资源初始化
    *opMemSize = 0;
    *streamNum = 0;
    *taskNum = 0;
    *aivCoreNum = 0;

    // 为了兼容，创建临时的 ResResponseGraphMode 结构
    ResResponseGraphMode resResponse;
    resResponse.opMemSize = 0;
    resResponse.streamNum = 0;
    resResponse.taskNum = 0;
    resResponse.aivCoreNum = 0;

    // aicpu引擎计算资源
    hccl::HcclCalcAicpuResOffline(&resResponse);

    // 其他引擎补充在下面

    // 将结果复制到输出参数
    *opMemSize = resResponse.opMemSize;
    *streamNum = resResponse.streamNum;
    *taskNum = resResponse.taskNum;
    *aivCoreNum = resResponse.aivCoreNum;

    return HCCL_SUCCESS;
}

HcclResult HcclCalcOpResOfflineGraphMode(OpParamGraphMode opParam, uint64_t* opMemSize, uint64_t* streamNum, uint64_t* taskNum, uint64_t* aivCoreNum)
{
    if (opMemSize == nullptr || streamNum == nullptr || taskNum == nullptr || aivCoreNum == nullptr) {
        return HCCL_E_PARA;
    }
    // 资源初始化
    *opMemSize = 0;
    *streamNum = 0;
    *taskNum = 0;
    *aivCoreNum = 0;

    // 为了兼容，创建临时的 ResResponseGraphMode 结构
    ResResponseGraphMode resResponse;
    resResponse.opMemSize = 0;
    resResponse.streamNum = 0;
    resResponse.taskNum = 0;
    resResponse.aivCoreNum = 0;

    // aicpu引擎计算资源
    hccl::HcclCalcAicpuResOffline(&resResponse);

    // 其他引擎补充在下面

    // 将结果复制到输出参数
    *opMemSize = resResponse.opMemSize;
    *streamNum = resResponse.streamNum;
    *taskNum = resResponse.taskNum;
    *aivCoreNum = resResponse.aivCoreNum;

    return HCCL_SUCCESS;
}

namespace hccl {
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode *resResponse)
{
    if (resResponse == nullptr) {
        return HCCL_E_PARA;
    }
    uint64_t aicpuOpMemSize = 0;
    uint64_t aicpuStreamNum = 0;
    uint64_t aicpuTaskNum = 3;

    resResponse->opMemSize = std::max(resResponse->opMemSize, aicpuOpMemSize);
    resResponse->streamNum = std::max(resResponse->streamNum, aicpuStreamNum);
    resResponse->taskNum = std::max(resResponse->taskNum, aicpuTaskNum);
    return HCCL_SUCCESS;
}
} // namespace hccl
