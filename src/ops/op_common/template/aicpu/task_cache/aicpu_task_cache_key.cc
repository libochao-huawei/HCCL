/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_key.h"

#include <sstream>

#include "aicpu_task_cache_utils.h"

namespace ops_hccl {

HcclResult AicpuTaskCacheKey::GetAicpuTaskCacheTag(const OpParam& param, std::string& cacheTag)
{
    // 校验opType
    const HcclCMDType opType = param.opType;
    CHK_RET(opType == HcclCMDType::HCCL_CMD_INVALID,
        HCCL_ERROR("[AicpuTaskCacheKey][AicpuTaskCacheKey] opType is invalid"),
        HCCL_E_PARA);

    // 暂时不考虑v类算子 (应该被cache使能约束拦截, 不应该进入本函数), dataType一定不是reserved
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        dataType = param.all2AllDataDes.sendType;
    } else if (AicpuTaskCacheUtils::IsNonVariableOpType(opType)) { // 非alltoall的非v类算子
        dataType = param.dataType;
    }
    CHK_RET(dataType == HcclDataType::HCCL_DATA_TYPE_RESERVED,
        HCCL_ERROR("[AicpuTaskCacheKey][AicpuTaskCacheKey] dataType is reserved"),
        HCCL_E_PARA);

    // 获取其他字段
    const HcclReduceOp reduceType = param.reduceType;
    const bool isZeroCopy = param.isZeroCopy;
    const OpMode opMode = param.opMode;
    const uint64_t inputSize = param.inputSize;

    // 使用'-'作为间隔符, 拼接cacheTag
    // 注意: 把input size放在前面, 如果需要解析, 可以减少解析开销
    // 注意: commId放在最后, 如果需要解析, 无需考虑commId中含有delimiter的情况
    const char delimiter = '-';
    const char* commId = param.commName;
    std::ostringstream oss;
    oss << inputSize << delimiter
        << static_cast<uint8_t>(opType) << delimiter
        << static_cast<uint8_t>(dataType) << delimiter
        << static_cast<uint8_t>(reduceType) << delimiter
        << static_cast<uint8_t>(isZeroCopy) << delimiter
        << static_cast<uint8_t>(opMode) << delimiter
        << commId;
    cacheTag = oss.str();
        
    HCCL_INFO("[AicpuTaskCacheKey][GetAicpuTaskCacheTag] cacheTag[%s] from commId[%s] opType[%d] dataType[%d] "
        "reduceType[%d] isZeroCopy[%d] inputSize[%llu] opMode[%d]",
        cacheTag.c_str(), commId, opType, dataType, reduceType, isZeroCopy, inputSize, opMode);

    return HCCL_SUCCESS;
}

}