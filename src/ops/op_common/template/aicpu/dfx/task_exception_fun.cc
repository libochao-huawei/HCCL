/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx/task_exception_fun.h"

#include <string>
#include <sstream>
#include <memory>
#include "log.h"
#include "alg_param.h"

namespace ops_hccl {
HcclResult CreateScatter(OpParam *param, ScatterOpInfo *opInfo)
{
    if (param == nullptr || opInfo == nullptr) {
        HCCL_ERROR("%s fail, param[%p] or opInfo[%p] is nullptr", __func__, param, opInfo);
        return HCCL_E_PTR;
    }
    strcpy(opInfo->algTag, param->algTag);
    strcpy(opInfo->commName, param->commName);
    opInfo->count = param->DataDes.count;
    opInfo->dataType = param->DataDes.dataType;
    opInfo->opType = param->opType;
    opInfo->root = param->root;
    opInfo->inputPtr = param->inputPtr;
    opInfo->outputPtr = param->outputPtr;
    return HCCL_SUCCESS;
}

void GetScatterOpInfo(const void *opInfo, char *outPut, size_t size)
{
    const ScatterOpInfo *info = reinterpret_cast<const ScatterOpInfo *>(opInfo);
    std::stringstream ss;
    ss << "tag:" << info->algTag << ", ";
    ss << "group:" << info->commName << ", ";
    ss << "count:" << info->count << ", ";
    ss << "dataType:" << info->dataType << ", ";
    ss << "opType:" << info->opType << ", ";
    ss << "rootId:" << info->root << ", ";
    ss << "dstAddr:0x" << std::hex << info->inputPtr << ", ";
    ss << "srcAddr:0x" << std::hex << info->outputPtr << ".";

    std::string strTmp = ss.str();
    s32 sRet = strncpy_s(outPut, size, strTmp.c_str(), std::min(size, strTmp.size()));
    if (strTmp.size() >= size || sRet != EOK) {
        HCCL_ERROR("%s strncpy_s fail, src size[%u], dst size[%u], sRet[%d]", strTmp.size(), size, sRet);
    }
}

}