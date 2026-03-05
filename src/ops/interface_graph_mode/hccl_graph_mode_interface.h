/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_GRAPH_MODE_INTERFACE_H
#define HCCL_GRAPH_MODE_INTERFACE_H

#include <cstdint>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <acl/acl.h>


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
constexpr uint32_t RES_PACK_TAG_LENGTH = 255;
// 图模式编译阶段资源计算入参
struct OpParamGraphMode
{
    char opType[64]; // 算子类型
}; 

// 图模式编译阶段申请资源
struct ResResponseGraphMode {
    u64 opMemSize = 0;  // 额外申请的scratch数量（不包括cclBuff）
    u32 streamNum = 0;  // 除用户流以外，额外申请的流（不包括算子device展开申请的流）
    u32 taskNum = 0;    // task数量，一般为前同步 + kernel + 后同步
    u32 aivCoreNum = 0;
};

// 图模式执行阶段传入的资源
struct ResPackGraphMode {
    char tag[RES_PACK_TAG_LENGTH];
    std::vector<aclrtStream> streams;
    void* scratchMemAddr;
    u64 scratchMemSize;
};
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_GRAPH_MODE_INTERFACE_H