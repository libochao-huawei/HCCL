/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OPS_HCCL_SRC_OPS_ALLGATHER_OP
#define OPS_HCCL_SRC_OPS_ALLGATHER_OP

#include <string>
#include "hccl.h"
#include "alg_param.h"

#ifdef __cplusplus
extern "C" {
#endif

// 对外导出的 Ring 版 AllGather 接口：
// - sendBuf: 当前 rank 的输入数据起始地址
// - recvBuf: AllGather 后的完整输出地址（按 rank 顺序拼接）
// - sendCount/dataType: 当前 rank 输入元素数和类型
// - comm/stream: 通信域和执行流
HcclResult HcclAllGatherRing(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
// 内部实现入口（out-place）：把输入 sendBuf 聚合后写入 recvBuf。
// tag 用于标识本次集合通信任务，便于资源管理和日志追踪。
HcclResult AllGatherRingOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
    aclrtStream stream, const std::string &tag);
}

#endif
