/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: op type header file
 * Author: zhangqingwei
 * Create: 2024-02-04
 */

#ifndef HCCLV1_OP_TYPE_H
#define HCCLV1_OP_TYPE_H

#include "enum_factory.h"
namespace hccl {

MAKE_ENUM(OpType, BROADCAST, ALLREDUCE, REDUCE, SEND, RECEIVE, ALLGATHER, ALLGATHERV, REDUCESCATTER, REDUCESCATTERV, ALLTOALLV, ALLTOALLVC,
          ALLTOALL, GATHER, SCATTER, BATCH_SEND_RECV, RESERVED)

inline std::string OpTypeToString(OpType type)
{
    return type.Describe();
}

} // namespace hccl
#endif