/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ReduceScatter语义校验文件头
 * Author: shenyutian
 * Create: 2024-03-15
 */
#ifndef HCCLV1_REDUCE_SCATTER_SEMANTICS_CHECKER_H
#define HCCLV1_REDUCE_SCATTER_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckReduceScatterSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                           CheckerReduceOp reduceType);
}

#endif