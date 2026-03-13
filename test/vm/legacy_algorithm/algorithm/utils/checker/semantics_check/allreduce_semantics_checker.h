/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: AllReduce语义校验文件头
 * Author: yinding
 * Create: 2024-02-04
 */
#ifndef HCCLV1_ALLREDUCE_SEMANTICS_CHECKER_H
#define HCCLV1_ALLREDUCE_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckAllReduceSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                       CheckerReduceOp reduceType);

}

#endif