/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ReduceScatterV语义校验文件头
 */
#ifndef HCCLV1_REDUCE_SCATTER_V_SEMANTICS_CHECKER_H
#define HCCLV1_REDUCE_SCATTER_V_SEMANTICS_CHECKER_H
 
#include "hccl_types.h"
#include "check_utils.h"
 
namespace checker {
 
HcclResult TaskCheckReduceScatterVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                            CheckerOpParam &opParam);
}
 
#endif