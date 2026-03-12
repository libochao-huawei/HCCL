/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Scatter语义校验文件头
 * Author: yinding
 * Create: 2024-10-30
 */
#ifndef HCCLV1_SCATTER_SEMANTICS_CHECKER_H
#define HCCLV1_SCATTER_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckScatterSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize, RankId root);

}

#endif