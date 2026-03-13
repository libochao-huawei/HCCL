/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: All2All语义校验文件头
 * Author: yinding
 * Create: 2024-09-30
 */
#ifndef HCCLV1_ALL2ALL_SEMANTICS_CHECKER_H
#define HCCLV1_ALL2ALL_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckAll2AllSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                     CheckerOpParam &opParam);

}

#endif