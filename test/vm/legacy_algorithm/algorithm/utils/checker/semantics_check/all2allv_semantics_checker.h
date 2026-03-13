/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: All2AllV语义校验文件头
 * Author: yinding
 * Create: 2024-10-15
 */
#ifndef HCCLV1_ALL2ALLV_SEMANTICS_CHECKER_H
#define HCCLV1_ALL2ALLV_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckAll2AllVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                      CheckerOpParam &opParam);

}

#endif