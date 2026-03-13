/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Broadcast语义校验文件头
 * Author: huangweihao
 * Create: 2024-10-10
 */
#ifndef HCCLV1_BROADCAST_SEMANTICS_CHECKER_H
#define HCCLV1_BROADCAST_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckBroadcastSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize, RankId root);

}

#endif