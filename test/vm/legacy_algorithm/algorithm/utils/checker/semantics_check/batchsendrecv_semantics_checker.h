/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: BatchSendRecv语义校验头文件
 * Author: Jiwei
 * Create: 2024-11-11
 */
#ifndef HCCLV1_BATCH_SEND_RECV_SEMANTICS_CHECKER_H
#define HCCLV1_BATCH_SEND_RECV_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckBatchSendRecvSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize);

}

#endif