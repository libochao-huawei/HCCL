/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: SendRecv语义校验头文件，sendrecv必须同时出现
 * Author: huangweihao
 * Create: 2024-10-10
 */
#ifndef HCCLV1_SEND_RECV_SEMANTICS_CHECKER_H
#define HCCLV1_SEND_RECV_SEMANTICS_CHECKER_H

#include "hccl_types.h"
#include "check_utils.h"

namespace checker {

HcclResult TaskCheckSendRecvSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                      RankId srcRank, RankId dstRank);

}

#endif