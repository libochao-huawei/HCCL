/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim interface header
 */

#ifndef HCCL_SIM_OP_FLOW_H
#define HCCL_SIM_OP_FLOW_H

#include <memory>
#include "hccl_sim_params.h"
bool PrepareSimParams(SimParams *params);
bool VerifySimResult(SimParams *params);
void InternalProcess(SimParams *params);
#endif // HCCL_SIM_OP_FLOW_H