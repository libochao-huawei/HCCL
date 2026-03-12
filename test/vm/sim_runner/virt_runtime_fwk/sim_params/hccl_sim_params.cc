/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim params
 */

#include "hccl_sim_params.h"
#include "rts_stub.h"
#include <iostream>
#include <thread>
#include "hccl.h"
#include "hccl_sim_pub_stub.h"

SimParams::~SimParams()
{
    if (!isAicpu) {
        aclrtFreeHost(sendBuf);
        aclrtFreeHost(recvBuf);
    } 
}
