/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim params header
 */

#ifndef HCCL_SIM_PARAMS_H
#define HCCL_SIM_PARAMS_H
#include "../hccl_sim_situation.h"
// situation是公用的配置， simParams是执行起来之后各个线程拿到的集合通信参数
struct SimParams {
    Situation situation;
    int serverId;
    int deviceId;
    int myRank;
    std::string commId;
    void* sendBuf;
    void* recvBuf;
    int sendCount;
    int recvCount;
    u64 recvmemSize;
    bool isAicpu{false};
    bool mc2Flag{false};

    virtual ~SimParams();
};
#endif
