/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor resource manager
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_RESOURCE_V1_H
#define HCCL_SIM_CCU_RESOURCE_V1_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "CcuResourceCommon.h"
#include "hccl_sim_pub_stub.h"

using namespace std;

class CcuResouceManager;

class CcuResouceV1 {
public:
    CcuResouceV1(int rankId, uint32_t rankSize);
    ~CcuResouceV1() = default;
    void InitInstrInfo(const array<CcuInstrData, DIE_NUM> &ccuInstrInfo);
    void Reset();

private:
    int rankId_{0};
    int rankSize_{0};
    vector<ShmCb *> allShmBase_{};                 // CCU资源：rank进程共享（共享内存中）
    friend CcuResouceManager;                      // 声明资源管理类为友元类
    array<rtCcuTaskInfo_t, DIE_NUM> ccuTaskInfos_; // SQE task参数：rank进程独立
    array<CcuInstrData, DIE_NUM>    instrSpace_;   // ccu指令空间资源：rank进程独立
    array<array<CcuInfo, SimCcuV1::MAX_CCU_CHANNEL_NUM>, DIE_NUM> channelId2RmtRankMap_; // channelId映射表：rank进程独立
};

#endif // HCCL_SIM_CCU_RESOURCE_V1_H
