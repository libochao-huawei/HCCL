/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: coll service stub 头文件
 * Author:
 * Create: 2025-06-20
 */

#ifndef HCCLV2_ALG_CONFIG_H
#define HCCLV2_ALG_CONFIG_H

#include "dma_mode.h"

using namespace checker;

namespace Hccl {

constexpr u32 INVALID_UINT = 0xFFFFFFFF;

struct RankInfo_f {
    s32 rankId;                         // rank 标识，userRank,cloud时hcom计算填入
    std::string superPodId;             // 超节点标识
    u32 superPodIdx = INVALID_UINT;     // SuperPod在ranktable中的自然顺序（用户指定）
    u32 superDeviceId = INVALID_UINT;   // 超节点device id，超节点内唯一
    u32 devicePhyId;
    u32 serverIdx = INVALID_UINT;       // Server在ranktable中的自然顺序（用户指定）
};


struct TopoMetaParam {
    u32 superPodNum = 0;                 // 集群超节点总数
    u32 serverNum = 0;                   // 集群内服务器总数
    u32 rankNum = 0;                     // 通信域内的rank总数
    u32 deviceNum = 0;                   // 当前通信域内device数量
    std::vector<RankInfo_f> rankList;    // 通信域内所有的rank信息
};

struct DavidAlgConfig {
    u32 maxTmpMemSize = 200*1024*1024;
    bool enableDetour = false;
    bool enableAllign = false;
    u64  allignSize = 0;
    DmaMode dmaMode = DmaMode::DEFAULT;
};

}// namespace Hccl
#endif