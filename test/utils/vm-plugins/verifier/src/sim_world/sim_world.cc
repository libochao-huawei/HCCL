/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_world.h"
#include <iostream>
#include <iomanip>
#include "sim_stream.h"
#include "sim_task_queue.h"
#include "sim_npu.h"

namespace HcclSim {
SimWorld* SimWorld::Global()
{
    static SimWorld* globalSimWorld = new SimWorld;
    return globalSimWorld;
}

void SimWorld::InitSimWorld(const TopoMeta& topoMeta, CheckerDevType devType)
{
    commDomain_.Init(topoMeta);
    InitSimNpuRes(topoMeta, devType);
}

void SimWorld::Reset()
{
    notifyIdGen = 0;
    streamIdGen = 0;
    commDomain_.Reset();
    simNpus_.clear();
    SimTaskQueue::Global()->Reset();
}

void SimWorld::InitSimNpuRes(const TopoMeta& topoMeta, CheckerDevType devType)
{
    simNpus_.clear();
    PodId podId = 0;
    for (const auto& superPod : topoMeta) {
        SerId serId = 0;
        for (const auto& server : superPod) {
            for (const auto& phyId : server) {
                NpuPos npuPos{podId, serId, phyId};
                simNpus_[podId][serId][phyId] = CreateSimNpu(npuPos, devType);
            }
            serId++;
        }
        podId++;
    }
}

SimNpu SimWorld::CreateSimNpu(const NpuPos& npuPos, CheckerDevType devType)
{
    SimNpu simNpu;
    simNpu.SetDevType(devType);
    simNpu.InitSimNpuRes(npuPos);
    return simNpu;
}

SimNpu& SimWorld::GetSimNpuByRankId(RankId rankId)
{
    NpuPos npuPos = commDomain_.GetNpuPosByRankId(rankId);
    return simNpus_[npuPos.superpodId][npuPos.serverId][npuPos.phyId];
}

NpuPos SimWorld::GetNpuPosByRankId(RankId rankId)
{
    return commDomain_.GetNpuPosByRankId(rankId);
}
}