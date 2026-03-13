/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor resource manager
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "CcuResourceManager.h"
#include "CcuExecutorManager.h"
#include "ccuMicrocodeV1.h"
#include "SimRunnerMgr.h"

void CcuResouceV1::Reset()
{
    
}

CcuResouceV1::CcuResouceV1(int rankId, uint32_t rankSize) {
    rankId_ = rankId;
    allShmBase_.resize(rankSize);
    for (uint32_t devId = 0; devId < rankSize; devId++) {
        void* rankShmBasePtr = SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetShmCbBaseByRank(devId);
        if (rankShmBasePtr == nullptr) {
            HCCL_ERROR("[CcuResouceManager][Init] GetShmCbBase ptr is nullptr, rankId[%d]");
            return;
        }
        allShmBase_[devId] = reinterpret_cast<ShmCb *>(rankShmBasePtr);
    }
}

void CcuResouceV1::InitInstrInfo(const array<CcuInstrData, DIE_NUM> &ccuInstrInfo)
{
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        instrSpace_[dieId] = ccuInstrInfo[dieId];
    }
}
