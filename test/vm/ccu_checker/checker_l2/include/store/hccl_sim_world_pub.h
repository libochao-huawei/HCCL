/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_WORLD_PUB_H
#define HCCL_SIM_WORLD_PUB_H

#include "hccl_sim_shm_manager.h"
#include "hccl_common_defs.h"
#include "hccl_shm_pub.h"

#include <iostream>

using NpuPos2Index = ShmMap<uint32_t, size_t>;

// Init
HcclSim::HcclVmResult InitSimWorld(const TopoMeta* topoMeta);

// GetInfo
HcclSim::HcclVmResult GetNpuNum(uint32_t* npuNum);
HcclSim::HcclVmResult GetNpuByNpuPos(const ShmNpuPos& npuPos, ShmSimNpu** simNpu);
HcclSim::HcclVmResult GetNpuByIndex(uint32_t npuIndex, ShmSimNpu** simNpu);

// CommDomain
HcclSim::HcclVmResult SetCommDomain(uint32_t rankSize, uint32_t rankId, const ShmNpuPos& npuPos);
HcclSim::HcclVmResult GetNpuPosByRankId(uint32_t rankId, ShmNpuPos* npuPos);
HcclSim::HcclVmResult GetNpuByRankId(uint32_t rankId, ShmSimNpu** simNpu);
HcclSim::HcclVmResult GetRankIdByNpuPos(const ShmNpuPos& npuPos, uint32_t* rankId);

// Memory
HcclSim::HcclVmResult AllocNpuMemoryGetIdx(const uint32_t rankId, const uint64_t size, size_t* memBlockIdx);
HcclSim::HcclVmResult AllocNpuMemory(const uint32_t rankId, const uint64_t size, void** addr);
HcclSim::HcclVmResult AllocNpuMemory(const ShmNpuPos& npuPos, const uint64_t size, void** addr);
// For Checker
HcclSim::HcclVmResult MockAllocNpuMemory(const uint32_t rankId, const uint64_t size, void** addr);
HcclSim::HcclVmResult MockAllocNpuMemory(const ShmNpuPos& npuPos, const uint64_t size, void** addr);
HcclSim::HcclVmResult RegisterNpuMemory(const uint32_t rankId, const void* addr, const uint64_t size, const uint8_t bufferType);

// Stream
HcclSim::HcclVmResult AllocStream(const uint32_t rankId, void** stream);
HcclSim::HcclVmResult AllocStream(const ShmNpuPos& npuPos, void** stream);
HcclSim::HcclVmResult AllocMainStream(const uint32_t rankId, void** stream);
HcclSim::HcclVmResult AllocSlaveStream(const uint32_t rankId, void** stream);
HcclSim::HcclVmResult ReleaseStream(void* stream);
HcclSim::HcclVmResult ReleaseStream(const uint64_t streamId);

// Notify
HcclSim::HcclVmResult AllocNotify(const uint32_t rankId, void** notify);
HcclSim::HcclVmResult AllocNotify(const ShmNpuPos& npuPos, void** notify);
HcclSim::HcclVmResult ReleaseNotify(void* notify);
HcclSim::HcclVmResult ReleaseNotify(const uint64_t notifyId);
HcclSim::HcclVmResult GetNotifyValue(const uint64_t notifyId, bool* value);
HcclSim::HcclVmResult SetNotifyValue(const uint64_t notifyId, bool value);
HcclSim::HcclVmResult WaitNotifyValue(const uint64_t notifyId, bool* result);

void test();    // todo delete

#endif //HCCL_SIM_WORLD_PUB_H
