/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_AIV_KERNEL_TYPES_H
#define OPS_HCCL_P2P_AIV_KERNEL_TYPES_H

#include <cstdint>

namespace ops_hccl_p2p_aiv {

constexpr uint32_t kP2pAivTaskSend = 0;
constexpr uint32_t kP2pAivTaskRecv = 1;
constexpr uint32_t kP2pAivFlagReadyValue = 1;
constexpr uint32_t kP2pAivFlagDoneValue = 1;
constexpr uint32_t kP2pAivKernelLaunchTimeoutSec = 1800;
constexpr uint32_t kP2pAivKernelBlockNum = 1;
constexpr uint64_t kP2pAivSyncCtxSize = 4096;
constexpr uint32_t kP2pAivKernelCtxVersion = 1;

struct P2pAivSyncState {
    uint32_t ready = 0;
    uint32_t done = 0;
    uint32_t status = 0;
    uint32_t reserved = 0;
};

struct P2pAivKernelParam {
    uint32_t version = kP2pAivKernelCtxVersion;
    uint32_t taskType = 0;
    uint32_t dataType = 0;
    uint32_t rank = 0;
    uint32_t peerRank = 0;
    uint32_t blockNum = kP2pAivKernelBlockNum;
    uint64_t lenBytes = 0;
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t localBufferAddr = 0;
    uint64_t remoteBufferAddr = 0;
    uint64_t localSyncAddr = 0;
    uint64_t remoteSyncAddr = 0;
};

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_KERNEL_TYPES_H
