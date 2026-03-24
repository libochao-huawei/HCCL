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
constexpr uint32_t kP2pAivTagValue = 1;
constexpr uint32_t kP2pAivKernelLaunchTimeoutSec = 1800;
constexpr uint32_t kP2pAivKernelBlockNum = 1;
constexpr uint64_t kP2pAivCommInfoSize = 2 * 1024 * 1024;
constexpr uint64_t kP2pAivReadyFlagOffset = 0;
constexpr uint64_t kP2pAivDoneFlagOffset = 64;
constexpr uint64_t kP2pAivFlagAreaSize = 128;
constexpr uint32_t kP2pAivKernelCtxVersion = 1;

struct P2pAivKernelParam {
    uint32_t version = kP2pAivKernelCtxVersion;
    uint32_t taskType = 0;
    uint32_t dataType = 0;
    uint32_t rank = 0;
    uint32_t peerRank = 0;
    uint32_t blockNum = kP2pAivKernelBlockNum;
    uint32_t tag = kP2pAivTagValue;
    uint64_t lenBytes = 0;
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t localBufferAddr = 0;
    uint64_t remoteBufferAddr = 0;
    uint64_t localCommInfoAddr = 0;
    uint64_t remoteCommInfoAddr = 0;
};

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_KERNEL_TYPES_H
