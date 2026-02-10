/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "sim_communicator.h"
#include "hccl_sim_world_pub.h"
#include "hccl_common_defs.h"
#include <dlfcn.h>
#include "sim_communicator.h"

using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif
constexpr uint32_t DATA_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t),     // int8
    sizeof(int16_t),    // int16
    sizeof(int32_t),    // int32
    2,              // fp16
    sizeof(float),  // fp32
    sizeof(int64_t),    // int64
    sizeof(uint64_t),    // uint64
    sizeof(uint8_t),     // uint8
    sizeof(uint16_t),    // uint16
    sizeof(uint32_t),    // uint32
    8,              // fp64
    2,              // bfp16
    16,             // int128
    1,              // hif8
    1,              // fp8e4m3
    1               // fp8e5m2
};

uint64_t CalcDataSize(HcclDataType dataType, uint64_t dataCount)
{
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        // invalid data type
        printf("[CalcDataSize] invalid dataType %d\n", dataType);
        return 0;
    }
    uint64_t dataTypeSize = DATA_SIZE_TABLE[dataType];
    return dataTypeSize * dataCount;
}


HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    printf("HcclScatter called with parameters:\n");
    printf("  sendBuf = %p\n", sendBuf);
    printf("  recvBuf = %p\n", recvBuf);
    printf("  recvCount = %lu\n", recvCount);
    printf("  dataType = %d\n", dataType);
    printf("  root = %u\n", root);
    printf("  comm = %p\n", comm);
    printf("  stream = %p\n", stream);

    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    uint32_t curRank = simCommunicator->GetRankId();
    uint64_t dataSize = CalcDataSize(dataType, recvCount);

    uint32_t mode = SHMManager::GetHcclVmMode();
    if (mode == HcclSim::HcclVmMode::CHECKER) {
        // checker
        HcclSim::HcclVmResult ret = RegisterNpuMemory(curRank, sendBuf, dataSize, 0);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] SetIndependentOpConfig fail\n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
        ret = RegisterNpuMemory(curRank, recvBuf, dataSize, 1);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            printf("[ERROR] [%s] SetIndependentOpConfig fail\n", __func__);
            return HcclResult::HCCL_E_PARA;
        }
    }
    
    using HcclScatterFunc = HcclResult (*)(void *, void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    HcclScatterFunc hcclScatterFunc = reinterpret_cast<HcclScatterFunc>(dlsym(RTLD_NEXT, __func__));
    if (hcclScatterFunc != nullptr) {
        return hcclScatterFunc(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    } else {
        printf("[ERROR] dlsym %s failed\n", __func__);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

#ifdef __cplusplus
}
#endif  // __cplusplus