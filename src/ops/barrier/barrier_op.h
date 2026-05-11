/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_BARRIER_BARRIER_OP
#define OPS_HCCL_SRC_OPS_BARRIER_BARRIER_OP

#include <string>
#include <hccl/hccl_types.h>
#include "hccl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Hierarchical barrier for intra-pod and inter-pod synchronization
 *
 * Implements barrier using AllGather. Each rank writes its rankId as token,
 * AllGather collects all tokens, stream synchronize ensures all ranks reached barrier point.
 *
 * @param intraComm Intra-pod communicator for within-pod synchronization
 * @param interComm Inter-pod communicator for across-pod synchronization
 * @param stream Stream for execution
 * @return HcclResult
 */
extern HcclResult HcclBarrierV2(HcclComm intraComm, HcclComm interComm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {

HcclResult CheckBarrierDPUInputPara(const HcclComm comm, const aclrtStream stream);

}

#endif // OPS_HCCL_SRC_OPS_BARRIER_BARRIER_OP
