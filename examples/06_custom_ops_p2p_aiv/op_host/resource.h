/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_P2P_AIV_RESOURCE_H
#define OPS_HCCL_P2P_AIV_RESOURCE_H

#include "common.h"

namespace ops_hccl_p2p_aiv {

HcclResult PrepareP2pAivResource(
    HcclComm comm, uint32_t peerRank, const char *opName, aclrtStream stream, P2pAivResource *resource);
HcclResult ExecuteSend(const P2pAivResource &resource, const void *sendBuf, uint64_t lenBytes, HcclDataType dataType,
    aclrtStream stream);
HcclResult ExecuteRecv(const P2pAivResource &resource, void *recvBuf, uint64_t lenBytes, HcclDataType dataType,
    aclrtStream stream);

} // namespace ops_hccl_p2p_aiv

#endif // OPS_HCCL_P2P_AIV_RESOURCE_H
