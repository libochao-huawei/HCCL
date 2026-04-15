/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_GATHER_OP
#define OPS_HCCL_SRC_OPS_GATHER_OP

#include <string>
#include "hccl.h"
#include "alg_param.h"
#include "alg_type.h"
#include "execute_selector.h"
#include "executor_v2_base.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, uint32_t root,
                      HcclComm comm, aclrtStream stream);
HcclResult HcclGatherGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, uint32_t root,
                               const char* group, aclrtStream stream, const char *tag, void **streams,
                               size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSize);
#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult GatherOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, uint32_t root,
                          HcclComm comm, aclrtStream stream, const std::string &tag);
HcclResult GatherOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, uint32_t root,
                                   HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack);
HcclResult GatherOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, uint32_t root,
                                HcclComm comm, aclrtStream stream, const std::string &tag, OpMode opMode,
                                const ResPackGraphMode &resPack);

HcclResult CheckGatherInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream);

HcclResult GatherInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                              uint32_t root, aclrtStream stream, std::string &opTag);
HcclResult GatherEntryLog(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, uint32_t root,
                          aclrtStream stream, const std::string &tag, const std::string &opName);

}
#endif
