/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_ALL_GATHER_OP
#define OPS_HCCL_SRC_OPS_ALL_GATHER_OP

#include <string>
#include "hccl.h"

#include "alg_param.h"
#include "alg_type.h"
#include "execute_selector.h"
#include "executor_v2_base.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult AllGatherOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                             aclrtStream stream, const std::string &tag);

HcclResult CheckAllGatherInputPara(HcclComm comm, void *sendBuf, void *recvBuf, aclrtStream stream);

}
#endif