/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_CONTEXT_RESOURCE_H
#define HCOMM_CCU_CONTEXT_RESOURCE_H

#include <vector>
#include <array>
#include <unordered_map>

#include "ccu_datatype_v1.h"

namespace hcomm { 

struct CcuRepResource {
    std::array<std::vector<CcuRep::CcuBuf>, CCU_MAX_IODIE_NUM>  ccubufs;
    std::array<std::vector<CcuRep::CcuBuf>, CCU_MAX_IODIE_NUM>  blockCcubufs;
    std::array<std::vector<CcuRep::Executor>, CCU_MAX_IODIE_NUM>   executor;
    std::array<std::vector<CcuRep::Executor>, CCU_MAX_IODIE_NUM>   blockExecutor;
    std::array<std::vector<CcuRep::CompletedEvent>, CCU_MAX_IODIE_NUM> completedEvent;
    std::array<std::vector<CcuRep::CompletedEvent>, CCU_MAX_IODIE_NUM> blockCompletedEvent;
    std::array<std::vector<CcuRep::Address>, CCU_MAX_IODIE_NUM>    address;
    std::array<std::vector<CcuRep::Variable>, CCU_MAX_IODIE_NUM>   continuousVariable;
    std::array<std::vector<CcuRep::Variable>, CCU_MAX_IODIE_NUM>   variable;
    std::array<std::vector<CcuRep::LocalNotify>, CCU_MAX_IODIE_NUM> localNotify;
};

// Context共享资源
struct CcuSharedResource {
    std::unordered_map<std::string, CcuRep::LocalNotify> sharedNotifies;
};

}; // namespace hcomm

#endif // HCOMM_CCU_CONTEXT_RESOURCE_H