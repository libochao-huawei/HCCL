/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_INTERFACE_H
#define CCU_INTERFACE_H

#include <memory>

#include "ccu_rep_context_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

void AppendToContext(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBase> rep);
std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock(CcuRepContext* context);
void SetCurrentBlock(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBlock> repBlock);
Variable CreateVariable(CcuRepContext* context);

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_INTERFACE_H