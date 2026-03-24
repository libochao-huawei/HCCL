/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_LOOPGROUP_CALL_H
#define CCU_LOOPGROUP_CALL_H

#include <vector>
#include <string>

#include "ccu_datatype_v1.h"
#include "ccu_loopcall_v1.h"
#include "ccu_rep_context_v1.h"
#include "ccu_rep_loopblock_v1.h"

namespace hcomm {
namespace CcuRep {

class LoopGroupCall {
public:
    explicit LoopGroupCall(CcuRepContext *context, std::string label = "") : context(context), label(label)
    {
    }
    void Run(const std::vector<LoopCall> &loopVec, const std::vector<Variable> &loopCfg,
             const std::vector<Executor> &executors, Variable paraCfgIn, Variable offsetCfgIn) const;

private:
    CcuRepContext *context;
    std::string    label;

    uint64_t paraCfg{0};
    uint64_t offsetCfg{0};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_LOOPGROUP_CALL_H