/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_CONDITION_H
#define CCU_CONDITION_H

#include "ccu_rep_jump_v1.h"
#include "ccu_rep_jumplabel_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

#define CCU_IF(x) CCU_IF_HELPER1(__COUNTER__, x)

#define CCU_IF_HELPER1(ctr, x) CCU_IF_HELPER2(ctr, x)

#define CCU_IF_HELPER2(ctr, x)                                                                                         \
    for (auto __ccuConditionHidden##ctr = CcuRep::Condition(this, x); __ccuConditionHidden##ctr.Check();               \
         __ccuConditionHidden##ctr.Run())

class Condition {
public:
    Condition(CcuRepContext *context, CcuRelationalOperator<Variable, uint64_t> rel);
    ~Condition();
    bool Check() const;
    void Run();

private:
    CcuRepContext *context{nullptr};
    bool isExecuted{false};

    std::shared_ptr<CcuRepJumpBase>  jump{nullptr};
    std::shared_ptr<CcuRepJumpLabel> endLabel{nullptr};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_CONDITION_H