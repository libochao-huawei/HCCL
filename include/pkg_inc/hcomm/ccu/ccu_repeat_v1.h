/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPEAT_H
#define CCU_REPEAT_H

#include "ccu_rep_jump_v1.h"
#include "ccu_rep_jumplabel_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

#define CCU_WHILE(x)                                                                                                   \
    for (auto __ccuRepeatHidden = CcuRep::Repeat(this, x); __ccuRepeatHidden.Check(); __ccuRepeatHidden.Run())
#define CCU_BREAK __ccuRepeatHidden.Break()

class Repeat {
public:
    Repeat(CcuRepContext *context, CcuRelationalOperator<Variable, uint64_t> rel);
    ~Repeat();
    void Break();
    bool Check() const;
    void Run();

private:
    CcuRepContext *context{nullptr};
    bool isExecuted{false};

    std::shared_ptr<CcuRepJumpBase>  jump{nullptr};
    std::shared_ptr<CcuRepJumpLabel> beginLabel{nullptr};
    std::shared_ptr<CcuRepJumpLabel> endLabel{nullptr};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPEAT_H