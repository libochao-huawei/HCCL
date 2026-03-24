/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_JUMP_H
#define CCU_REPRESENTATION_JUMP_H

#include <memory>

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_jumplabel_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepJumpBase : public CcuRepBase {
public:
    explicit CcuRepJumpBase(const std::string &label, const Variable &targetInstrId);
    void                Reference(std::shared_ptr<CcuRepJumpLabel> refRep);

protected:
    std::string                      label;
    std::shared_ptr<CcuRepJumpLabel> jumpLabel{nullptr};
    Variable                         targetInstrId;
    CcuInstr                        *instr{nullptr};
};

class CcuRepJump : public CcuRepJumpBase {
public:
    explicit CcuRepJump(const std::string &label, const Variable &targetInstrId);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
};

class CcuRepJumpNE : public CcuRepJumpBase {
public:
    CcuRepJumpNE(const std::string &label, const Variable &targetInstrId, const Variable &condition, uint64_t expected);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

private:
    Variable condition;
    uint64_t expected{0};
};

class CcuRepJumpEQ : public CcuRepJumpBase {
public:
    CcuRepJumpEQ(const std::string &label, const Variable &targetInstrId, const Variable &condition, uint64_t expected);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;

private:
    Variable condition;
    uint64_t expected{0};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_JUMP_H