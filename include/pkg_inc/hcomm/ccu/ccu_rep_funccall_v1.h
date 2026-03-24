/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_FUNC_CALL_H
#define CCU_REPRESENTATION_FUNC_CALL_H

#include "ccu_rep_base_v1.h"
#include "ccu_rep_funcblock_v1.h"
#include "ccu_rep_reference_manager_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepFuncCall : public CcuRepBase {
public:
    explicit CcuRepFuncCall(const std::string &label);
    explicit CcuRepFuncCall(const Variable &funcAddrVar);
    bool               Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string        Describe() override;
    uint16_t InstrCount() override;
    const std::string &GetLabel() const;

    void Reference(std::shared_ptr<CcuRepFuncBlock> refRep);
    void SetFuncManager(CcuRepReferenceManager *funcManager);

    void SetInArg(const Variable &var);
    void SetOutArg(const Variable &var);
    void SetInArg(const std::vector<Variable> &varList);
    void SetOutArg(const std::vector<Variable> &varList);

    int32_t GetCallLayer();

private:
    CcuRepReferenceManager *funcManager{nullptr};

    std::string                      label;
    std::shared_ptr<CcuRepFuncBlock> funcBlock{nullptr};
    Variable                         funcAddrVar;

    std::vector<CcuRepArg> inArgs;
    std::vector<CcuRepArg> outArgs;
    uint32_t               inArgCount{0};
    uint32_t               outArgCount{0};

    CcuInstr *instr{nullptr};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // _CCU_REPRESENTATION_FUNC_CALL_H