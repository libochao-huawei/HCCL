/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_FUNC_BLOCK_H
#define CCU_REPRESENTATION_FUNC_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_arg_v1.h"
#include "ccu_rep_reference_manager_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepFuncBlock : public CcuRepBlock {
public:
    explicit CcuRepFuncBlock(const std::string &label);
    bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    uint16_t InstrCount() override;
 
    void SetFuncManager(CcuRepReferenceManager *funcManager);
 
    void DefineInArg(const Variable &var);
    void DefineOutArg(const Variable &var);
    void DefineInArg(const std::vector<Variable> &varList);
    void DefineOutArg(const std::vector<Variable> &varList);
 
    void     SetCallLayer(uint16_t callLayer);
    uint16_t GetCallLayer() const;
 
private:
    CcuRepReferenceManager *funcManager{nullptr};
 
    std::vector<CcuRepArg> inArgs;
    std::vector<CcuRepArg> outArgs;
    uint32_t               inArgCount{0};
    uint32_t               outArgCount{0};
 
    uint16_t callLayer{0};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // _CCU_REPRESENTATION_FUNC_BLOCK_H