/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOP_CALL_H
#define HCOMM_CCU_REPRESENTATION_LOOP_CALL_H

#include "ccu_rep_base_v1.h"
#include "ccu_rep_loopblock_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepLoopCall : public CcuRepBase {
public:
    explicit CcuRepLoopCall(const std::string &label);
    bool               Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string        Describe() override;
    uint16_t InstrCount() override;
    const std::string &GetLabel() const;
 
    void Reference(std::shared_ptr<CcuRepLoopBlock> refRep);
 
    void SetInArg(const Variable &var);
    void SetInArg(const std::vector<Variable> &varList);
    void SetInArg(const Memory &mem);
    void SetInArg(const std::vector<Memory> &memList);

    void SetInArg(const LocalAddr &addr);
    void SetInArg(const RemoteAddr &addr);
    void SetInArg(const std::vector<LocalAddr> &addrList);
    void SetInArg(const std::vector<RemoteAddr> &addrList);

private:
    std::string                      label;
    std::shared_ptr<CcuRepLoopBlock> loopBlock{nullptr};
 
    std::vector<CcuRepArg> inArgs;
    uint32_t               inArgCount{0};
    uint32_t               inArgInstrCount{0};  // 处理LoopCall的入参需要的指令数
 
    CcuInstr *instr{nullptr};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOOP_CALL_H