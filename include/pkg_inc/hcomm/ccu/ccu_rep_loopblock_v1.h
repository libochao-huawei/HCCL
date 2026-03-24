/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOP_BLOCK_H
#define HCOMM_CCU_REPRESENTATION_LOOP_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_arg_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepLoopBlock : public CcuRepBlock {
public:
    explicit CcuRepLoopBlock(const std::string &label);
    std::string Describe() override;
 
    void DefineArg(Variable var);
    void DefineArg(Memory mem);
    void DefineArg(LocalAddr addr);
    void DefineArg(RemoteAddr addr);

    void DefineArg(const std::vector<Variable> varList);
    void DefineArg(const std::vector<Memory> memList);
    void DefineArg(const std::vector<LocalAddr> addrList);
    void DefineArg(const std::vector<RemoteAddr> addrList);

    CcuRepArg &GetArg(uint16_t index);
 
private:
    std::vector<CcuRepArg> args;
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOOP_BLOCK_H