/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context header file
 * Create: 2025-02-18
 */

#ifndef HCCL_CCU_INSTR_INFO_H
#define HCCL_CCU_INSTR_INFO_H

#include <vector>
#include "ccu_microcode.h"

namespace hcomm {
namespace CcuRep {

struct CcuInstrInfo {
    std::vector<CcuInstr> instrVec;
    uint16_t              startInstrId{0};
    uint16_t              instrCount{0};
    uint16_t              missionStartInstrId{0};
    uint16_t              missionInstrCount{0};
};

}; // namespace CcuRep
}; // namespace hcomm

#endif // HCCL_CCU_INSTR_INFO_H