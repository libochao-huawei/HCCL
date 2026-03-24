/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_BASE
#define CCU_REPRESENTATION_BASE

#include <string>

#include "ccu_microcode_v1.h"
#include "ccu_rep_type_v1.h"

namespace hcomm {
namespace CcuRep {

struct TransDep {
    int32_t  logicalId;
    uint16_t dieId;
    uint16_t reserveXnId;
    uint16_t reserveGsaId;
    uint16_t reserveCkeId;
    uint16_t reserveChannalId[2]; //  0: selfLoopBack; 1: inter die, 0xffff为无效值，rep翻译时检查
    uint64_t xnBaseAddr;
    uint64_t ccuResSpaceTokenInfo;
    uint64_t memTokenInfo;
    uint16_t commXn[3]; // 3个Xn
    uint16_t commGsa[2]; // 2个GSA
    uint16_t commSignal; // 1个CKE
    uint16_t loadXnId;
    bool isFuncBlock;
};

class CcuRepBase {
public:
    explicit CcuRepBase();
    virtual ~CcuRepBase();
    virtual bool        Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) = 0;
    virtual std::string Describe()                                     = 0;

    CcuRepType Type() const;
    bool       Translated() const;
    uint16_t StartInstrId() const;
    virtual uint16_t InstrCount();

protected:
    CcuRepType type{CcuRepType::BASE};
    bool       translated{false};
    uint16_t   instrId{0};
    uint16_t   instrCount{0};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_BASE