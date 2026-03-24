/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_BLOCK_H
#define CCU_REPRESENTATION_BLOCK_H

#include <vector>
#include <memory>

#include "ccu_rep_base_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepBlock : public CcuRepBase {
public:
    explicit CcuRepBlock(const std::string &label = "");
    bool                Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
    uint16_t InstrCount() override;
    const std::string &GetLabel() const;
    std::vector<std::shared_ptr<CcuRepBase>> &GetReps();
    void                                      Append(std::shared_ptr<CcuRepBase> rep);
    std::shared_ptr<CcuRepBase> GetRepByInstrId(uint16_t instrId);

private:
    std::string label;
    std::vector<std::shared_ptr<CcuRepBase>> repVec;
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_BLOCK_H