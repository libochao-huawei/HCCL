/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_JUMPLABEL_H
#define CCU_REPRESENTATION_JUMPLABEL_H

#include "ccu_rep_block_v1.h"

namespace hcomm {
namespace CcuRep {

class CcuRepJumpLabel : public CcuRepBlock {
public:
    explicit CcuRepJumpLabel(const std::string &label);
    std::string Describe() override;
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_JUMPLABEL_H