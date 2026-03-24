/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_LOOP_BLOCK_H
#define CCU_LOOP_BLOCK_H

#include "ccu_rep_block_v1.h"
#include "ccu_rep_loopblock_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

class LoopBlock {
public:
    LoopBlock(CcuRepContext *context, std::string label);
    ~LoopBlock();

    template <typename... Arguments> LoopBlock &operator()(const Arguments &...args)
    {
        DefineInArgHelper(args...);
        return *this;
    }

private:
    template <typename First> void DefineInArgHelper(const First &first)
    {
        repLoopBlock->DefineArg(first);
    }

    template <typename First, typename... Rest> void DefineInArgHelper(const First &first, const Rest &...rest)
    {
        repLoopBlock->DefineArg(first);
        DefineInArgHelper(rest...);
    }

    CcuRepContext *context{nullptr};
    std::string    label;

    std::shared_ptr<CcuRepLoopBlock> repLoopBlock{nullptr};
    std::shared_ptr<CcuRepBlock>     curActiveBlock{nullptr};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_LOOP_BLOCK_H