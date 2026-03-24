/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_LOOP_CALL_H
#define CCU_LOOP_CALL_H

#include "ccu_rep_loopcall_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

class LoopCall {
public:
    LoopCall(CcuRepContext *context, const std::string &label);
    const std::string &GetLabel() const
    {
        return label;
    }

    template <typename... Arguments> LoopCall &operator()(const Arguments &...args)
    {
        SetArgHelper(args...);
        AppendToContext();
        return *this;
    }

    LoopCall &operator()()
    {
        AppendToContext();
        return *this;
    }

    void AppendToContext();

private:
    template <typename First> void SetArgHelper(const First &first)
    {
        repLoopCall->SetInArg(first);
    }

    template <typename First, typename... Rest> void SetArgHelper(const First &first, const Rest &...rest)
    {
        repLoopCall->SetInArg(first);
        SetArgHelper(rest...);
    }

    CcuRepContext *context{nullptr};
    std::string    label;

    std::shared_ptr<CcuRepLoopCall> repLoopCall{nullptr};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_LOOP_CALL_H