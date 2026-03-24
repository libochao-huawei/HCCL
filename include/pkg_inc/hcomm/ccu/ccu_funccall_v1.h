/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_FUNC_CALL_H
#define CCU_FUNC_CALL_H

#include "ccu_rep_funccall_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {

namespace CcuRep {

class FuncCall {
public:
    FuncCall(CcuRepContext *context, std::string label);
    FuncCall(CcuRepContext *context, const Variable &funcAddr);

    template <typename... Arguments> FuncCall &operator()(const Arguments &...args)
    {
        SetArgHelper(args...);
        AppendToContext();
        return *this;
    }

    FuncCall &operator()()
    {
        AppendToContext();
        return *this;
    }

    void AppendToContext();

    template <typename T> void SetInArg(T &&arg)
    {
        repFuncCall->SetInArg(std::forward<T>(arg));
    }

    template <typename T> void SetOutArg(T &&arg)
    {
        repFuncCall->SetOutArg(std::forward<T>(arg));
    }

private:
    template <typename First> void SetArgHelper(const First &first)
    {
        repFuncCall->SetInArg(first);
    }

    template <typename First, typename... Rest> void SetArgHelper(const First &first, const Rest &...rest)
    {
        repFuncCall->SetInArg(first);
        SetArgHelper(rest...);
    }

    CcuRepContext *context{nullptr};
    std::string    label;

    std::shared_ptr<CcuRepFuncCall> repFuncCall{nullptr};
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_FUNC_CALL_H