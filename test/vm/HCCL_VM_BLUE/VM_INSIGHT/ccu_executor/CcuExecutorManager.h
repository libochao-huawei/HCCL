/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor create func mgr
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_EXECUTOR_MANAGER_H
#define HCCL_SIM_CCU_EXECUTOR_MANAGER_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "FakeStreamMgr.h"
#include "CcuExecutorBase.h"
#include "SimulatorBase.h"
#include "ccuMicrocodeV1.h"

using CcuExecutorCreateFunc = std::function<std::unique_ptr<CcuExecutorBase>(
    int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)>;

// ccu executor实例创建接口的管理类
class CcuExecutorCreateFuncMgr {
public:
    static CcuExecutorCreateFuncMgr& Instance();
    CcuExecutorCreateFuncMgr(const CcuExecutorCreateFuncMgr&) = delete;
    CcuExecutorCreateFuncMgr& operator=(const CcuExecutorCreateFuncMgr&) = delete;

    void RegFunc(uint16_t instrType, const CcuExecutorCreateFunc& func) {
        container[instrType] = func;
    }

    const CcuExecutorCreateFunc GetFunc(uint16_t instrType) {
        auto res = container.find(instrType);
        if (res == container.end()) {
            return nullptr;
        }
        return res->second;
    }

private:
    CcuExecutorCreateFuncMgr() = default;
    ~CcuExecutorCreateFuncMgr() = default;

private:
    std::map<uint16_t, CcuExecutorCreateFunc> container{};
};

// 根据指令类型注册创建ccuExecutor实例的函数
class CcuExecutorCreateFuncRegister {
public:
    CcuExecutorCreateFuncRegister(uint16_t type, uint16_t code, const CcuExecutorCreateFunc& func) {
        Hccl::CcuRep::CcuInstrHeader ccuInstr = Hccl::CcuRep::InstrHeader(type, code);
        CcuExecutorCreateFuncMgr::Instance().RegFunc(ccuInstr.header, func);
    }
    ~CcuExecutorCreateFuncRegister() = default;
};

// 根据指令类型创建对应的ccuExecutor实例
class CcuExecutorFactory {
public:
    static std::unique_ptr<CcuExecutorBase> MakeCcuExecutorInstance(
        uint16_t instrType, int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator)
    {
        auto createFunc = CcuExecutorCreateFuncMgr::Instance().GetFunc(instrType);
        if (createFunc == nullptr) {
            return nullptr;
        }
        return createFunc(streamId, rankId, dieId, instr, ccuSimulator);
    }
    ~CcuExecutorFactory() = default;
};

#define REG_CCU_EXECUTOR_CREATE_FUNC(type, code, className)                                                        \
    static CcuExecutorCreateFuncRegister g_reg##className(                                             \
        type, code, [](int streamId, int rankId, int dieId, const Hccl::CcuRep::CcuInstr &instr, CcuSimulator *ccuSimulator) { \
            return std::make_unique<className>(streamId, rankId, dieId, instr, ccuSimulator);                              \
        });

#endif // HCCL_SIM_CCU_EXECUTOR_MANAGER_H
