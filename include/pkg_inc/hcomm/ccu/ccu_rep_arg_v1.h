/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_ARG_H
#define HCOMM_CCU_REPRESENTATION_ARG_H

#include <vector>
#include <memory>

#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

enum class CcuArgType {
    VARIABLE,
    MEMORY,
    VARIABLE_LIST,
    MEMORY_LIST,
    LOCAL_ADDR,       //1. 新增枚举值
    LOCAL_ADDR_LIST,  // 2. List 类型，以防后面需要 vector<LocalAddr>
    REMOTE_ADDR,
    REMOTE_ADDR_LIST,
};
 
struct CcuRepArg {
    explicit CcuRepArg(const Variable &var) : type(CcuArgType::VARIABLE), var(var)
    {
    }
    explicit CcuRepArg(const Memory &mem) : type(CcuArgType::MEMORY), mem(mem)
    {
    }
    explicit CcuRepArg(const std::vector<Variable> &varList)
        : type(CcuArgType::VARIABLE_LIST), varList(varList)
    {
    }
    explicit CcuRepArg(const std::vector<Memory> &memList)
        : type(CcuArgType::MEMORY_LIST), memList(memList)
    {
    }
    // 新增：LocalAddr 的构造函数
    explicit CcuRepArg(const LocalAddr &addr) 
        : type(CcuArgType::LOCAL_ADDR), localAddr(addr)
    {
    }
    // 新增：LocalAddr 列表的构造函数
    explicit CcuRepArg(const std::vector<LocalAddr> &addrList)
        : type(CcuArgType::LOCAL_ADDR_LIST), localAddrList(addrList)
    {
    }
    explicit CcuRepArg(const RemoteAddr &addr) 
        : type(CcuArgType::REMOTE_ADDR), remoteAddr(addr) 
    {
    }
    explicit CcuRepArg(const std::vector<RemoteAddr> &addrList)
        : type(CcuArgType::REMOTE_ADDR_LIST), remoteAddrList(addrList) 
    {
    }
 
    CcuArgType            type;
    Variable              var;
    Memory                mem;
    std::vector<Variable> varList;
    std::vector<Memory>   memList;
    LocalAddr             localAddr; // 3. 新增成员变量来存储
    std::vector<LocalAddr>  localAddrList; // 新增成员变量
    RemoteAddr              remoteAddr;
    std::vector<RemoteAddr> remoteAddrList;
};

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_ARG_H