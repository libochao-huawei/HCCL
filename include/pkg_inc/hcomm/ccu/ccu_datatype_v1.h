/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_DATATYPE_H
#define HCOMM_CCU_DATATYPE_H

#include <memory>

#include "ccu_rep_base_v1.h"
#include "ccu_operator_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

class Variable;
class Address;

class CcuPhyRes {
public:
    CcuPhyRes() = default;
    ~CcuPhyRes() = default;
    void     Reset(uint16_t id);
    void     SetDieId(uint16_t dieId);
    uint16_t Id() const;
    uint16_t DieId() const;

private:
    uint16_t dieId{0};
    uint16_t id{0};
};

class CcuVirRes {
public:
    CcuVirRes(CcuRepContext *context);
    virtual ~CcuVirRes() = default;
    void Reset(uint16_t id);
    void Reset(uint16_t id, uint16_t dieId);
    void SetDieId(uint16_t dieId);
    virtual uint16_t Id() const;
    uint16_t DieId() const;
protected:
    std::shared_ptr<CcuPhyRes> phyRes{nullptr};
    CcuRepContext                *context{nullptr};
};

class Variable : public CcuVirRes {
public:
    explicit Variable(CcuRepContext *context = nullptr);
    Variable(const Variable& other);
    void operator=(Variable&& other);

    void operator=(uint64_t immediate);
    void operator=(const Variable &other);
    void operator=(CcuArithmeticOperator<Variable, Variable> op);

    CcuArithmeticOperator<Variable, Variable> operator+(const Variable &varB) const;
    CcuArithmeticOperator<Variable, Address>  operator+(const Address &addrB) const;

    void operator+=(const Variable &other);

    CcuRelationalOperator<Variable, uint64_t> operator!=(uint64_t immediate) const;
    CcuRelationalOperator<Variable, uint64_t> operator==(uint64_t immediate) const;
};

class Address : public CcuVirRes {
public:
    explicit Address(CcuRepContext *context = nullptr);
    Address(const Address& other);
    void operator=(Address&& other);

    void operator=(uint64_t immediate);
    void operator=(const Address &other);
    void operator=(const Variable &other);

    void operator=(CcuArithmeticOperator<Variable, Address> op);
    void operator=(CcuArithmeticOperator<Address, Variable> op);
    void operator=(CcuArithmeticOperator<Address, Address> op);

    CcuArithmeticOperator<Variable, Address> operator+(const Variable &varB) const;
    CcuArithmeticOperator<Address, Address>  operator+(const Address &addrB) const;

    void operator+=(const Variable &other);
};

class MaskSignal : public CcuVirRes {
public:
    explicit MaskSignal(CcuRepContext *context = nullptr);
};

class CcuBuffer : public CcuVirRes {
public:
    explicit CcuBuffer(CcuRepContext *context = nullptr);
    uint16_t Id() const override;
    static constexpr uint16_t CCUBUFFER_DIE_ID_BIT = 0x8000; // bit15 表示MS所在的IO Die id
};

class CcuBuf : public CcuVirRes {
public:
    explicit CcuBuf(CcuRepContext *context = nullptr);
    uint16_t Id() const override;
    static constexpr uint16_t CCUBUFFER_DIE_ID_BIT = 0x8000; // bit15 表示MS所在的IO Die id
};

class Executor : public CcuVirRes {
public:
    explicit Executor(CcuRepContext *context = nullptr);
};

class Memory {
public:
    Memory() = default;
    Memory(Address addr, Variable token) : addr(addr), token(token)
    {
    }
    Address  addr;
    Variable token;
};

/*------------------将Memory改为LocalAddr与RemoteAddr------------------------*/
class LocalAddr {
public:
    LocalAddr() = default;
    LocalAddr(Address addr, Variable token) : addr(addr), token(token)
    {
    }

    Address  addr;
    Variable token;
};

class RemoteAddr {
public:
    RemoteAddr() = default;
    RemoteAddr(Address addr, Variable token) : addr(addr), token(token)
    {
    }

    Address  addr;
    Variable token;
};

/*------------------将MaskSignal改为LocalNotify------------------------*/
class LocalNotify : public CcuVirRes {
public:
    explicit LocalNotify(CcuRepContext *context = nullptr);
};

class CompletedEvent : public CcuVirRes {
public:
    explicit CompletedEvent(CcuRepContext *context = nullptr);
    void SetMask(uint32_t compeletedMask);
    uint32_t mask{1};
};

};     // namespace CcuRep
};     // namespace hcomm
#endif // HCOMM_CCU_DATATYPE_H