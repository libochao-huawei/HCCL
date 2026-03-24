/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation datatype define file
 * Author: sunzhepeng
 * Create: 2024-07-06
 */

#ifndef CCU_OPERATOR
#define CCU_OPERATOR

#include <stdexcept>

namespace hcomm {
namespace CcuRep {

template <typename lhsT, typename rhsT> class CcuOperator {
public:
    CcuOperator(lhsT lhs, rhsT rhs) : lhs(lhs), rhs(rhs)
    {
    }
    lhsT lhs;
    rhsT rhs;
};

enum class CcuArithmeticOperatorType { ADDITION, INVALID };

template <typename lhsT, typename rhsT> class CcuArithmeticOperator : public CcuOperator<lhsT, rhsT> {
public:
    CcuArithmeticOperator(lhsT lhs, rhsT rhs, CcuArithmeticOperatorType type)
        : CcuOperator<lhsT, rhsT>(lhs, rhs), type(type)
    {
        Check();
    }
    void Check() const
    {
        // Hccl::THROW<Hccl::CcuApiException>("Invalid Arithmetic Operator");
        throw std::runtime_error("Invalid Arithmetic Operator");
    }

    CcuArithmeticOperatorType type{CcuArithmeticOperatorType::INVALID};
};

enum class CcuRelationalOperatorType { EQUAL, NOT_EQUAL, INVALID };

template <typename lhsT, typename rhsT> class CcuRelationalOperator : public CcuOperator<lhsT, rhsT> {
public:
    CcuRelationalOperator(lhsT lhs, rhsT rhs, CcuRelationalOperatorType type)
        : CcuOperator<lhsT, rhsT>(lhs, rhs), type(type)
    {
        Check();
    }
    void Check() const
    {
        // Hccl::THROW<Hccl::CcuApiException>("Invalid Relational Operator");
        throw std::runtime_error("Invalid Relational Operator");
    }

    CcuRelationalOperatorType type{CcuRelationalOperatorType::INVALID};
};

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_OPERATOR