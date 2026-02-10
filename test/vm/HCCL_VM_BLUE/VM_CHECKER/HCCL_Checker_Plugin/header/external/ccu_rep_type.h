/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation type header file
 * Create: 2025-02-18
 */

#ifndef HCCL_CCU_REPRESENTATION_TYPE_H
#define HCCL_CCU_REPRESENTATION_TYPE_H

namespace hcomm {
namespace CcuRep {

enum class CcuRepType {
    BASE,
    BLOCK,
    NOP,

    LOAD,
    STORE,
    LOAD_ARG,
    LOAD_VAR,

    ASSIGN,
    ADD,
    SET_LOOP,

    JUMP,
    JUMP_NE,
    JUMP_EQ,
    JUMP_LABEL,

    FUNC_CALL,
    FUNC_BLOCK,

    LOOP_CALL,
    LOOP,
    LOOPGROUP,
    LOOP_BLOCK,
    LOOPGROUP_BLOCK,

    LOC_POST_SEM,
    LOC_WAIT_SEM,
    REM_POST_SEM,
    REM_WAIT_SEM,
    REM_POST_VAR,
    REM_WAIT_GROUP,

    READ,
    WRITE,
    LOCAL_CPY,
    LOCAL_REDUCE,
    REM_MEM,

    BUF_READ,
    BUF_WRITE,
    BUF_LOC_READ,
    BUF_LOC_WRITE,
    BUF_REDUCE,

    POST_SHARED_VAR,
    POST_SHARED_SEM,
};

enum class AssignSubType { INVALID, IMD_TO_VARIABLE, IMD_TO_ADDR, VAR_TO_ADDR, ADDR_TO_ADDR, VAR_TO_VAR };

enum class AddSubType {
    INVALID,
    ADDR_PLUS_VAR_TO_ADDR,
    ADDR_PLUS_ADDR_TO_ADDR,
    VAR_PLUS_VAR_TO_VAR,
    SELF_ADD_ADDRESS,
    SELF_ADD_VARIABLE
};

}; // namespace CcuRep
}; // namespace hcomm

#endif // HCCL_CCU_REPRESENTATION_TYPE_H