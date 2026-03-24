/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_TYPE_H
#define CCU_REPRESENTATION_TYPE_H

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

    LOC_RECORD_EVENT,
    LOC_WAIT_EVENT,
    LOC_WAIT_NOTIFY,
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

    RECORD_SHARED_NOTIFY,
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

#endif // _CCU_REPRESENTATION_TYPE_H