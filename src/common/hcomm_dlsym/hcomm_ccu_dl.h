/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

#ifndef HCOMM_CCU_DL_H
#define HCOMM_CCU_DL_H

#include "dlsym_common.h"
#include "hccl_inner.h"   // 原始头文件，包含所有类型和声明

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    CCU_SUCCESS = 0,               /**< success */
    CCU_E_PARA = 1,                /**< parameter error */
    CCU_E_PTR = 2,                 /**< empty pointer */
    CCU_E_MEMORY = 3,              /**< memory error */
    CCU_E_INTERNAL = 4,            /**< internal error */
    CCU_E_NOT_SUPPORT = 5,         /**< not support feature */
    CCU_E_NOT_FOUND = 6,           /**< not found specific resource */
    CCU_E_UNAVAIL = 7,             /**< resource unavailable */
    CCU_E_SYSCALL = 8,             /**< call system interface error */
    CCU_E_TIMEOUT = 9,             /**< timeout */
    CCU_E_OPEN_FILE_FAILURE = 10,  /**< open file fail */
    CCU_E_TCP_CONNECT = 11,        /**< tcp connect fail */
    CCU_E_ROCE_CONNECT = 12,       /**< roce connect fail */
    CCU_E_TCP_TRANSFER = 13,       /**< tcp transfer fail */
    CCU_E_ROCE_TRANSFER = 14,      /**< roce transfer fail */
    CCU_E_RUNTIME = 15,            /**< call runtime api fail */
    CCU_E_DRV = 16,                /**< call driver api fail */
    CCU_E_PROFILING = 17,          /**< call profiling api fail */
    CCU_E_CCE = 18,                /**< call cce api fail */
    CCU_E_NETWORK = 19,            /**< call network api fail */
    CCU_E_AGAIN = 20,              /**< try again */
    CCU_E_REMOTE = 21,             /**< error cqe */
    CCU_E_SUSPENDING = 22,         /**< error communicator suspending */
    CCU_E_OPRETRY_FAIL = 23,       /**< retry constraint */
    CCU_E_OOM = 24,                /**< out of memory */
    CCU_E_IN_STATUS = 1041,        /**< The error information is in the status. */

    /*
     * 以下错误码采用显式分段赋值，避免隐式自增导致的 ABI 漂移：
     *   - 1042..1099  driver / 其它通用错误，预留 ~58 个槽位
     *   - 1100..1199  资源不可用专属网段，CCU_CHK_RES_UNAVAIL 依赖
     *                 (ccuRet > CCU_E_RES_UNAVAIL_START && ccuRet < CCU_E_RES_UNAVAIL_END)
     *                 该谓词；新资源类型必须在 1101..1198 内追加，
     *                 严禁在该范围插入非"资源不可用"语义的错误码。
     *   - 1200..      其它分类错误
     * 任何新增错误码必须显式赋值，不得依赖隐式自增。
     */
    CCU_E_DRV_INIT_FAILED = 1042,
    CCU_E_DRV_BUSY        = 1043,

    /* === 资源不足类错误（必须落在 (START, END) 区间内）=== */
    CCU_E_RES_UNAVAIL_START = 1100,

    CCU_E_CHANNEL_CTX_UNAVAIL = 1101,
    CCU_E_JETTY_CTX_UNAVAIL   = 1102,
    CCU_E_WQEBB_UNAVAIL       = 1103,
    CCU_E_MS_UNAVAIL          = 1104,
    CCU_E_LOOP_UNAVAIL        = 1105,
    CCU_E_CKE_UNAVAIL         = 1106,
    CCU_E_XN_UNAVAIL          = 1107,
    CCU_E_GSA_UNAVAIL         = 1108,
    /* 新资源类型在此追加，下一可用值为 1109，严禁超过 1198 */

    CCU_E_RES_UNAVAIL_END = 1199,

    /* === 其它分类错误 === */
    CCU_E_TRANSLATE_FAILED    = 1200,
    CCU_E_ALREADY_BOUND       = 1201,
    CCU_E_LOOP_BODY_UNDEFINED = 1202,

    CCU_E_RESERVED = 0x7FFFFFFF    /**< reserved，固定哨兵值 */
} CcuResult;

typedef enum {
    CCU_CONDITION_EQ = 0,
    CCU_CONDITION_NE = 1,
} CcuConditionType;

typedef uint64_t CcuLoop;
typedef uint64_t CcuLoopGroup;
typedef uint64_t CcuLoopExecutors;

typedef struct {
    uint64_t addrOffset;
    uint64_t loopIterNum;
} CcuLoopConfig;

typedef struct {
    uint64_t addrOffset;
    uint64_t bufferOffset;
    uint64_t eventOffset;
    uint64_t repeatNum;
    uint64_t repeatLoopIdx;
} CcuLoopGroupConfig;


typedef uint64_t CcuInsHandle;

typedef uint64_t CcuKernelHandle;

typedef uint64_t CcuVariableHandle;

typedef uint64_t CcuAddressHandle;

typedef uint64_t CcuEventHandle;

typedef uint64_t CcuBufferHandle;

typedef uint64_t CcuLocalAddrHandle;

typedef uint64_t CcuRemoteAddrHandle;

typedef uint64_t ThreadHandle;

typedef uint64_t ChannelHandle;

typedef int32_t HcommResult;

typedef void *CcuKernelArg;

typedef CcuResult (*CcuKernelFunc)(CcuKernelArg arg);

DECL_WEAK_FUNC(HcclResult, HcclCommQueryCcuIns, HcclComm comm, CcuInsHandle *insHandles, uint32_t *insNum);
DECL_WEAK_FUNC(CcuResult, HcommCcuKernelRegisterStart, CcuInsHandle insHandle);
DECL_WEAK_FUNC(CcuResult, HcommCcuKernelRegister, CcuInsHandle insHandle, const char *kernelFuncName, const void *kernelFunc, const void *kernelArg, CcuKernelHandle *kernelHandle);
DECL_WEAK_FUNC(CcuResult, HcommCcuKernelRegisterEnd, CcuInsHandle insHandle);
DECL_WEAK_FUNC(CcuResult, HcommCcuKernelLaunch, ThreadHandle threadHandle, CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argSize);
DECL_WEAK_FUNC(CcuResult, CcuVariableAlloc, CcuVariableHandle *varHandle);
DECL_WEAK_FUNC(CcuResult, CcuAddressAlloc, CcuAddressHandle *addrHandle);
DECL_WEAK_FUNC(CcuResult, CcuEventAlloc, CcuEventHandle *eventHandle);
DECL_WEAK_FUNC(CcuResult, CcuBufferAlloc, CcuBufferHandle *bufHandle);
DECL_WEAK_FUNC(CcuResult, CcuLocalAddrAlloc, CcuLocalAddrHandle *localAddrHandle, CcuAddressHandle *addrHandle, CcuVariableHandle *tokenHandle);
DECL_WEAK_FUNC(CcuResult, CcuRemoteAddrAlloc, CcuRemoteAddrHandle *remoteAddrHandle, CcuAddressHandle *addrHandle, CcuVariableHandle *tokenHandle);
DECL_WEAK_FUNC(CcuResult, CcuBlockVariableAlloc, CcuVariableHandle *varHandles, uint32_t count);
DECL_WEAK_FUNC(CcuResult, CcuBlockEventAlloc, CcuEventHandle *eventHandles, uint32_t count);
DECL_WEAK_FUNC(CcuResult, CcuBlockBufferAlloc, CcuBufferHandle *bufHandles, uint32_t count);
DECL_WEAK_FUNC(CcuResult, CcuVariableCreateByChannel, ChannelHandle channel, uint32_t varIndex, CcuVariableHandle *varHandle);
DECL_WEAK_FUNC(CcuResult, CcuVariableAssignImm, CcuVariableHandle resVar, uint64_t immediate);
DECL_WEAK_FUNC(CcuResult, CcuVariableAssignVar, CcuVariableHandle dstVarHandle, CcuVariableHandle srcVarHandle);
DECL_WEAK_FUNC(CcuResult, CcuVariableAddVarToVar, CcuVariableHandle resVar, CcuVariableHandle varA, CcuVariableHandle varB);
DECL_WEAK_FUNC(CcuResult, CcuAddressAssignImm, CcuAddressHandle addr, uint64_t immediate);
DECL_WEAK_FUNC(CcuResult, CcuAddressAssignAddr, CcuAddressHandle dstAddrHandle, CcuAddressHandle srcAddrHandle);
DECL_WEAK_FUNC(CcuResult, CcuAddressAssignVar, CcuAddressHandle addr, CcuVariableHandle var);
DECL_WEAK_FUNC(CcuResult, CcuAddressAddVarToAddr, CcuAddressHandle resAddr, CcuAddressHandle lhsAddr, CcuVariableHandle rhsVar);
DECL_WEAK_FUNC(CcuResult, CcuAddressAddAddrToAddr, CcuAddressHandle resAddr, CcuAddressHandle addrA, CcuAddressHandle addrB);
DECL_WEAK_FUNC(CcuResult, CcuAddressAddAssignVar, CcuAddressHandle addr, CcuVariableHandle var);
DECL_WEAK_FUNC(CcuResult, CcuLoadArg, CcuVariableHandle varHandle, uint32_t argId);
DECL_WEAK_FUNC(CcuResult, CcuLoadVar, uint64_t addr, CcuVariableHandle varHandle, uint32_t num);
DECL_WEAK_FUNC(CcuResult, CcuLoadVarFromVarAddr, CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num);
DECL_WEAK_FUNC(CcuResult, CcuStoreVar, uint64_t addr, CcuVariableHandle varHandle, uint32_t num);
DECL_WEAK_FUNC(CcuResult, CcuStoreVarToVarAddr, CcuVariableHandle addrHandle, CcuVariableHandle varHandle, uint32_t num);
DECL_WEAK_FUNC(CcuResult, CcuEventRecord, CcuEventHandle eventHandle, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuEventWait, CcuEventHandle eventHandle, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuNotifyRecord, ChannelHandle channel, uint32_t remoteNotifyIdx, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuNotifyWait, ChannelHandle channel, uint32_t localNotifyIdx, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuWriteVariableWithNotify, ChannelHandle channel, CcuVariableHandle varHandle,uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalNotifyRecord, const char *notifyTag, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalNotifyWait, const char *notifyTag, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalCopyMemToMem, CcuLocalAddrHandle dst, CcuLocalAddrHandle src, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalCopyMemToBuffer, CcuBufferHandle dst, CcuLocalAddrHandle src,CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalCopyBufferToMem, CcuLocalAddrHandle dst, CcuBufferHandle src, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalMemReduce, CcuLocalAddrHandle dst, CcuLocalAddrHandle src, CcuVariableHandle len, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuLocalBufferReduce, CcuBufferHandle* buffers, uint32_t count, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuReadMemToMem, ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuReadMemToBuffer, ChannelHandle channel, CcuBufferHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuReadMemToMemReduce, ChannelHandle channel, CcuLocalAddrHandle localHandle, CcuRemoteAddrHandle remoteHandle, CcuVariableHandle len, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuWriteMemToMem, ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuWriteBufferToMem, ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuBufferHandle localHandle, CcuVariableHandle len, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuWriteMemToMemReduce, ChannelHandle channel, CcuRemoteAddrHandle remoteHandle, CcuLocalAddrHandle localHandle, CcuVariableHandle len, HcclDataType dataType, HcclReduceOp opType, CcuEventHandle event, uint16_t mask);
DECL_WEAK_FUNC(CcuResult, CcuIfBegin, CcuVariableHandle var, uint64_t immediate, CcuConditionType condType, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuIfElse, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuIfEnd, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuFlushPendingIfs);
DECL_WEAK_FUNC(CcuResult, CcuWhileBegin, CcuVariableHandle var, uint64_t immediate, CcuConditionType condType, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuWhileEnd, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuDoWhileBegin, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuDoWhileEnd, CcuVariableHandle var, uint64_t immediate, CcuConditionType condType, const char *label);
DECL_WEAK_FUNC(CcuResult, CcuFuncBlockLookup, const void *funcPtr, uint64_t *outHandle);
DECL_WEAK_FUNC(CcuResult, CcuFuncBlockBegin, const void *funcPtr, uint64_t *outHandle);
DECL_WEAK_FUNC(CcuResult, CcuFuncBlockEnd, uint64_t handle);
DECL_WEAK_FUNC(CcuResult, CcuFuncDefineInArg, uint64_t handle, CcuVariableHandle formal);
DECL_WEAK_FUNC(CcuResult, CcuFuncCall, uint64_t handle, const CcuVariableHandle *inArgs, uint32_t numIn);
DECL_WEAK_FUNC(void, _CcuIfStackPush, const char *label);
DECL_WEAK_FUNC(void, _CcuIfStackMarkBodyDone);
DECL_WEAK_FUNC(const char *, _CcuIfStackPopForElse);
DECL_WEAK_FUNC(void, _CcuDoWhileStackPush, const char *label);
DECL_WEAK_FUNC(const char *, _CcuDoWhileStackPopForWhile);
DECL_WEAK_FUNC(CcuResult, CcuLoopCreate, CcuLoop *loop);
DECL_WEAK_FUNC(CcuResult, _CcuLoopBodyEnter, CcuLoop loop);
DECL_WEAK_FUNC(CcuResult, _CcuLoopBodyExit, CcuLoop loop);
DECL_WEAK_FUNC(CcuResult, CcuLoopGroupCreate, CcuLoopGroup *group, uint32_t maxLoopNum, const CcuLoopGroupConfig *config);
DECL_WEAK_FUNC(CcuResult, CcuLoopGroupCreateFromVar, CcuLoopGroup *group, uint32_t maxLoopNum, CcuVariableHandle parallelVar, CcuVariableHandle offsetVar);
DECL_WEAK_FUNC(CcuResult, CcuLoopGroupAddLoop, CcuLoopGroup group, CcuLoop loop, const CcuLoopConfig *config);
DECL_WEAK_FUNC(CcuResult, CcuLoopGroupAddLoopFromVar, CcuLoopGroup group, CcuLoop loop, CcuVariableHandle loopParamVar);
DECL_WEAK_FUNC(HcommResult, HcommCcuGetMemToken, uint64_t srcVa, uint64_t size, uint64_t *tokenInfo);

void HcclCcuDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_CCU_DL_H