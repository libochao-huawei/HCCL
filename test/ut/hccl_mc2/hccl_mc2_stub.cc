/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>

#include "hccl_mc2.h"
#include "param_check.h"
#include "alg_env_config.h"
#include "op_common.h"
#include "coll_alg_v2_exec_registry.h"
#include "execute_selector.h"
#include "executor_v2_base.h"
#include "dtype_common.h"

static DevType g_utDeviceType = DevType::DEV_TYPE_950;

extern "C" void SetMc2UtDeviceType(DevType type)
{
    g_utDeviceType = type;
}

extern "C" void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...)
{
    (void)moduleId;
    (void)level;
    (void)fmt;
}

bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn()
{
    return false;
}

void RptInputErr(std::string errorCode, std::vector<std::string> key, std::vector<std::string> value)
{
    (void)errorCode;
    (void)key;
    (void)value;
}

extern "C" HcclResult hrtGetDeviceType(DevType &devType)
{
    devType = g_utDeviceType;
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    (void)comm;
    if (rankSize == nullptr) {
        return HCCL_E_PTR;
    }
    *rankSize = 2U;
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    (void)comm;
    if (rank == nullptr) {
        return HCCL_E_PTR;
    }
    *rank = 0U;
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclGetCommName(HcclComm comm, char *commName)
{
    (void)comm;
    if (commName == nullptr) {
        return HCCL_E_PTR;
    }
    (void)std::strcpy(commName, "stub_comm");
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclEngineCtxGet(HcclComm comm, const char *engineTag, CommEngine engine, void **ctx, uint64_t *size)
{
    (void)comm;
    (void)engineTag;
    (void)engine;
    (void)ctx;
    (void)size;
    return HCCL_E_NOT_FOUND;
}

extern "C" HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine, uint64_t size, void **ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    if (ctx == nullptr) {
        return HCCL_E_PTR;
    }
    *ctx = std::malloc(size);
    return (*ctx == nullptr) ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

extern "C" HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream,
    uint32_t notifyNum, ThreadHandle *thread)
{
    (void)comm;
    (void)engine;
    (void)stream;
    (void)notifyNum;
    if (thread == nullptr) {
        return HCCL_E_PTR;
    }
    *thread = 0;
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum, const ThreadHandle *threads,
    CommEngine dstCommEngine, ThreadHandle *exportedThreads)
{
    (void)comm;
    (void)threadNum;
    (void)threads;
    (void)dstCommEngine;
    if (exportedThreads != nullptr) {
        *exportedThreads = 0;
    }
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclCreateOpResCtxInner(HcclComm comm, uint8_t opType, HcclDataType srcDataType,
    HcclDataType dstDataType, HcclReduceOp reduceType, uint64_t count, char *algConfig, uint32_t commEngine,
    void **opResCtx)
{
    (void)comm;
    (void)opType;
    (void)srcDataType;
    (void)dstDataType;
    (void)reduceType;
    (void)count;
    (void)algConfig;
    (void)commEngine;
    (void)opResCtx;
    return HCCL_SUCCESS;
}

extern "C" aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    (void)kind;
    if (dst == nullptr || src == nullptr || destMax < count) {
        return ACL_ERROR_INVALID_PARAM;
    }
    (void)std::memcpy(dst, src, count);
    return ACL_ERROR_NONE;
}

extern "C" int sprintf_s(char *strDest, size_t destMax, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    int ret = std::vsnprintf(strDest, destMax, format, ap);
    va_end(ap);
    return ret;
}

extern "C" int strcpy_s(char *strDest, size_t destMax, const char *strSrc)
{
    if (strDest == nullptr || strSrc == nullptr || destMax == 0U) {
        return -1;
    }
    size_t len = std::strlen(strSrc);
    if (len + 1U > destMax) {
        return -1;
    }
    (void)std::memcpy(strDest, strSrc, len + 1U);
    return 0;
}

namespace ops_hccl {

InsCollAlgBase::InsCollAlgBase() = default;
InsCollAlgBase::~InsCollAlgBase() = default;

std::string InsCollAlgBase::Describe() const
{
    return "ut_fake";
}

HcclResult InsCollAlgBase::FastLaunch(const OpParam &param, const CcuFastLaunchCtx *resCtx)
{
    (void)param;
    (void)resCtx;
    return HCCL_SUCCESS;
}

class FakeInsCollAlgBase : public InsCollAlgBase {
public:
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
        AlgHierarchyInfoForAllLevel &algHierarchyInfo) override
    {
        (void)comm;
        (void)topoInfo;
        (void)algHierarchyInfo;
        return HCCL_SUCCESS;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override
    {
        (void)comm;
        (void)param;
        (void)topoInfo;
        (void)algHierarchyInfo;
        (void)resourceRequest;
        return HCCL_SUCCESS;
    }

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override
    {
        (void)param;
        (void)resCtx;
        return HCCL_SUCCESS;
    }
};

HcclResult InitEnvConfig()
{
    return HCCL_SUCCESS;
}

const bool &GetExternalInputHcclEnableEntryLog()
{
    static const bool enabled = false;
    return enabled;
}

const bool &GetExternalInputHcclAivOnlyMode()
{
    static const bool enabled = false;
    return enabled;
}

HcclResult HcomCheckDataType(const HcclDataType dataType)
{
    (void)dataType;
    return HCCL_SUCCESS;
}

HcclResult HcomCheckReductionOp(const HcclReduceOp op)
{
    (void)op;
    return HCCL_SUCCESS;
}

HcclResult HcomCheckUserRank(const u32 totalRanks, const u32 userRank)
{
    if (userRank >= totalRanks) {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCheckTag(const char *tag)
{
    if (tag == nullptr || std::strlen(tag) == 0U) {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    (void)dataType;
    (void)needReduce;
    return HCCL_SUCCESS;
}

HcclResult LogHcclExit(const std::string &opName, const char *tag, HcclUs startut)
{
    (void)opName;
    (void)tag;
    (void)startut;
    return HCCL_SUCCESS;
}

HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam &param)
{
    (void)comm;
    (void)param;
    return HCCL_SUCCESS;
}

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    return HCCL_SUCCESS;
}

HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails *topoInfo)
{
    (void)opType;
    (void)topoInfo;
    return HCCL_SUCCESS;
}

HcclResult SetCommEngine(OpParam &param)
{
    param.engine = CommEngine::COMM_ENGINE_AICPU;
    return HCCL_SUCCESS;
}

HcclResult LoadAICPUKernel(void)
{
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName)
{
    (void)algName;
    (void)std::strncpy(param.commName, "stub_comm", sizeof(param.commName) - 1U);
    return HCCL_SUCCESS;
}

HcclResult SingleRankProc(const OpParam &param)
{
    (void)param;
    return HCCL_SUCCESS;
}

HcclResult HcclGetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<InsCollAlgBase> &executor,
    TopoInfoWithNetLayerDetails *topoInfo, std::unique_ptr<AlgResourceCtxSerializable> &resCtxHost,
    void **resCtxSequence, bool &isResourceReused)
{
    (void)comm;
    (void)param;
    (void)executor;
    (void)topoInfo;
    (void)resCtxHost;
    if (resCtxSequence != nullptr) {
        *resCtxSequence = reinterpret_cast<void *>(0x1);
    }
    isResourceReused = false;
    return HCCL_SUCCESS;
}

ExecuteSelector::ExecuteSelector() {}

HcclResult ExecuteSelector::Run(OpParam &opParam, TopoInfoWithNetLayerDetails *topoInfo,
    std::string &selectAlgName) const
{
    (void)opParam;
    (void)topoInfo;
    selectAlgName = "stub_alg";
    return HCCL_SUCCESS;
}

CollAlgExecRegistryV2 &CollAlgExecRegistryV2::Instance()
{
    static CollAlgExecRegistryV2 instance;
    return instance;
}

std::unique_ptr<InsCollAlgBase> CollAlgExecRegistryV2::GetAlgExec(const HcclCMDType type, const std::string &tag)
{
    (void)type;
    (void)tag;
    return std::unique_ptr<InsCollAlgBase>(new FakeInsCollAlgBase());
}
}
