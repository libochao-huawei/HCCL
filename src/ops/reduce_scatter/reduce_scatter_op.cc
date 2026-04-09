/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_op.h"
#include "op_common_ops.h"
#include "topo_host.h"
#include "scoped_perf_timer.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

namespace {
constexpr const char *HCCL_RS_PERF_STAGE_ENV = "HCCL_RS_PERF_STAGE";
constexpr const char *HCCL_RS_PERF_SLOW_US_ENV = "HCCL_RS_PERF_SLOW_US";
constexpr const char *HCCL_RS_PERF_VERBOSE_ENV = "HCCL_RS_PERF_VERBOSE";
constexpr size_t HCCL_RS_PERF_MAX_STAGE_NUM = 16U;

std::string NormalizePerfEnvValue(const char *envValue)
{
    std::string value = (envValue == nullptr) ? "" : envValue;
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool IsPerfEnvEnabled(const char *envName)
{
    const char *envValue = std::getenv(envName);
    if (envValue == nullptr) {
        return false;
    }
    const std::string normalized = NormalizePerfEnvValue(envValue);
    return normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes";
}

u64 GetPerfEnvU64(const char *envName, u64 defaultValue)
{
    const char *envValue = std::getenv(envName);
    if (envValue == nullptr || envValue[0] == '\0') {
        return defaultValue;
    }
    char *endPtr = nullptr;
    unsigned long long parsedValue = strtoull(envValue, &endPtr, 10);
    if (endPtr == envValue) {
        return defaultValue;
    }
    return static_cast<u64>(parsedValue);
}

class ReduceScatterPerfTrace {
public:
    class ScopedStage {
    public:
        ScopedStage(ReduceScatterPerfTrace *trace, const char *stageName) : trace_(trace)
        {
            if (trace_ != nullptr) {
                trace_->BeginStage(stageName);
            }
        }

        ~ScopedStage()
        {
            if (trace_ != nullptr) {
                trace_->EndStage();
            }
        }

        ScopedStage(const ScopedStage &) = delete;
        ScopedStage &operator=(const ScopedStage &) = delete;

        ScopedStage(ScopedStage &&other) noexcept : trace_(other.trace_)
        {
            other.trace_ = nullptr;
        }

        ScopedStage &operator=(ScopedStage &&other) noexcept
        {
            if (this != &other) {
                if (trace_ != nullptr) {
                    trace_->EndStage();
                }
                trace_ = other.trace_;
                other.trace_ = nullptr;
            }
            return *this;
        }

    private:
        ReduceScatterPerfTrace *trace_ = nullptr;
    };

    explicit ReduceScatterPerfTrace(const char *scopeName, uint64_t recvCount)
        : scopeName_(scopeName == nullptr ? "ReduceScatter" : scopeName),
          recvCount_(recvCount),
          enabled_(IsPerfEnvEnabled(HCCL_RS_PERF_STAGE_ENV)),
          verbose_(IsPerfEnvEnabled(HCCL_RS_PERF_VERBOSE_ENV)),
          slowUs_(GetPerfEnvU64(HCCL_RS_PERF_SLOW_US_ENV, 0))
    {
        if (enabled_) {
            const HcclUs now = TIME_NOW();
            totalStartUs_ = now;
            stageStartUs_ = now;
        }
    }

    HcclResult Record(HcclResult ret)
    {
        if (ret != HcclResult::HCCL_SUCCESS) {
            result_ = ret;
        }
        return ret;
    }

    ScopedStage Scope(const char *stageName)
    {
        return ScopedStage(this, stageName);
    }

    void BeginStage(const char *stageName)
    {
        if (!enabled_) {
            return;
        }
        const HcclUs now = TIME_NOW();
        FlushCurrentStage(now);
        currentStageName_ = stageName;
        stageStartUs_ = now;
    }

    void EndStage()
    {
        if (!enabled_) {
            return;
        }
        const HcclUs now = TIME_NOW();
        FlushCurrentStage(now);
        currentStageName_ = nullptr;
        stageStartUs_ = now;
    }

    void SetTag(const char *tag)
    {
        if (enabled_ && tag != nullptr) {
            tag_ = tag;
        }
    }

    void SetRankInfo(u32 rankSize, u32 userRank)
    {
        if (!enabled_) {
            return;
        }
        rankSize_ = rankSize;
        userRank_ = userRank;
    }

    void SetAlgName(const std::string &algName)
    {
        if (enabled_) {
            algName_ = algName;
        }
    }

    void SetPath(const char *path)
    {
        if (enabled_ && path != nullptr) {
            path_ = path;
        }
    }

    void SetResult(HcclResult ret)
    {
        if (ret != HcclResult::HCCL_SUCCESS || result_ == HcclResult::HCCL_SUCCESS) {
            result_ = ret;
        }
    }

    ~ReduceScatterPerfTrace()
    {
        if (!enabled_) {
            return;
        }

        const HcclUs now = TIME_NOW();
        FlushCurrentStage(now);
        const u64 totalUs = static_cast<u64>(DURATION_US(now - totalStartUs_).count());
        const bool isSlow = (slowUs_ > 0) && (totalUs >= slowUs_);
        if (!(verbose_ || slowUs_ == 0 || isSlow || result_ != HcclResult::HCCL_SUCCESS)) {
            return;
        }

        std::string stageInfo;
        u64 trackedUs = 0;
        for (size_t i = 0; i < stageCount_; ++i) {
            if (!stageInfo.empty()) {
                stageInfo += ", ";
            }
            stageInfo += stageCosts_[i].name;
            stageInfo += "[";
            stageInfo += std::to_string(stageCosts_[i].costUs);
            stageInfo += "]us";
            trackedUs += stageCosts_[i].costUs;
        }
        if (overflowUs_ > 0) {
            if (!stageInfo.empty()) {
                stageInfo += ", ";
            }
            stageInfo += "other[";
            stageInfo += std::to_string(overflowUs_);
            stageInfo += "]us";
            trackedUs += overflowUs_;
        }
        if (totalUs > trackedUs) {
            if (!stageInfo.empty()) {
                stageInfo += ", ";
            }
            stageInfo += "gap[";
            stageInfo += std::to_string(totalUs - trackedUs);
            stageInfo += "]us";
        }

        HCCL_ERROR("[%s][PerfStage] tag[%s], rank[%u/%u], recvCount[%llu], path[%s], alg[%s], total[%llu]us, ret[%d], stages{%s}",
            scopeName_.c_str(), tag_.c_str(), userRank_, rankSize_, static_cast<unsigned long long>(recvCount_),
            path_.c_str(), algName_.c_str(), static_cast<unsigned long long>(totalUs), static_cast<int>(result_),
            stageInfo.empty() ? "none" : stageInfo.c_str());
        std::fprintf(stderr,
            "[PerfStage][%s] tag[%s], rank[%u/%u], recvCount[%llu], path[%s], alg[%s], total[%llu]us, ret[%d], stages{%s}\n",
            scopeName_.c_str(), tag_.c_str(), userRank_, rankSize_, static_cast<unsigned long long>(recvCount_),
            path_.c_str(), algName_.c_str(), static_cast<unsigned long long>(totalUs), static_cast<int>(result_),
            stageInfo.empty() ? "none" : stageInfo.c_str());
        std::fflush(stderr);
    }

private:
    struct StageCost {
        const char *name = nullptr;
        u64 costUs = 0;
    };

    void FlushCurrentStage(HcclUs now)
    {
        if (currentStageName_ == nullptr || now < stageStartUs_) {
            return;
        }
        AppendStageCost(currentStageName_, static_cast<u64>(DURATION_US(now - stageStartUs_).count()));
    }

    void AppendStageCost(const char *stageName, u64 costUs)
    {
        if (stageName == nullptr) {
            return;
        }
        for (size_t i = 0; i < stageCount_; ++i) {
            if (stageCosts_[i].name != nullptr && strcmp(stageCosts_[i].name, stageName) == 0) {
                stageCosts_[i].costUs += costUs;
                return;
            }
        }
        if (stageCount_ < stageCosts_.size()) {
            stageCosts_[stageCount_].name = stageName;
            stageCosts_[stageCount_].costUs = costUs;
            ++stageCount_;
            return;
        }
        overflowUs_ += costUs;
    }

    std::string scopeName_;
    uint64_t recvCount_ = 0;
    bool enabled_ = false;
    bool verbose_ = false;
    u64 slowUs_ = 0;
    HcclUs totalStartUs_{};
    HcclUs stageStartUs_{};
    const char *currentStageName_ = nullptr;
    std::array<StageCost, HCCL_RS_PERF_MAX_STAGE_NUM> stageCosts_{};
    size_t stageCount_ = 0;
    u64 overflowUs_ = 0;
    std::string tag_ = "NA";
    std::string path_ = "default";
    std::string algName_ = "NA";
    u32 rankSize_ = INVALID_VALUE_RANKSIZE;
    u32 userRank_ = INVALID_VALUE_RANKID;
    HcclResult result_ = HcclResult::HCCL_SUCCESS;
};
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclReduceScatter");
    ReduceScatterPerfTrace perfTrace("HcclReduceScatter", recvCount);
    const char *rsPerfStageEnv = std::getenv(HCCL_RS_PERF_STAGE_ENV);
    const char *rsPerfVerboseEnv = std::getenv(HCCL_RS_PERF_VERBOSE_ENV);
    if (rsPerfStageEnv != nullptr || rsPerfVerboseEnv != nullptr) {
        std::fprintf(stderr,
            "[RS_ENTRY] stageEnv=%s verboseEnv=%s recvCount=%llu comm=%p stream=%p\n",
            rsPerfStageEnv == nullptr ? "null" : rsPerfStageEnv,
            rsPerfVerboseEnv == nullptr ? "null" : rsPerfVerboseEnv,
            static_cast<unsigned long long>(recvCount), comm, stream);
        std::fflush(stderr);
    }

    perfTrace.BeginStage("PreCheck");
    if (GetHcommVersion() < 90000000) { // compat handle
        perfTrace.SetPath("compat_inner_fallback");
        HcclResult ret = perfTrace.Record(HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream));
        perfTrace.EndStage();
        return ret;
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(perfTrace.Record(hrtGetDeviceType(deviceType)));
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
#else
    if (deviceType != DevType::DEV_TYPE_910_95) {
#endif
        perfTrace.SetPath("device_inner_fallback");
        HcclResult ret = perfTrace.Record(HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream));
        perfTrace.EndStage();
        return ret;
    }
    perfTrace.EndStage();

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内

    perfTrace.BeginStage("InitEnvConfig");
    CHK_RET(perfTrace.Record(InitEnvConfig()));
    perfTrace.EndStage();

    OpParam param;
    perfTrace.BeginStage("InputCheck");
    if (recvCount == 0) {
        HCCL_WARNING("input recvCount is 0, return reduce scatter success");
        perfTrace.EndStage();
        perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
        return HCCL_SUCCESS;
    }
    CHK_RET(perfTrace.Record(CheckReduceScatterInputPara(comm, sendBuf, recvBuf, stream)));
    CHK_RET(perfTrace.Record(CheckCount(recvCount)));
    CHK_RET(perfTrace.Record(CheckDataType(dataType, true)));
    CHK_RET(perfTrace.Record(CheckReduceOp(dataType, op)));
    perfTrace.EndStage();

    perfTrace.BeginStage("CommInfo");
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(perfTrace.Record(HcclGetRankSize(comm, &rankSize)));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(perfTrace.Record(HcclGetRankId(comm, &userRank)));
    perfTrace.SetRankInfo(rankSize, userRank);
    CHK_RET(perfTrace.Record(HcomCheckUserRank(rankSize, userRank)));
    CHK_RET(perfTrace.Record(HcclGetCommName(comm, param.commName)));
    perfTrace.EndStage();

    perfTrace.BeginStage("BuildTag");
    int ret = sprintf_s(param.tag, sizeof(param.tag), "ReduceScatter_%s", param.commName);
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        perfTrace.Record(HCCL_E_INTERNAL);
        perfTrace.EndStage();
        return HCCL_E_INTERNAL;
    }
    perfTrace.SetTag(param.tag);
    CHK_RET(perfTrace.Record(HcclCheckTag(param.tag)));
    perfTrace.EndStage();

    perfTrace.BeginStage("EntryLog");
    CHK_RET(perfTrace.Record(
        ReduceScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, op, stream, param.tag, "HcclReduceScatter")));
    perfTrace.EndStage();

    perfTrace.BeginStage("ReduceScatterOutPlace");
    CHK_RET(perfTrace.Record(ReduceScatterOutPlace(param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream,
        rankSize)));
    perfTrace.EndStage();

    perfTrace.BeginStage("LogHcclExit");
    CHK_RET(perfTrace.Record(LogHcclExit("HcclReduceScatter", param.tag, startut)));
    perfTrace.EndStage();

    perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
 	     HcclReduceOp op, const char* group, aclrtStream stream, const char* tag, void** streams,
 	     size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclReduceScatterGraphMode");
    ReduceScatterPerfTrace perfTrace("HcclReduceScatterGraphMode", recvCount);

    HcclComm comm = nullptr;
    perfTrace.BeginStage("GetCommHandle");
    HcomGetCommHandleByGroup(group, &comm);
    perfTrace.EndStage();

    HCCL_INFO("[HcclReduceScatterGraphMode] get group name: %s", group);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内

    perfTrace.BeginStage("InitEnvConfig");
    CHK_RET(perfTrace.Record(InitEnvConfig()));
    perfTrace.EndStage();

    perfTrace.BeginStage("InputCheck");
    if (recvCount == 0) {
        HCCL_WARNING("input recvCount is 0, return reduce scatter success");
        perfTrace.EndStage();
        perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
        return HCCL_SUCCESS;
    }
    CHK_RET(perfTrace.Record(CheckReduceScatterInputPara(comm, sendBuf, recvBuf, stream)));
    CHK_RET(perfTrace.Record(CheckCount(recvCount)));
    CHK_RET(perfTrace.Record(CheckDataType(dataType, true)));
    CHK_RET(perfTrace.Record(CheckReduceOp(dataType, op)));
    perfTrace.EndStage();

    perfTrace.BeginStage("CommInfo");
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(perfTrace.Record(HcclGetCommName(comm, commName)));
    const string opTag = "ReduceScatter_" + string(commName);
    perfTrace.SetTag(opTag.c_str());
    CHK_RET(perfTrace.Record(HcclCheckTag(opTag.c_str())));
    CHK_RET(perfTrace.Record(HcclCheckTag(tag)));

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(perfTrace.Record(HcclGetRankSize(comm, &rankSize)));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(perfTrace.Record(HcclGetRankId(comm, &userRank)));
    perfTrace.SetRankInfo(rankSize, userRank);
    CHK_RET_AND_PRINT_IDE(perfTrace.Record(HcomCheckUserRank(rankSize, userRank)), opTag.c_str());
    perfTrace.EndStage();

    perfTrace.BeginStage("BuildResPack");
    ResPackGraphMode resPack;
    strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1);

    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }

    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;
    perfTrace.EndStage();

    perfTrace.BeginStage("EntryLog");
    CHK_RET(perfTrace.Record(ReduceScatterEntryLog(
        sendBuf, recvBuf, recvCount, dataType, op, stream, opTag.c_str(), "HcclReduceScatterGraphMode")));
    perfTrace.EndStage();

    perfTrace.BeginStage("ReduceScatterOutPlaceGraphMode");
    CHK_RET_AND_PRINT_IDE(
        perfTrace.Record(ReduceScatterOutPlaceGraphMode(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, tag,
            resPack)),
        opTag.c_str());
    perfTrace.EndStage();

    perfTrace.BeginStage("LogHcclExit");
    CHK_RET(perfTrace.Record(LogHcclExit("HcclReduceScatterGraphMode", opTag.c_str(), startut)));
    perfTrace.EndStage();

    perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult CheckReduceScatterInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclReduceScatter", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

static HcclResult PrepareReduceScatterParam(OpParam &param, void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream, u32 userRankSize,
 	OpMode opMode)
{
    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 inputSize = outputSize * userRankSize;

    param.stream = stream;
    param.reduceType = op;
    param.opMode = opMode;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = recvCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOutPlace(OpParam &param, void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, u32 userRankSize)
{
    HCCL_INFO("Start to execute ReduceScatterOutPlace");
    ReduceScatterPerfTrace perfTrace("ReduceScatterOutPlace", recvCount);
    perfTrace.SetTag(param.tag);
    perfTrace.SetRankInfo(userRankSize, INVALID_VALUE_RANKID);

    {
        auto scope = perfTrace.Scope("PrepareParam");
        CHK_RET(perfTrace.Record(PrepareReduceScatterParam(param, sendBuf, recvBuf, recvCount, dataType, op, comm,
            stream, userRankSize, OpMode::OPBASE)));
    }

    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    {
        auto scope = perfTrace.Scope("FastLaunchCheck");
        HCCL_SCOPED_PERF_ERR(simpleScopedZone, "ReduceScatter/FastLaunchCheck", HCCL_RS_PERF_STAGE_ENV,
            GetPerfEnvU64(HCCL_RS_PERF_SLOW_US_ENV, 0));
        simpleScopedZone.AppendExtra(std::string("tag[") + param.tag + "]");
        simpleScopedZone.AppendExtra(std::string("recvCount[") + std::to_string(recvCount) + "]");
        if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
            perfTrace.SetPath("ccu_fast_launch");
            {
                auto execScope = perfTrace.Scope("CcuFastLaunchExec");
                HcclResult ret = perfTrace.Record(HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx));
                return ret;
            }
        }
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    {
        auto scope = perfTrace.Scope("Selector");
        CHK_RET(perfTrace.Record(Selector(comm, param, topoInfo, algName)));
        perfTrace.SetAlgName(algName);
    }

    if (ShouldUseInnerOp(param.opExecuteConfig)) {
        perfTrace.SetPath("inner_op_fallback");
        perfTrace.BeginStage("InnerOpFallback");
        HcclResult ret = perfTrace.Record(HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream));
        perfTrace.EndStage();
        return ret;
    }
    if (userRankSize == 1) {
        perfTrace.SetPath("single_rank_proc");
        HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
        perfTrace.BeginStage("SingleRankProc");
        CHK_RET(perfTrace.Record(SingleRankProc(param)));
        perfTrace.EndStage();
        perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
        return HcclResult::HCCL_SUCCESS;
    }

    perfTrace.SetPath("hccl_exec_op");
    {
        auto scope = perfTrace.Scope("HcclExecOp");
        CHK_RET(perfTrace.Record(HcclExecOp(comm, param, topoInfo, algName)));
    }

    HCCL_INFO("Execute ReduceScatterOutPlace success.");
    perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterEntryLog(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    aclrtStream stream, const char *tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], reduceOp[%s], streamId[%d], deviceLogicId[%d]",
            tag, sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
 	HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute ReduceScatterOutPlaceGraphMode");
    ReduceScatterPerfTrace perfTrace("ReduceScatterOutPlaceGraphMode", recvCount);
    perfTrace.SetTag(tag.c_str());

    OpParam param;
    u32 userRankSize;
    perfTrace.BeginStage("GetRankSize");
    CHK_RET(perfTrace.Record(HcclGetRankSize(comm, &userRankSize)));
    perfTrace.SetRankInfo(userRankSize, INVALID_VALUE_RANKID);
    perfTrace.EndStage();

    perfTrace.BeginStage("PrepareParam");
    CHK_RET(perfTrace.Record(PrepareReduceScatterParam(param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream,
        userRankSize, OpMode::OFFLOAD)));
    perfTrace.EndStage();

    if (userRankSize == 1) {
        perfTrace.SetPath("single_rank_proc");
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        perfTrace.BeginStage("SingleRankProc");
        CHK_RET(perfTrace.Record(SingleRankProc(param)));
        perfTrace.EndStage();
        perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
        return HcclResult::HCCL_SUCCESS;
    }

    perfTrace.BeginStage("Selector");
    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(perfTrace.Record(Selector(comm, param, topoInfo, algName)));
    perfTrace.SetAlgName(algName);
    perfTrace.EndStage();

    perfTrace.SetPath("hccl_exec_op");
    perfTrace.BeginStage("HcclExecOp");
    CHK_RET(perfTrace.Record(HcclExecOp(comm, param, topoInfo, algName, resPack)));
    perfTrace.EndStage();

    HCCL_INFO("Execute ReduceScatterOutPlaceGraphMode success.");
    perfTrace.SetResult(HcclResult::HCCL_SUCCESS);
    return HCCL_SUCCESS;
}
}