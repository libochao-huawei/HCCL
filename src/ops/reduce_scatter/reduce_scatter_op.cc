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
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include "load_kernel.h"
#include "hcomm_host_profiling_dl.h"

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclReduceScatter");
    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
#ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_910_93) {
#else
    if (deviceType != DevType::DEV_TYPE_910_95 && deviceType != DevType::DEV_TYPE_910_93) {
#endif
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    // 入口的地方先解析环境变量
    CHK_RET(InitEnvConfig());

    if (deviceType == DevType::DEV_TYPE_910_93 && GetBirsEnableFromEnv()) {
        HCCL_INFO("BIRS mode enabled for ReduceScatter");
    } else {
        HCCL_INFO("BIRS mode disabled for ReduceScatter");
        return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    }

    OpParam param;
    // 参数校验等工作
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);
    CHK_RET(CheckReduceScatterInputPara(comm, sendBuf, recvBuf, stream));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET(HcomCheckUserRank(rankSize, userRank));
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(HcclGetCommName(comm, param.commName));
    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "ReduceScatter_%s", param.commName);
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);
    CHK_RET(HcclCheckTag(param.tag));
    CHK_RET(CheckReduceOp(dataType, op));

    /* 接口交互信息日志 */
    CHK_RET(ReduceScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, op, stream, param.tag, "HcclReduceScatter"));

    // 执行ReduceScatter
    CHK_RET(ReduceScatterOutPlace(param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize));

    CHK_RET(LogHcclExit("HcclReduceScatter", param.tag, startut));

    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
 	     HcclReduceOp op, const char* group, aclrtStream stream, const char* tag, void** streams,
 	     size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclReduceScatterGraphMode");
    HcclComm comm = nullptr;
    HcomGetCommHandleByGroup(group, &comm);
    HCCL_INFO("[HcclReduceScatterGraphMode] get group name: %s", group);
    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    CHK_RET(InitEnvConfig());
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);
    CHK_RET(CheckReduceScatterInputPara(comm, sendBuf, recvBuf, stream));

    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string opTag = "ReduceScatter_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    CHK_RET(HcclCheckTag(tag));

    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, true));
    CHK_RET(CheckReduceOp(dataType, op));

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());

    ResPackGraphMode resPack;
    strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1);

    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }

    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;
    /* 接口交互信息日志 */
    CHK_RET(ReduceScatterEntryLog(sendBuf, recvBuf, recvCount, dataType, op, stream, opTag.c_str(), "HcclReduceScatterGraphMode"));
    CHK_RET_AND_PRINT_IDE(
        ReduceScatterOutPlaceGraphMode(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, tag, resPack),
        opTag);
    CHK_RET(LogHcclExit("HcclReduceScatterGraphMode", opTag.c_str(), startut));
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

    if (param.commName[0] == '\0') {
        CHK_RET(HcclGetCommName(comm, param.commName));
    }
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
    CHK_RET(PrepareReduceScatterParam(param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream, userRankSize,
 	    OpMode::OPBASE));
    #ifdef MACRO_DEV_TYPE_NEW
    if (param.deviceType == DevType::DEV_TYPE_950 && (GetHcommVersion() >= 90000000)) {
    #else
    if (param.deviceType == DevType::DEV_TYPE_910_95) {
    #endif
        CHK_RET(HcclGetOpExpansionMode(comm, param));
        CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
        if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
            return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
        }
        
        std::string algName;
        std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
        CHK_RET(Selector(comm, param, topoInfo, algName));
        if (ShouldUseInnerOp(param.opExecuteConfig)) {
            return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
        }
        if (userRankSize == 1) {
            HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
            CHK_RET(SingleRankProc(comm, param));
            return HcclResult::HCCL_SUCCESS;
        }
        CHK_RET(HcclExecOp(comm, param, topoInfo, algName));
    } else {
        if (IsAiCpuMode(param.deviceType, userRankSize)) {
            HCCL_DEBUG("is aicpu mode");
            CHK_RET(LoadAICPUKernel());
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
        } else {
            HCCL_DEBUG("is host mode");
            param.engine = CommEngine::COMM_ENGINE_CPU_TS;
        }
    
        uint64_t beginTime;
        if (HcommIsProfilingSupported()) {
            beginTime = HcommGetProfilingSysCycleTime();
        }
        CHK_RET(ExecOp(comm, param));  //保留原有A3流程
        if (HcommIsProfilingSupported()) {
           // 获取profiling op上报的信息
            HcomProInfoTmp profInfo;
            std::string algTypeStr = TransferAlgTypeStr(param.algType);
            CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()));
            CHK_SAFETY_FUNC_RET(strcpy_s(profInfo.commName, sizeof(profInfo.commName), param.commName));
            profInfo.beginTime = beginTime;
            profInfo.dataCount = param.DataDes.count;
            profInfo.dataType = static_cast<uint8_t>(param.DataDes.dataType);
            profInfo.cmdType = static_cast<uint8_t>(param.opType);
            CHK_PRT(HcommProfilingReportOp(profInfo));

            if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
                CHK_PTR_NULL(param.resCtx);
                AlgResourceCtx* tmpCtx = reinterpret_cast<AlgResourceCtx*>(param.resCtx);
                profInfo.slaveThreadNum = tmpCtx->slaveThreadNum;
                char* curThreadPtr = reinterpret_cast<char*>(param.resCtx); // 拿到所有host下发的thread
                curThreadPtr += sizeof(AlgResourceCtx);// 偏移指针
                ThreadHandle* curThreads = reinterpret_cast<ThreadHandle *>(curThreadPtr);
                CHK_PRT(HcommProfilingUnRegThread(profInfo,curThreads));
            }
        }
    }
    HCCL_INFO("Execute ReduceScatterOutPlace success.");
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
    OpParam param;
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));
    
    CHK_RET(PrepareReduceScatterParam(param, sendBuf, recvBuf, recvCount, dataType, op, comm, stream, userRankSize,
        OpMode::OFFLOAD));

    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);

    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclGetOpExpansionMode(comm, param));
    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute ReduceScatterOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}

bool GetBirsEnableFromEnv() {
    const char* val = std::getenv("HCCL_BIRS_ENABLE");
    if (val == nullptr) return false;
    std::string str = val;
    if (str == "TRUE") {
       return true;
    } else {
       return false;
    }
}

}