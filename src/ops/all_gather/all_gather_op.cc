/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_op.h"
#include "op_common_ops.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

#include <fstream>
#include <sys/time.h>
#include <unistd.h>   // for getpid (可选)
#include <cstdlib>    // for getenv

using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

// 获取数据类型大小
static size_t GetDataTypeSize(HcclDataType dataType) {
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:   return 1;
        case HCCL_DATA_TYPE_INT16:  return 2;
        case HCCL_DATA_TYPE_INT32:  return 4;
        case HCCL_DATA_TYPE_INT64:  return 8;
        case HCCL_DATA_TYPE_UINT64: return 8;
        case HCCL_DATA_TYPE_FP16:   return 2;
        case HCCL_DATA_TYPE_FP32:   return 4;
        case HCCL_DATA_TYPE_FP64:   return 8;
        case HCCL_DATA_TYPE_BFP16:   return 2;
        default:
            HCCL_WARNING("Unknown HcclDataType %d", (int)dataType);
            return 4;
    }
}

static bool IsSaveEnabled() {
    static bool enabled = []() {
        char *env = std::getenv("HCCL_ALLGATHER_DUMP_ENABLE");
        return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
    }();
    return enabled;
}

// 保存设备缓冲区到文件（同步拷贝）
static void SaveDeviceBufferToFile(const char *tag, void *deviceBuf, uint64_t elemCount,
                                   HcclDataType dataType, HcclComm comm, uint64_t displayCount) {
    if (deviceBuf == nullptr || elemCount == 0) {
        return;
    }
    size_t elemSize = GetDataTypeSize(dataType);
    if (elemSize == 0) {
        HCCL_WARNING("Invalid dataType, skip saving %s", tag);
        return;
    }

    size_t totalBytes = elemCount * elemSize;
    const size_t MAX_SAVE_BYTES = 1ULL << 30; // 1GB 限制
    if (totalBytes > MAX_SAVE_BYTES) {
        HCCL_WARNING("%s size %zu exceeds max save limit %zu, skip saving", tag, totalBytes, MAX_SAVE_BYTES);
        return;
    }

    void *hostBuf = malloc(totalBytes);
    if (hostBuf == nullptr) {
        HCCL_WARNING("Failed to allocate %zu bytes for %s host buffer", totalBytes, tag);
        return;
    }

    aclError ret = aclrtMemcpy(hostBuf, totalBytes, deviceBuf, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("aclrtMemcpy failed for %s, error=%d", tag, ret);
        free(hostBuf);
        return;
    }

    // 获取 rankId
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &rankId));

    // 构造文件名
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    char filename[512];
    snprintf(filename, sizeof(filename),
             "allgather_%s_rank%u_%ld_%06ld_count%llu.bin",
             tag, rankId, tv.tv_sec, tv.tv_usec, (unsigned long long)displayCount);

    // 写文件
    std::ofstream ofs(filename, std::ios::binary);
    if (ofs) {
        ofs.write(static_cast<char*>(hostBuf), totalBytes);
        ofs.close();
        HCCL_INFO("Saved %s to %s, size=%zu bytes", tag, filename, totalBytes);
    } else {
        HCCL_WARNING("Failed to open file %s for %s", filename, tag);
    }

    free(hostBuf);
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                         aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclAllGather");

    if (GetHcommVersion() < 90000000) { // compat handle
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all gather success"), HCCL_SUCCESS);

    // ========== 公共保存（sendBuf） ==========
    if (IsSaveEnabled()) {
        SaveDeviceBufferToFile("send", sendBuf, sendCount, dataType, comm, sendCount);
    }

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(AllGatherInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, stream, opTag));

    CHK_RET(AllGatherEntryLog(sendBuf, recvBuf, sendCount, dataType, stream, opTag, "HcclAllGather"));

    // 执行AllGather
    CHK_RET_AND_PRINT_IDE(AllGatherOutPlace(sendBuf, recvBuf, sendCount, dataType, comm, stream, opTag), opTag.c_str());

    // ========== 公共保存（recvBuf） ==========
    if (IsSaveEnabled()) {
        // 获取通信域大小
        uint32_t rankSize = 1;
        if (HcclCommGetSize(comm, &rankSize) != HCCL_SUCCESS) {
            HCCL_WARNING("Failed to get rankSize, assume 1 for recvBuf saving");
            rankSize = 1;
        }
        uint64_t recvElemCount = sendCount * rankSize;
        SaveDeviceBufferToFile("recv", recvBuf, recvElemCount, dataType, comm, sendCount);
    }
    
    CHK_RET(LogHcclExit("HcclAllGather", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("Start to run execute HcclAllGatherGraphMode");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclAllGatherGraphMode] get group name: %s", group);
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return all gather success"), HCCL_SUCCESS);

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(AllGatherInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, stream, opTag));
    
    // 检查tag有效性
    CHK_RET(HcclCheckTag(tag));
    
    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    // 设置tag
    if (strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1) != 0) {
        HCCL_ERROR("failed to fill resPack.tag");
        return HCCL_E_INTERNAL;
    }
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;
    std::string tagStr = tag;

    CHK_RET(AllGatherEntryLog(sendBuf, recvBuf, sendCount, dataType, stream, opTag, "HcclAllGatherGraphMode"));

    // 执行AllGather
    CHK_RET_AND_PRINT_IDE(AllGatherOutPlaceGraphMode(sendBuf, recvBuf, sendCount, dataType, comm, stream, tagStr, resPack), tagStr.c_str());

    CHK_RET(LogHcclExit("HcclAllGatherGraphMode", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}
namespace ops_hccl {

HcclResult AllGatherInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, aclrtStream stream, std::string &opTag)
{
    // 入口的地方先解析环境变量，在初始化环境变量的时候需要设置为AICPU展开
    CHK_RET(InitEnvConfig());
    // 参数校验等工作
    // 检查入参指针有效性
    CHK_RET(CheckAllGatherInputPara(comm, sendBuf, recvBuf, stream));
    // tag有效性,是否过长
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    opTag = "AllGather_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));
    // 检查sendCount是否合法(超出系统上限)
    CHK_RET(CheckCount(sendCount));
    // 检查数据类型是否支持
    CHK_RET(CheckDataType(dataType, false));
    // 检查rank有效性，是否超出rankSize
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult CheckAllGatherInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf, const aclrtStream stream)
{
    // 入参合法性校验
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclAllGather", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclAllGather", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclAllGather", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclAllGather", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult AllGatherOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                                   aclrtStream stream, const std::string &tag, OpMode opMode, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute AllGatherOutPlaceCommon");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = sendCount * perDataSize;    // all gather 每个rank上一份数据
    u64 outputSize = inputSize * userRankSize;  // 每个卡上结果为rankSize份数据

    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = opMode;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    // 参数准备
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = sendCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param.enableDetour = false;
    param.deviceType = deviceType;
    
    CHK_RET(HcclGetOpExpansionMode(comm, param));

    CcuFastLaunchCtx *ccuFastLaunchCtx = nullptr;
    if (ShouldGoCcuFastLaunch(comm, param, &ccuFastLaunchCtx)) {
        return HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
    }
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    HCCL_INFO("Execute AllGatherOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                                      aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute AllGatherOutPlaceGraphMode");
    CHK_RET(AllGatherOutPlaceCommon(sendBuf, recvBuf, sendCount, dataType, comm, stream, tag, OpMode::OFFLOAD, resPack));
    HCCL_INFO("Execute AllGatherOutPlaceGraphMode success.");
    return HCCL_SUCCESS;
}


HcclResult AllGatherOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
                                      aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute AllGatherOutPlace");
    CHK_RET(AllGatherOutPlaceCommon(sendBuf, recvBuf, sendCount, dataType, comm, stream, tag, OpMode::OPBASE, ResPackGraphMode()));
    HCCL_INFO("Execute AllGatherOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult AllGatherEntryLog(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, aclrtStream stream, const std::string &tag, const std::string &opName)
{
    /* 接口交互信息日志 */
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}
}  // namespace ops_hccl