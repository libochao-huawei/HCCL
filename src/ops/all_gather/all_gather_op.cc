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
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>


using namespace std;
using namespace ops_hccl;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

namespace {
constexpr uint64_t DEFAULT_ALLGATHER_DUMP_COUNT = 46263936;
constexpr const char *ALLGATHER_DUMP_ENABLE_ENV = "HCCL_ALLGATHER_DUMP_ENABLE";
constexpr const char *ALLGATHER_DUMP_COUNT_ENV = "HCCL_ALLGATHER_DUMP_COUNT";
constexpr const char *ALLGATHER_DUMP_PATH_ENV = "HCCL_ALLGATHER_DUMP_PATH";
constexpr const char *DEFAULT_ALLGATHER_DUMP_PATH = "/tmp/hccl_allgather_dump";
std::atomic<uint64_t> g_allGatherDumpSeq{0};

bool IsAllGatherDumpEnabled()
{
    const char *enable = std::getenv(ALLGATHER_DUMP_ENABLE_ENV);
    return enable != nullptr && std::string(enable) == "1";
}

uint64_t GetAllGatherDumpCount()
{
    const char *dumpCount = std::getenv(ALLGATHER_DUMP_COUNT_ENV);
    if (dumpCount == nullptr || dumpCount[0] == '\0') {
        return DEFAULT_ALLGATHER_DUMP_COUNT;
    }
    char *end = nullptr;
    uint64_t count = std::strtoull(dumpCount, &end, 10);
    if (end == dumpCount || *end != '\0') {
        HCCL_WARNING("[AllGatherDump] invalid %s[%s], use default[%llu].",
                     ALLGATHER_DUMP_COUNT_ENV, dumpCount, DEFAULT_ALLGATHER_DUMP_COUNT);
        return DEFAULT_ALLGATHER_DUMP_COUNT;
    }
    return count;
}

bool ShouldDumpAllGather(uint64_t sendCount)
{
    if (!IsAllGatherDumpEnabled()) {
        return false;
    }
    uint64_t dumpCount = GetAllGatherDumpCount();
    return dumpCount == 0 || sendCount == dumpCount;
}

std::string GetAllGatherDumpPath()
{
    const char *dumpPath = std::getenv(ALLGATHER_DUMP_PATH_ENV);
    if (dumpPath == nullptr || dumpPath[0] == '\0') {
        return DEFAULT_ALLGATHER_DUMP_PATH;
    }
    return std::string(dumpPath);
}

std::string SanitizeFilePart(const std::string &value)
{
    std::string result;
    result.reserve(value.size());
    for (char ch : value) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) || ch == '_' || ch == '-' || ch == '.') {
            result.push_back(ch);
        } else {
            result.push_back('_');
        }
    }
    return result;
}

bool EnsureDumpDir(const std::string &dumpPath)
{
    if (mkdir(dumpPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0 || errno == EEXIST) {
        return true;
    }
    HCCL_WARNING("[AllGatherDump] mkdir path[%s] failed, errno[%d].", dumpPath.c_str(), errno);
    return false;
}

HcclResult DumpAllGatherDeviceBuffer(const void *deviceBuf, uint64_t elementCount, uint64_t byteSize,
    HcclDataType dataType, aclrtStream stream, uint32_t rank, uint32_t rankSize, const std::string &tag,
    const std::string &stage)
{
    CHK_PRT_RET(deviceBuf == nullptr || byteSize == 0,
                HCCL_WARNING("[AllGatherDump] skip empty %s dump, ptr[%p], byteSize[%llu].",
                             stage.c_str(), deviceBuf, byteSize),
                HCCL_SUCCESS);

    aclError aclRet = aclrtSynchronizeStream(stream);
    CHK_PRT_RET(aclRet != ACL_SUCCESS,
                HCCL_WARNING("[AllGatherDump] synchronize stream failed before %s dump, ret[%d].",
                             stage.c_str(), aclRet),
                HCCL_SUCCESS);

    void *hostBuf = nullptr;
    aclRet = aclrtMallocHost(&hostBuf, byteSize);
    CHK_PRT_RET(aclRet != ACL_SUCCESS || hostBuf == nullptr,
                HCCL_WARNING("[AllGatherDump] malloc host buffer failed for %s, byteSize[%llu], ret[%d].",
                             stage.c_str(), byteSize, aclRet),
                HCCL_SUCCESS);

    aclRet = aclrtMemcpy(hostBuf, byteSize, deviceBuf, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        HCCL_WARNING("[AllGatherDump] copy %s device buffer failed, byteSize[%llu], ret[%d].",
                     stage.c_str(), byteSize, aclRet);
        (void)aclrtFreeHost(hostBuf);
        return HCCL_SUCCESS;
    }

    const std::string dumpPath = GetAllGatherDumpPath();
    if (!EnsureDumpDir(dumpPath)) {
        (void)aclrtFreeHost(hostBuf);
        return HCCL_SUCCESS;
    }

    uint64_t seq = g_allGatherDumpSeq.fetch_add(1);
    std::ostringstream fileName;
    fileName << dumpPath << "/allgather_" << stage << "_rank" << rank << "_ranks" << rankSize
             << "_count" << elementCount << "_dtype" << SanitizeFilePart(GetDataTypeEnumStr(dataType))
             << "_seq" << seq << "_pid" << getpid() << "_" << SanitizeFilePart(tag) << ".pt";

    std::ofstream out(fileName.str(), std::ios::binary);
    if (!out.is_open()) {
        HCCL_WARNING("[AllGatherDump] open dump file[%s] failed.", fileName.str().c_str());
        (void)aclrtFreeHost(hostBuf);
        return HCCL_SUCCESS;
    }

    out << "HCCL_ALLGATHER_DUMP_V1\n"
        << "stage=" << stage << "\n"
        << "rank=" << rank << "\n"
        << "rank_size=" << rankSize << "\n"
        << "count=" << elementCount << "\n"
        << "dtype=" << GetDataTypeEnumStr(dataType) << "\n"
        << "byte_size=" << byteSize << "\n"
        << "tag=" << tag << "\n"
        << "data\n";
    out.write(static_cast<const char *>(hostBuf), byteSize);
    out.close();
    HCCL_WARNING("[AllGatherDump] dump %s data to [%s], byteSize[%llu].",
                 stage.c_str(), fileName.str().c_str(), byteSize);
    (void)aclrtFreeHost(hostBuf);
    return HCCL_SUCCESS;
}
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

    HcclUs startut = TIME_NOW();// 走老流程的判断时间不统计在内
    std::string opTag;
    CHK_RET(AllGatherInitAndCheck(comm, sendBuf, recvBuf, sendCount, dataType, stream, opTag));

    CHK_RET(AllGatherEntryLog(sendBuf, recvBuf, sendCount, dataType, stream, opTag, "HcclAllGather"));

    // 执行AllGather
    CHK_RET_AND_PRINT_IDE(AllGatherOutPlace(sendBuf, recvBuf, sendCount, dataType, comm, stream, opTag), opTag.c_str());

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
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));

    u32 perDataSize = DATATYPE_SIZE_TABLE[dataType];
    u64 inputSize = sendCount * perDataSize;    // all gather 每个rank上一份数据
    u64 outputSize = inputSize * userRankSize;  // 每个卡上结果为rankSize份数据
    if (ShouldDumpAllGather(sendCount)) {
        CHK_RET(DumpAllGatherDeviceBuffer(sendBuf, sendCount, inputSize, dataType, stream, userRank,
                                          userRankSize, tag, "input"));
    }

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
        CHK_RET(HcclExecOpCcuFastLaunch(comm, param, ccuFastLaunchCtx));
        if (ShouldDumpAllGather(sendCount)) {
            CHK_RET(DumpAllGatherDeviceBuffer(recvBuf, sendCount * userRankSize, outputSize, dataType, stream, userRank,
                                              userRankSize, tag, "output"));
        }
        return HCCL_SUCCESS;
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        CHK_RET(HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream));
        if (ShouldDumpAllGather(sendCount)) {
            CHK_RET(DumpAllGatherDeviceBuffer(recvBuf, sendCount * userRankSize, outputSize, dataType, stream, userRank,
                                              userRankSize, tag, "output"));
        }
        return HCCL_SUCCESS;
    }
    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, enter SingleRankProc", __func__);
        CHK_RET(SingleRankProc(comm, param));
        if (ShouldDumpAllGather(sendCount)) {
            CHK_RET(DumpAllGatherDeviceBuffer(recvBuf, sendCount * userRankSize, outputSize, dataType, stream, userRank,
                                              userRankSize, tag, "output"));
        }
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    if (ShouldDumpAllGather(sendCount)) {
        CHK_RET(DumpAllGatherDeviceBuffer(recvBuf, sendCount * userRankSize, outputSize, dataType, stream, userRank,
                                          userRankSize, tag, "output"));
    }
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
