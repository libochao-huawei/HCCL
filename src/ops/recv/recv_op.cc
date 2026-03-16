/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "recv_op.h"
#include "op_common_ops.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;

extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclRecv(
    void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("[HcclRecv] Start.");
    if (!CheckHCCLIndependentOp()) {
        return HcclRecvInner(recvBuf, count, dataType, srcRank, comm, stream);
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return HcclRecvInner(recvBuf, count, dataType, srcRank, comm, stream);
    }
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclRecvInner(recvBuf, count, dataType, srcRank, comm, stream);
    }

    CHK_RET(InitEnvConfig());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    u32 userRank = INVALID_VALUE_RANKID;
    std::string tag;
    // 参数获取和校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("[HcclRecv] input count is 0, return recv success"), HcclResult::HCCL_SUCCESS);
    CHK_RET(GetAndCheckRecvPara(comm, recvBuf, count, dataType, srcRank, rankSize, userRank, tag));

    CHK_RET_AND_PRINT_IDE(RecvExec(recvBuf, count, dataType, srcRank, comm, stream, rankSize, tag), tag.c_str());

    HCCL_INFO("[HcclRecv][%d]<-[%d] Success.", userRank, srcRank);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclRecvGraphMode(
    void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const char* group, aclrtStream stream,
    const char* tag, void **streams, size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_INFO("[HcclRecvGraphMode] Start.");
    // 根据group获取通信域
    HcclComm comm = nullptr;
    HCCL_INFO("[HcclRecvGraphMode] get group name: %s", group);
    HcomGetCommHandleByGroup(group, &comm);

    CHK_RET(InitEnvConfig());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    u32 userRank = INVALID_VALUE_RANKID;
    std::string opTag;
    // 参数获取和校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("[HcclRecvGraphMode] input count is 0, return recv success"), HcclResult::HCCL_SUCCESS);
    CHK_RET(GetAndCheckRecvPara(comm, recvBuf, count, dataType, srcRank, rankSize, userRank, opTag));
    CHK_RET(HcclCheckTag(tag));

    // 拼装ResPackGraphMode
    ResPackGraphMode resPack;
    // 设置tag
    strncpy_s(resPack.tag, sizeof(resPack.tag), tag, sizeof(resPack.tag) - 1);
    // 设置streams
    if (streams != nullptr && streamCount > 0) {
        for (size_t i = 0; i < streamCount; i++) {
            resPack.streams.push_back(static_cast<aclrtStream>(streams[i]));
        }
    }
    // 设置scratchMem
    resPack.scratchMemAddr = scratchMemAddr;
    resPack.scratchMemSize = scratchMemSize;

    // 执行Recv
    CHK_RET_AND_PRINT_IDE(RecvExecGraphMode(recvBuf, count, dataType, srcRank, comm, stream, rankSize, tag, resPack), opTag.c_str());

    HCCL_INFO("[HcclRecvGraphMode][%d]<-[%d] Success.", userRank, srcRank);
    return HcclResult::HCCL_SUCCESS;
}

namespace ops_hccl {
    HcclResult GetAndCheckRecvPara(
        const HcclComm comm, const void *recvBuf, const uint64_t count, const HcclDataType dataType,
        const uint32_t srcRank, u32 &rankSize, u32 &userRank, std::string &tag)
    {
        // 参数校验
        RPT_INPUT_ERR(
            comm == nullptr,
            "EI0003",
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
            std::vector<std::string>({"HcclRecv", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(
            recvBuf == nullptr,
            "EI0003",
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
            std::vector<std::string>({"HcclRecv", "recvBuf", "nullptr", "please check recvBuf"}));
        CHK_PTR_NULL(recvBuf);
        CHK_RET(CheckCount(count));
        CHK_RET(CheckDataType(dataType, false));
        CHK_RET(HcclGetRankSize(comm, &rankSize));
        CHK_RET(HcclGetRankId(comm, &userRank));
        CHK_PRT_RET(userRank == srcRank, HCCL_ERROR("[HcclRecv] srcRank cannot be equal to self."), HcclResult::HCCL_E_NOT_SUPPORT);
        char commName[COMM_INDENTIFIER_MAX_LENGTH];
        CHK_RET(HcclGetCommName(comm, commName));
        tag = "Recv_" + string(commName) + "_" + std::to_string(srcRank) + "_" + std::to_string(userRank);
        CHK_RET(HcclCheckTag(tag.c_str()));
        CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());
        CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, srcRank), tag.c_str());
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult GenerateRecvOpParam(
        OpParam &param, void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
        const HcclComm comm, const aclrtStream stream, const std::string &tag)
    {
        // 获取通信域名称
        CHK_RET(HcclGetCommName(comm, param.commName));
        u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
        u64 dataSize = count * dataTypeSize;
        param.opType = HcclCMDType::HCCL_CMD_RECEIVE;
        param.enableDetour = false;

        DevType deviceType = DevType::DEV_TYPE_COUNT;
        CHK_RET(hrtGetDeviceType(deviceType));
        param.deviceType = deviceType;

        // topoInfo的tag，所有相同的算子可以共享
        int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
        if (ret <= 0) {
            HCCL_ERROR("failed to fill param.tag");
            return HcclResult::HCCL_E_INTERNAL;
        }

        param.stream = stream;
        param.inputPtr = nullptr;
        param.inputSize = dataSize;
        param.sendRecvRemoteRank = srcRank;
        param.outputPtr = recvBuf;
        param.outputSize = dataSize;
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult RecvExec(
        void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
        const HcclComm comm, const aclrtStream stream,
        const u32 &rankSize, const std::string &tag)
    {
        HCCL_DEBUG("[RecvExec][%s] Start.", tag);
        
        // 参数构建
        OpParam param;
        CHK_RET(GenerateRecvOpParam(param, recvBuf, count, dataType, srcRank, comm, stream, tag));
        param.opMode = OpMode::OPBASE;

        if (rankSize == 1) {
            HCCL_WARNING("[RecvExec][%s] ranksize == 1, enter SingleRankProc", tag.c_str());
            CHK_RET(SingleRankProc(param));
            return HcclResult::HCCL_SUCCESS;
        }

        std::string algName;
        std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
        CHK_RET(Selector(comm, param, topoInfo, algName));
        CHK_RET(HcclExecOp(comm, param, topoInfo, algName));

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult RecvExecGraphMode(
        void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
        const HcclComm comm, const aclrtStream stream,
        const u32 &rankSize, const std::string &tag, const ResPackGraphMode &resPack)
    {
        HCCL_DEBUG("[RecvExecGraphMode][%s] Start.", tag.c_str());

        // 参数构建
        OpParam param;
        CHK_RET(GenerateRecvOpParam(param, recvBuf, count, dataType, srcRank, comm, stream, tag));
        param.opMode = OpMode::OFFLOAD;

        if (rankSize == 1) {
            HCCL_WARNING("[RecvExecGraphMode][%s] ranksize == 1, enter SingleRankProc", tag.c_str());
            CHK_RET(SingleRankProc(param));
            return HcclResult::HCCL_SUCCESS;
        }

        std::string algName;
        std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
        CHK_RET(Selector(comm, param, topoInfo, algName));
        CHK_RET(HcclExecOp(comm, param, topoInfo, algName));

        return HcclResult::HCCL_SUCCESS;
    }
} // namespace ops_hccl
