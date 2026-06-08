/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // 为 dlsym 的 RTLD_NEXT 提供声明
#endif
#include <dlfcn.h>

#include "barrier_op.h"
#include "op_common_ops.h"
#include <string>


using namespace std;
using namespace ops_hccl;

// 旧 hcomm 未导出 HcclBarrierInner（该符号在本次改名时才新增）。
// 用弱引用，使其在旧 hcomm 上缺失时解析为空指针，而不是链接/加载期未定义符号报错，
// 从而支持「旧 hcomm + 新 hccl」组合。
#pragma weak HcclBarrierInner

namespace {
// 回退到老 barrier 流程，兼容「旧 hcomm + 新 hccl」：
//   1) 新/配套 hcomm：HcclBarrierInner 存在，直接调用；
//   2) 旧 hcomm：HcclBarrierInner 弱引用为空，用 dlsym(RTLD_NEXT) 委派到旧 hcomm
//      导出的老 HcclBarrier（oldBarrier != &HcclBarrier 防止自递归）；
//   3) 两者都拿不到：返回明确错误，提示升级 hcomm。
HcclResult BarrierFallbackToOldFlow(HcclComm comm, aclrtStream stream)
{
    using BarrierFn = HcclResult (*)(HcclComm, aclrtStream);
    BarrierFn innerFn = HcclBarrierInner;  // 弱引用：旧 hcomm 上为空
    if (innerFn != nullptr) {
        return innerFn(comm, stream);
    }
    HCCL_WARNING("[Barrier] HcclBarrierInner not found (old hcomm?), delegate to legacy HcclBarrier via dlsym");
    BarrierFn oldBarrier = reinterpret_cast<BarrierFn>(dlsym(RTLD_NEXT, "HcclBarrier"));
    if (oldBarrier != nullptr && oldBarrier != &HcclBarrier) {
        return oldBarrier(comm, stream);
    }
    HCCL_ERROR("[Barrier] cannot fallback: hcomm too old (missing HcclBarrierInner), please upgrade hcomm");
    return HCCL_E_NOT_SUPPORT;
}
}  // namespace

HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclBarrier");

    if (GetHcommVersion() < 90000000) { // compat handle
        return BarrierFallbackToOldFlow(comm, stream);
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95) {
    #endif
        return BarrierFallbackToOldFlow(comm, stream);
    }

    HcclUs startut = TIME_NOW();
    std::string opTag;
    CHK_RET(BarrierInitAndCheck(comm, stream, opTag));

    CHK_RET(BarrierEntryLog(stream, opTag, "HcclBarrier"));

    CHK_RET_AND_PRINT_IDE(BarrierOutPlace(comm, stream, opTag), opTag.c_str());

    CHK_RET(LogHcclExit("HcclBarrier", opTag.c_str(), startut));

    return HCCL_SUCCESS;
}

namespace ops_hccl {

HcclResult BarrierInitAndCheck(HcclComm comm, aclrtStream stream, std::string &opTag)
{
    CHK_RET(InitEnvConfig());
    CHK_RET(CheckBarrierInputPara(comm, stream));

    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    opTag = "Barrier_" + string(commName);
    CHK_RET(HcclCheckTag(opTag.c_str()));

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), opTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult CheckBarrierInputPara(const HcclComm comm, const aclrtStream stream)
{
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclBarrier", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                  std::vector<std::string>({"HcclBarrier", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    return HCCL_SUCCESS;
}

HcclResult BarrierOutPlace(HcclComm comm, aclrtStream stream, const std::string &tag)
{
    HCCL_INFO("Start to execute BarrierOutPlace");
    u32 userRankSize;
    CHK_RET(HcclGetRankSize(comm, &userRankSize));

    OpParam param{};
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.stream = stream;
    param.opMode = OpMode::OPBASE;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    if (ret <= 0) {
        HCCL_ERROR("failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }

    // Barrier 无数据搬运，所有 buffer/count 全部清零；模板内部不读这些字段。
    param.inputPtr = nullptr;
    param.inputSize = 0;
    param.outputPtr = nullptr;
    param.outputSize = 0;
    param.DataDes.count = 0;
    param.DataDes.dataType = HCCL_DATA_TYPE_FP32;  // 占位，模板不使用
    param.opType = HcclCMDType::HCCL_CMD_BARRIER;
    param.enableDetour = false;
    param.deviceType = deviceType;

    CHK_RET(HcclGetOpExpansionMode(comm, param));

    if (userRankSize == 1) {
        HCCL_WARNING("[%s] rankSize == 1, barrier returns directly", __func__);
        return HCCL_SUCCESS;
    }

    // 新流程（框内 AICPU + 框间 DPU）仅在「框间 host-DPU」场景启用，
    // 其余场景（普通 AICPU、单框、框间 device 链路等）回退到老的 HcclBarrierInner。
    if (!IsBarrierHostDpu(comm)) {
        HCCL_INFO("[BarrierOutPlace] not host-dpu scene, fallback to HcclBarrierInner");
        return BarrierFallbackToOldFlow(comm, stream);
    }

    std::string algName;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(Selector(comm, param, topoInfo, algName));
    if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
        return BarrierFallbackToOldFlow(comm, stream);
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, ResPackGraphMode()));
    HCCL_INFO("Execute BarrierOutPlace success.");
    return HCCL_SUCCESS;
}

HcclResult BarrierEntryLog(aclrtStream stream, const std::string &tag, const std::string &opName)
{
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], streamId[%d], deviceLogicId[%d]",
            tag.c_str(), streamId, deviceLogicId);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-" + opName + ":" + std::string(stackLogBuffer);
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
