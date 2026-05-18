/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_op.h"
#include "coll_alg_exec_registry.h"
#include "config_log.h"
#include "hcomm_primitives.h"
#include "load_kernel.h"
#include "op_common_ops.h"
#include "topo.h"
#include "topo_host.h"
#include "hcomm_host_profiling_dl.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>

using namespace std;
using namespace ops_hccl;
constexpr uint32_t ROOTINFO_INDENTIFIER_MAX_LENGTH = 128;
extern "C" unsigned int LaunchAicpuKernel(OpParam *param);

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount,
    HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to run execute HcclScatter");
    // 获取设备类型拦截混合组网
    HcclHeterogMode allDeviceType;
    CHK_RET(HcclGetHeterogMode(comm, &allDeviceType));
    if(allDeviceType != HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS) {
        HCCL_ERROR("[HcclScatter] Scatter only support singleDeviceType");
        return HCCL_E_NOT_SUPPORT;
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    if (!RunIndependentOpExpansion(deviceType)) {
       return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    // 入口的地方先解析环境变量, 调用位置有特殊要求，不要变化
    CHK_RET(InitEnvConfig());
    
    // AclGraph引导到老的流程上面
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950 && IsStreamCapture(stream)) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95 && IsStreamCapture(stream)) {
    #endif
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }
    // 重执行引导到老的流程上面
    if (deviceType == DevType::DEV_TYPE_910_93 && (GetExternalInputIntraServerRetryEnable()
        || GetExternalInputInterServerRetryEnable() || GetExternalInputInterSuperPodRetryEnable())) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    // 图模式引导到老的流程上面
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    }

    // Attention! zeroCopy模式、recompute等先不支持，且当前不引导到老的流程上

    HcclUs startut = TIME_NOW(); // 走老流程的判断时间不统计在内

    OpParam param;
    // 参数校验等工作
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return scatter success"), HCCL_SUCCESS);
    CHK_RET(CheckScatterInputPara(comm, recvBuf));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));
    if (userRank == root) {     // 本rank为root节点，send_buff不可以为空
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
            std::vector<std::string>({"HcclScatter", "nullptr", "sendBuf", "non-null pointer"}));
        CHK_PTR_NULL(sendBuf);
    }
    CHK_RET(HcomCheckUserRank(rankSize, root));
    CHK_RET(CheckCount(recvCount));
    CHK_RET(CheckDataType(dataType, false));
    CHK_RET(HcclGetCommName(comm, param.commName));
    // topoInfo的tag，所有相同的算子可以共享
    int ret = sprintf_s(param.tag, sizeof(param.tag), "Scatter_%s", param.commName);
    CHK_PRT_RET((ret <= 0), "failed to fill param.tag", HCCL_E_INTERNAL);
    CHK_RET(HcclCheckTag(param.tag));

    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], recvCount[%llu], data_type[%s], root[%u]",
               param.tag, sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root);

    /* 接口交互信息日志 */
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        ACLCHECK(aclrtGetDevice(&deviceLogicId));
        s32 streamId = 0;
        ACLCHECK(aclrtStreamGetId(stream, &streamId));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
                             param.tag, sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", param.tag));
        std::string logInfo = "Entry-HcclScatter:" + std::string(stackLogBuffer); // capture的entry信息待补充
        HCCL_RUN_INFO("%s", logInfo.c_str());
    }

    CHK_RET(ScatterOutPlace(param, sendBuf, recvBuf, recvCount, dataType, root, comm, stream, rankSize));

    CHK_RET(LogHcclExit("HcclScatter", param.tag, startut));
    return HCCL_SUCCESS;
}

namespace ops_hccl {

HcclResult CheckScatterInputPara(const HcclComm comm, const void *recvBuf)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclScatter", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),\
        std::vector<std::string>({"HcclScatter", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult ScatterExecOp(OpParam &param, void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream, u32 userRankSize, uint64_t beginTime)
{
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
        if (ShouldUseInnerOp(param.opExecuteConfig) && param.opMode == OpMode::OPBASE) {
            return HcclScatterInner(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
        }
        if (userRankSize == 1) {
            HCCL_WARNING("[%s] ranksize == 1, enter SingleRankProc", __func__);
            CHK_RET(SingleRankProc(comm, param));
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_RET(HcclExecOp(comm, param, topoInfo, algName));
    } else {
        ProcessA3(comm, param, beginTime);
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterOutPlace(OpParam &param, void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream, u32 userRankSize)
{
    uint64_t beginTime;
    if (HcommIsProfilingSupported()) {
        beginTime = HcommGetProfilingSysCycleTime();
    }

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 inputSize = outputSize * userRankSize;

    param.stream = stream;
    param.opMode = OpMode::OPBASE;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (IsAiCpuMode(deviceType, userRankSize)) {
        HCCL_DEBUG("is aicpu mode");
        CHK_RET(LoadAICPUKernel());
        param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    } else {
        HCCL_DEBUG("is host mode");
        param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    }

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.DataDes.count = recvCount;
    param.DataDes.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    param.deviceType = deviceType;
    
    CHK_RET(ScatterExecOp(param, sendBuf, recvBuf, recvCount, dataType, root, comm, stream, userRankSize, beginTime));
    HCCL_INFO("Execute ScatterOutPlace success.");
    return HCCL_SUCCESS;
}

}