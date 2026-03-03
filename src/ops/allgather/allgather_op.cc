/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "allgather_op.h"

#include <string>
#include <vector>
#include "hccl/base.h"
#include "config_log.h"
#include "param_check.h"
#include "coll_alg_exec_registry.h"
#include "executor_base.h"
#include "op_common.h"

using namespace std;
using namespace ops_hccl;

namespace {
// 与注册中心里的执行器名称保持一致，用于按名获取执行器实例。
constexpr char ALLGATHER_RING_EXEC_NAME[] = "AllGatherRingExecutor";
}

HcclResult HcclAllGatherRing(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
    aclrtStream stream)
{
    // 空输入直接成功返回，符合常见集合通信接口语义。
    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("[HcclAllGatherRing] sendCount is 0"), HCCL_SUCCESS);
    // 基础入参检查，避免后续执行阶段访问非法指针。
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    // 检查 count/数据类型是否合法。
    CHK_RET(CheckCount(sendCount));
    CHK_RET(CheckDataType(dataType, false));

    // 生成唯一 tag：前缀 + 通信域名，后续用于通信资源申请/释放。
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    const string tag = "AllGatherRing_" + string(commName);
    CHK_RET(HcclCheckTag(tag.c_str()));
    // 转入内部统一 out-place 实现。
    return AllGatherRingOutPlace(sendBuf, recvBuf, sendCount, dataType, comm, stream, tag);
}

namespace ops_hccl {
namespace {
HcclResult CalcAllGatherBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo **topoInfo)
{
    return HcclCalcTopoInfo(comm, param, topoInfo);
}

AlgType GetAllGatherRingAlgType()
{
    return AlgType(AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING, AlgTypeLevel1::ALG_LEVEL1_RING,
        AlgTypeLevel2::ALG_LEVEL2_RING);
}

HcclResult AllocAlgResource(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    AlgResourceCtx *resCtxHost)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    u64 sizePerCcl = cclBufferSize / 2;
    resCtxHost->cclInputMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, sizePerCcl};
    resCtxHost->cclOutputMem = HcclMem{HCCL_MEM_TYPE_DEVICE,
        static_cast<void *>(static_cast<char *>(cclBufferAddr) + sizePerCcl), sizePerCcl};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread =
        resRequest.notifyNumPerThread.empty() ? 0 : resRequest.notifyNumPerThread[0];

    char *curPtr = reinterpret_cast<char *>(resCtxHost) + sizeof(AlgResourceCtx);
    ThreadHandle *thread = reinterpret_cast<ThreadHandle *>(curPtr);
    CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
        resRequest.notifyNumOnMainThread, thread));
    curPtr += sizeof(ThreadHandle);

    for (u32 index = 0; index < resRequest.slaveThreadNum; index++) {
        thread = reinterpret_cast<ThreadHandle *>(curPtr);
        CHK_RET(HcclThreadAcquire(comm, param.engine, 1,
            resRequest.notifyNumPerThread[index], thread));
        curPtr += sizeof(ThreadHandle);
    }

    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        std::vector<HcclChannelDesc> &channelReq = resRequest.channels[level];
        u32 validChannelNum = channelReq.size();
        std::vector<ChannelHandle> channelHandles(validChannelNum);
        if (validChannelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, param.engine, channelReq.data(),
                validChannelNum, channelHandles.data()));
        }

        SubCommInfo &subCommInfo = resCtxHost->algHierarchyInfo.infos[level];
        ChannelInfo *channels = reinterpret_cast<ChannelInfo *>(curPtr);
        for (u32 rank = 0; rank < subCommInfo.localRankSize; rank++) {
            channels[rank].isValid = false;
        }
        for (u32 idx = 0; idx < validChannelNum; idx++) {
            HcclChannelDesc &channelDesc = channelReq[idx];
            u32 levelRank = 0;
            CHK_RET(GetSubCommRankByUserRank(channelDesc.remoteRank, level, resCtxHost->algHierarchyInfo, levelRank));
            channels[levelRank].isValid = true;
            channels[levelRank].remoteRank = channelDesc.remoteRank;
            channels[levelRank].protocol = channelDesc.channelProtocol;
            channels[levelRank].notifyNum = channelDesc.notifyNum;
            channels[levelRank].handle = channelHandles[idx];

            void *remoteBufferAddr = nullptr;
            uint64_t remoteBufferSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &remoteBufferAddr, &remoteBufferSize));
            channels[levelRank].remoteInput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize};
            channels[levelRank].remoteOutput = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteBufferAddr, remoteBufferSize};
        }
        curPtr += sizeof(ChannelInfo) * subCommInfo.localRankSize;
    }

    HCCL_INFO("[AllGatherRing][AllocAlgResource] success.");
    return HCCL_SUCCESS;
}

HcclResult GetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<ExecutorBase> &executor,
    TopoInfo *topoInfo, AlgType &algType, AlgResourceCtx **resCtx)
{
    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size) == HCCL_SUCCESS) {
        *resCtx = static_cast<AlgResourceCtx *>(ctx);
        HCCL_INFO("[AllGatherRing][GetAlgRes] reuse resource.");
        return HCCL_SUCCESS;
    }

    AlgHierarchyInfo algHierarchyInfo;
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcResRequest(comm, param, topoInfo, algHierarchyInfo, resRequest, algType));

    size = sizeof(AlgResourceCtx);
    size += sizeof(ThreadHandle) * (resRequest.slaveThreadNum + 1);
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        size += sizeof(ChannelInfo) * algHierarchyInfo.infos[level].localRankSize;
    }

    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    *resCtx = static_cast<AlgResourceCtx *>(ctx);
    AlgResourceCtx *resCtxHost = *resCtx;

    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algType = algType;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;
    CHK_RET(AllocAlgResource(comm, param, resRequest, resCtxHost));
    CHK_RET(HcclEngineCtxCopy(comm, param.engine, param.algTag, resCtxHost, size, 0));
    return HCCL_SUCCESS;
}

HcclResult ExecAllGatherRingOp(HcclComm comm, OpParam &param)
{
    // 计算通信拓扑与层级信息，后续用于算法选择和资源申请。
    TopoInfo *topoInfo = nullptr;
    CHK_RET(CalcAllGatherBaseTopoInfo(comm, param, &topoInfo));

    // 获取默认算法类型后，强制指定为 ring/ring（二级）实现。
    AlgType algType = GetAllGatherRingAlgType();

    // 生成算法标签，格式：<opTag>_<executorName>。
    int ret = sprintf_s(param.algTag, sizeof(param.algTag), "%s_%s", param.tag, ALLGATHER_RING_EXEC_NAME);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[ExecAllGatherRingOp] failed to fill param.algTag"), HCCL_E_INTERNAL);

    // 从执行器注册中心按名获取 Ring 执行器对象。
    std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(ALLGATHER_RING_EXEC_NAME);
    CHK_PRT_RET(executor.get() == nullptr, HCCL_ERROR("[ExecAllGatherRingOp] failed to find executor[%s]",
        ALLGATHER_RING_EXEC_NAME), HCCL_E_PARA);

    // 申请算法执行资源（线程/通道/缓冲等），并启动编排执行。
    AlgResourceCtx *resCtx = nullptr;
    CHK_RET(GetAlgRes(comm, param, executor, topoInfo, algType, &resCtx));
    CHK_RET(executor->Orchestrate(param, resCtx));
    // 把最终算法类型写回参数，便于后续日志/观测。
    param.algType = algType;
    return HCCL_SUCCESS;
}
}

HcclResult AllGatherRingOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
    aclrtStream stream, const std::string &tag)
{
    // 计算输入/输出总字节：
    // inputSize = 单 rank 输入大小
    // outputSize = inputSize * rankSize（拼接所有 rank）
    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 unitSize = SIZE_TABLE[dataType];
    u64 inputSize = sendCount * unitSize;
    u64 outputSize = inputSize * rankSize;

    // 组装统一 OpParam，交给执行器层处理。
    OpParam param;
    CHK_RET(HcclGetCommName(comm, param.commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag.c_str());
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[AllGatherRingOutPlace] failed to fill param.tag"), HCCL_E_INTERNAL);

    // 基本执行上下文。
    param.stream = stream;
    param.opMode = OpMode::OPBASE;
    param.engine = CommEngine::COMM_ENGINE_CPU_TS;
    // 输入输出缓冲信息。
    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    // 数据描述与算子描述。
    param.DataDes.count = sendCount;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    // AllGather 没有 root 语义，这里保持默认值 0。
    param.root = 0;

    // 记录设备类型，供资源与执行路径决策使用。
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    param.deviceType = deviceType;

    // 进入执行器编排。
    CHK_RET(ExecAllGatherRingOp(comm, param));
    HCCL_INFO("[AllGatherRingOutPlace] success.");
    return HCCL_SUCCESS;
}
}
