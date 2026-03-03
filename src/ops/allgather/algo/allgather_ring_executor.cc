/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "allgather_ring_executor.h"

#include "channel.h"
#include "topo.h"

namespace ops_hccl {
AllGatherRingExecutor::AllGatherRingExecutor() : ExecutorBase()
{
    // 本执行器只声明支持 ring/ring 算法组合。
    desc_.level1SupportedAlgos = {AlgTypeLevel1::ALG_LEVEL1_RING};
    desc_.level2SupportedAlgos = {AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult AllGatherRingExecutor::CalcResRequest(HcclComm comm, const OpParam& param, TopoInfo* topoInfo,
    AlgHierarchyInfo& algHierarchyInfo, AlgResourceRequest& resourceRequest, AlgType& algType)
{
    // 基于 A3 通用规则计算拓扑层级信息。
    CHK_RET(CalcGeneralTopoInfoForA3(comm, topoInfo, algHierarchyInfo));
    // 刷新算法类型（会结合执行器支持能力和外部指定）。
    CHK_RET(RefreshAlgType(algType));
    // 当前实现仅支持单机/单 superPod 场景的 ring。
    CHK_PRT_RET((topoInfo->moduleNum > 1 || topoInfo->superPodNum > 1),
        HCCL_ERROR("[AllGatherRingExecutor][CalcResRequest] only single-server/single-superPod ring is supported. "
        "moduleNum[%u] superPodNum[%u]", topoInfo->moduleNum, topoInfo->superPodNum), HCCL_E_NOT_SUPPORT);

    // ring 实现不依赖额外从线程通知。
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    // 逐层申请通信通道（level0/1/2）。
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcLevel0ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcLevel1ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    std::vector<HcclChannelDesc> level2Channels;
    CHK_RET(CalcLevel2ChannelRequest(param, topoInfo, algHierarchyInfo, algType, level2Channels));
    resourceRequest.channels.push_back(level2Channels);
    return HCCL_SUCCESS;
}

HcclResult AllGatherRingExecutor::Orchestrate(const OpParam &param, AlgResourceCtx* resCtx)
{
    // 缓存执行所需上下文。
    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = std::string(param.tag);
    algType_ = resCtx->algType;
    unitSize_ = SIZE_TABLE[param.DataDes.dataType];

    // 入参指针不能为空。
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    // AlgResourceCtx 后面是资源打包布局：
    // [ThreadHandle...][ChannelInfo...]
    ThreadHandle *threadHandlePtr =
        reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(algResource_) + sizeof(AlgResourceCtx));
    thread_ = threadHandlePtr[0];

    // 解析并按 level/rank 组织 channel 信息，便于模板层按层索引。
    ChannelInfo *channelInfoPtr = reinterpret_cast<ChannelInfo *>(threadHandlePtr + algResource_->slaveThreadNum + 1);
    AlgHierarchyInfo& algHierarchyInfo = resCtx->algHierarchyInfo;
    channels_.resize(algHierarchyInfo.levels);
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        u32 curLevelRankSize = algHierarchyInfo.infos[level].localRankSize;
        channels_[level].resize(curLevelRankSize);
        for (u32 rank = 0; rank < curLevelRankSize; rank++) {
            channels_[level][rank] = channelInfoPtr[rank];
        }
        channelInfoPtr += curLevelRankSize;
    }
    // 进入分片主循环。
    return RunLoop(param);
}

HcclResult AllGatherRingExecutor::RunLoop(const OpParam &param)
{
    // 使用 level0 子通信域运行 ring 算法。
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo_));
    const u32 rankSize = level0CommInfo_.localRankSize;
    const u32 rank = level0CommInfo_.localRank;
    // 该实现要求 level0 rankSize 与用户通信域 rankSize 一致。
    CHK_PRT_RET(rankSize != topoInfo_->userRankSize,
        HCCL_ERROR("[AllGatherRingExecutor][RunLoop] rankSize mismatch, level0RankSize[%u] userRankSize[%u]",
        rankSize, topoInfo_->userRankSize), HCCL_E_NOT_SUPPORT);

    // cclOutputMem 是中间聚合缓冲区。
    auto cclOutputMem = algResource_->cclOutputMem;
    CHK_PRT_RET((cclOutputMem.size == 0), HCCL_ERROR("[AllGatherRingExecutor][RunLoop] cclBuffer size is zero"),
        HCCL_E_PARA);

    // 计算分片大小：
    // - 每轮最多处理 maxCountPerLoop 个元素
    // - 对齐到 HCCL_MIN_SLICE_ALIGN，避免非对齐影响通信/拷贝
    const u64 totalCount = param.DataDes.count;
    const u64 inputSize = totalCount * unitSize_;
    u64 maxCountPerLoop = cclOutputMem.size / rankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize_;
    CHK_PRT_RET(maxCountPerLoop == 0, HCCL_ERROR("[AllGatherRingExecutor][RunLoop] maxCountPerLoop is zero"),
        HCCL_E_INTERNAL);

    // CPU/CPU_TS 引擎路径需要显式 acquire/release 通信资源。
    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommAcquireComm(param.commName);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherRingExecutor][RunLoop] [%s] HcommAcquireComm failed",
            param.commName), static_cast<HcclResult>(ret));
    }

    // curUserInputPtr 指向当前轮待处理输入起点。
    u8 *curUserInputPtr = static_cast<u8 *>(param.inputPtr);
    u64 processedCount = 0;
    while (processedCount < totalCount) {
        // 当前轮分片计数和字节大小。
        const u64 curCount = std::min(maxCountPerLoop, totalCount - processedCount);
        const u64 curSliceSize = curCount * unitSize_;
        const u64 curOutputSize = curSliceSize * rankSize;
        // 本轮输入和中间输出内存描述。
        HcclMem curInputMem{HCCL_MEM_TYPE_DEVICE, curUserInputPtr, curSliceSize};
        HcclMem curOutputMem{cclOutputMem.type, cclOutputMem.addr, curOutputSize};

        // 交给模板层执行 ring 通信。
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = curInputMem;
        execMem.outputMem = curOutputMem;

        CHK_RET(KernelRun(param, execMem));

        // 中间输出布局为 [rank0Slice][rank1Slice]...[rankN-1Slice]，
        // 需要把每个 rank 分片拷贝到用户最终输出的对应偏移。
        for (u32 i = 0; i < rankSize; i++) {
            void *src = static_cast<u8 *>(curOutputMem.addr) + i * curSliceSize;
            void *dst = static_cast<u8 *>(param.outputPtr) + i * inputSize + processedCount * unitSize_;
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, curSliceSize)));
        }
        // 前移到下一轮分片。
        curUserInputPtr += curSliceSize;
        processedCount += curCount;
    }

    // 与 acquire 对应释放通信资源。
    if (param.engine == CommEngine::COMM_ENGINE_CPU_TS || param.engine == CommEngine::COMM_ENGINE_CPU) {
        int32_t ret = HcommReleaseComm(param.commName);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherRingExecutor][RunLoop] [%s] HcommReleaseComm failed",
            param.commName), static_cast<HcclResult>(ret));
    }

    HCCL_INFO("[AllGatherRingExecutor][RunLoop] rank[%u] run success.", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherRingExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    // 获取 level0 子通信域信息，用于模板层 rank/rankSize 输入。
    CHK_RET(GetSubCommInfo(COMM_LEVEL0, level0CommInfo_));
    // 按 rank 划分等长 slice，供 ring 每步发送/接收使用。
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, unitSize_, level0CommInfo_.localRankSize, dataSegsSlice));

    // 获取 Ring 模板实例并准备执行参数。
    std::unique_ptr<AlgTemplateBase> tempAlg =
        AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALLGATHER_RING);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, thread_, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSegsSlice));
    // 异步执行 ring 流程。
    return tempAlg->RunAsync(level0CommInfo_.localRank, level0CommInfo_.localRankSize, channels_[COMM_LEVEL0]);
}

HcclResult AllGatherRingExecutor::PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum,
    std::vector<Slice> &dataSlice) const
{
    // sliceNum 通常等于 rankSize，不能为 0。
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[AllGatherRingExecutor][PrepareDataSlice] sliceNum is zero."),
        HCCL_E_PARA);
    dataSlice.resize(sliceNum);
    // 每个 slice 大小相同：dataCount * unitSize。
    const u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        // rank i 的 slice 在输出缓冲中的线性偏移。
        dataSlice[i].offset = i * sliceSize;
    }
    return HCCL_SUCCESS;
}

// 注册执行器：名字需与上层获取时常量 ALLGATHER_RING_EXEC_NAME 一致。
REGISTER_EXEC("AllGatherRingExecutor", AllGatherRing, AllGatherRingExecutor);
}
