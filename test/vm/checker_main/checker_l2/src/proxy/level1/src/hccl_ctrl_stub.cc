/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_common.h"
#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl_proxy_pub.h"
// #include "hccl/hccl_rankgraph.h"
#include <iostream>
#include "sim_communicator.h"
#include "hccl_proxy_pub.h"
#include "task_status_cache.h"
#include "hccl_sim_shm_manager.h"
#include "hccl_vm.h"

struct InitCommDomainBarrier {
    ipc::interprocess_mutex mutex {};
    ipc::interprocess_condition cond;
    uint32_t finCount{0};                     // 当前已到达的进程数

	InitCommDomainBarrier() = default;
	InitCommDomainBarrier(const InitCommDomainBarrier&) = delete;
	InitCommDomainBarrier& operator=(const InitCommDomainBarrier&) = delete;

	void Reset()
	{
		ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
		finCount = 0;
	}

    void Wait(uint32_t target) {
        ipc::scoped_lock<ipc::interprocess_mutex> lock(mutex);
        
        finCount++;

        if (finCount < target) {
            cond.wait(lock);
        } else {
            // 最后一个到的唤醒所有睡在条件变量上的进程
            cond.notify_all();
        }
    }
};

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
{
    printf("[%s] not support\n", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommDestroy(HcclComm comm)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    delete simCommunicator;
    printf("[HCCLCtrlStub][HcclCommDestroy]Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    const char* commId = simCommunicator->GetIdentifier().c_str();
    strcpy_s(commName, strlen(commId) + 1, commId);
    printf("[HCCLCtrlStub][HcclGetCommName] commName: %s\n", commName);
    printf("[HCCLCtrlStub][HcclGetCommName] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    *rankSize = simCommunicator->GetRankSize();
    printf("[HCCLCtrlStub][HcclGetRankSize] rankSize: %u\n", (*rankSize));
    printf("[HCCLCtrlStub][HcclGetRankSize] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    *rank = simCommunicator->GetRankId();
    printf("[HCCLCtrlStub][HcclGetRankId] rank: %u\n", (*rank));
    printf("[HCCLCtrlStub][HcclGetRankId] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBuffer(HcclComm comm, void ** buffer, uint64_t *size)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    CommBuffer commBuffer;
    HcclResult ret = simCommunicator->GetHcclBuffer(&commBuffer);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclGetHcclBuffer] Fail\n");
        return ret;
    }
    *buffer = commBuffer.addr;
    *size = commBuffer.size;
    printf("[HCCLCtrlStub][HcclGetHcclBuffer] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankGraph(HcclComm comm, GraphType type, void **graph, uint32_t *len)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclResult ret = simCommunicator->GetCommRankGraph(graph, len);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclGetRankGraph]GetCommRankGraph failed.\n");
        return ret;
    }
    printf("[HCCLCtrlStub][HcclGetRankGraph]Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine, uint64_t size, void **ctx)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::SimContextMgr* simContextMgr = simCommunicator->contextManager_.get();
    HcclMem hcclMem;
    hcclMem.size = size;
    HcclResult ret = simContextMgr->CreateCommEngineCtx(ctxTag, engine, &hcclMem);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclCreateEngineCtx]CreateCommEngineCtx failed.\n");
        return ret;
    }
    *ctx = hcclMem.addr;
    printf("[HCCLCtrlStub][HcclCreateEngineCtx] engineCtx %p\n", hcclMem.addr);
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine, void **ctx, uint64_t *size)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::SimContextMgr* simContextMgr = simCommunicator->contextManager_.get();
    HcclMem hcclMem;
    HcclResult ret = simContextMgr->GetCommEngineCtx(ctxTag, engine, &hcclMem);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclGetEngineCtx]HcclGetEngineCtx failed.\n");
        return ret;
    } 
    *ctx = hcclMem.addr;
    *size = hcclMem.size;
    printf("[HCCLCtrlStub][HcclGetEngineCtx] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag, const void *srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::SimContextMgr* simContextMgr = simCommunicator->contextManager_.get();
    HcclMem hcclMem;
    HcclResult ret = simContextMgr->GetCommEngineCtx(ctxTag, engine, &hcclMem);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclEngineCtxCopy]HcclGetEngineCtx failed.\n");
        return ret;
    } 
    int ret2 = memcpy_s(reinterpret_cast<uint8_t*>(hcclMem.addr) + dstCtxOffset, hcclMem.size, srcCtx, size);
    if (ret2 != 0) {
        printf("[HCCLCtrlStub][HcclEngineCtxCopy]memcpy_s failed.\n");
        return HCCL_E_MEMORY;
    }
    printf("[HCCLCtrlStub][HcclEngineCtxCopy] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescs,uint32_t channelNum, ChannelHandle *channels)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::SimChannelMgr* simChannelMgr = simCommunicator->channelMgr_.get();
    const char* channelTag = "Mock_Channel_Tag";
    HcclResult ret = simChannelMgr->ChannelCommCreate(simCommunicator->GetIdentifier(), channelTag, engine, channelDescs, channelNum, channels);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclChannelCreate]ChannelCommCreate failed.\n");
        return ret;
    }
    printf("[HCCLCtrlStub][HcclChannelCreate] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void **buffer, uint64_t *size)
{
    printf("[HCCLCtrlStub][HcclChannelGetHcclBuffer]CommChannelGetHcclBuffer start.\n");
    uint32_t mode = SHMManager::GetHcclVmMode();
    CommBuffer commBuffer;
    if (mode == HcclSim::HcclVmMode::CHECKER) {
        // checker
        HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
        HcclProxy::SimChannelMgr* simChannelMgr = simCommunicator->channelMgr_.get();
        HcclResult ret = simChannelMgr->MockCommChannelGetHcclBuffer(channel, &commBuffer);
        if (ret != HCCL_SUCCESS) {
            printf("[HCCLCtrlStub][HcclChannelGetHcclBuffer][Checker] CommChannelGetHcclBuffer failed.\n");
            return ret;
        }
        printf("[HCCLCtrlStub][HcclChannelGetHcclBuffer][Checker] Success\n");
    } else {
        HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
        HcclProxy::SimChannelMgr* simChannelMgr = simCommunicator->channelMgr_.get();
        HcclResult ret = simChannelMgr->CommChannelGetHcclBuffer(channel, &commBuffer);
        if (ret != HCCL_SUCCESS) {
            printf("[HCCLCtrlStub][HcclChannelGetHcclBuffer] CommChannelGetHcclBuffer failed.\n");
            return ret;
        }
        printf("[HCCLCtrlStub][HcclChannelGetHcclBuffer] Success\n");
    }
    *buffer = commBuffer.addr;
    *size = commBuffer.size;
    return HCCL_SUCCESS; 
}

HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle *threads)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::SimThreadMgr *simThreadMgr = simCommunicator->independentOpThreadMgr_.get();
    HcclResult ret = simThreadMgr->CommAllocThreadRes(comm, engine, threadNum, notifyNumPerThread, threads);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclThreadAcquire]CommAllocThreadRes failed.\n");
        return ret;
    }
    printf("[HCCLCtrlStub][HcclThreadAcquire] Success\n");
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::SimThreadMgr *simThreadMgr = simCommunicator->independentOpThreadMgr_.get();
    HcclResult ret = simThreadMgr->CommAllocThreadResByStream(engine, stream, notifyNum, thread);
    if (ret != HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclThreadAcquireWithStream]CommAllocThreadResByStream failed.\n");
        return ret;
    }
    printf("[HCCLCtrlStub][HcclThreadAcquireWithStream] Success\n");
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceType(DevType &devType)
{
    devType = DevType::DEV_TYPE_910B;
    printf("[HCCLCtrlStub][hrtGetDeviceType]Return DEV_TYPE_910B.\n");
    return HCCL_SUCCESS;
}

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
{
    HcclRootInfo rootInfoTmp = {0};
    *rootInfo = rootInfoTmp;
    printf("[HCCLCtrlStub][HcclGetRootInfo]Success\n");
    return HCCL_SUCCESS;
}

const char* const SHM_COMM_INIT_BARRIER = "SimCommDomainInitBarrier";
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)
{
    if (!curr_dev.second.IsValid()) {
        printf("[HcclCommInitRootInfo]Init SimCommunicator failed, cannot find current device.\n");
        return HCCL_E_INTERNAL;
    }
    TaskStatusCache::GetInstance().SetCurrentRankId(rank);
    HcclSim::HcclVmResult ret = SetCommDomain(nRanks, rank, curr_dev.second);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        printf("[HcclCommInitRootInfo] SetCommDomain failed.\n");
        return HcclResult::HCCL_E_INTERNAL;
    }
    // SHM内创建共享barrier
    InitCommDomainBarrier* barrierPtr = SHMManager::GetSegment().find_or_construct<InitCommDomainBarrier>(SHM_COMM_INIT_BARRIER)();
    barrierPtr->Wait(nRanks);
    // TopoMeta topo = {{{0, 1}}}; // todo 干掉TopoMeta
    ShmCommDomain* commDomain = SHMManager::FindShmObject<ShmCommDomain>(SHM_MODULE_COMM_DOMAIN);

    HcclProxy::SimCommunicator *simCommunicator = new HcclProxy::SimCommunicator();
    HcclResult initRet;
    uint32_t mode = SHMManager::GetHcclVmMode();
    if (mode == HcclSim::HcclVmMode::CHECKER) {
        // checker
        initRet = simCommunicator->MockInit(commDomain, rank, nRanks);
    } else{
        // runner
        initRet = simCommunicator->Init(commDomain, rank, nRanks);
    }
    if (initRet != HCCL_SUCCESS) {
        printf("[HcclCommInitRootInfo]Init SimCommunicator failed.\n");
        return initRet;
    }
    *comm = static_cast<void*>(simCommunicator);
    printf("[HcclCommInitRootInfo]nRanks:%d, rank:%d, HcclRootInfo:%p, HcclComm:%p\n", nRanks, rank, rootInfo, comm);
    printf("[HCCLCtrlStub][HcclCommInitRootInfo] Success\n");
    return HCCL_SUCCESS;
}


int32_t HcommBatchModeStart(const char *batchTag)
{
    printf("[%s] not support. Return success.\n", __func__);
    return 0;
}

int32_t HcommBatchModeEnd(const char *batchTag)
{
    printf("[%s] not support. Return success.\n", __func__);
    return 0;
}
int32_t HcommAcquireComm(const char* commId)
{
    printf("[%s] not support. Return success.\n", __func__);
    return 0;
}

int32_t HcommReleaseComm(const char* commId)
{
    printf("[%s] not support. Return success.\n", __func__);
    return 0;
}

HcclResult HcclGetNetLayers(HcclComm comm, uint32_t **netLayers, uint32_t *netLayerNum)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::TopoModel *topoModel = simCommunicator->topoModel_.get();
    topoModel->GetNetLayers(netLayers, netLayerNum);
    printf("[HCCLCtrlStub][HcclGetNetLayers] Success\n");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetInstRanksByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t **ranks, uint32_t *rankNum)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::TopoModel *topoModel = simCommunicator->topoModel_.get();
    HcclResult ret = topoModel->GetInstRanksByNetLayer(curr_dev.first, netLayer, ranks, rankNum);
    printf("[HCCLCtrlStub][HcclGetInstRanksByNetLayer] curRank: %u, netLayer: %u, rankNum: %u\n", curr_dev.first, netLayer, *rankNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclGetInstRanksByNetLayer] failed\n");
        return ret;
    }
    printf("[HCCLCtrlStub][HcclGetInstRanksByNetLayer] Success\n");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::TopoModel *topoModel = simCommunicator->topoModel_.get();
    HcclResult ret = topoModel->GetInstSizeByNetLayer(simCommunicator->GetRankId(), netLayer, rankNum);
    printf("[HCCLCtrlStub][HcclGetInstSizeByNetLayer] curRank: %u, netLayer: %u, rankNum: %u\n", simCommunicator->GetRankId(), netLayer, *rankNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclGetInstSizeByNetLayer] failed\n");
        return ret;
    }
    printf("[HCCLCtrlStub][HcclGetInstSizeByNetLayer] Success\n");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, CommTopo *topoType)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::TopoModel *topoModel = simCommunicator->topoModel_.get();
    DevType devType = DevType::DEV_TYPE_910B;
    HcclResult ret = hrtGetDeviceType(devType);
    if (ret != HcclResult::HCCL_SUCCESS) {
        printf("[HCCLCtrlStub][HcclGetInstTopoTypeByNetLayer] hrtGetDeviceType failed\n");
        return ret;
    }
    topoModel->GetInstTopoTypeByNetLayer(devType, netLayer, topoType);
    printf("[HCCLCtrlStub][HcclGetInstTopoTypeByNetLayer] devType: %d, netLayer: %u\n", (int)devType, netLayer);
    printf("[HCCLCtrlStub][HcclGetInstTopoTypeByNetLayer] Success\n");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetInstSizeListByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t **instSizeList, uint32_t *listSize)
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    HcclProxy::TopoModel *topoModel = simCommunicator->topoModel_.get();
    topoModel->GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
    printf("[HCCLCtrlStub][HcclGetInstSizeListByNetLayer] netLayer: %u, listSize: %u\n", netLayer, *listSize);
    printf("[HCCLCtrlStub][HcclGetInstSizeListByNetLayer] Success\n");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink **links, uint32_t *linkNum) 
{
    HcclProxy::SimCommunicator *simCommunicator = static_cast<HcclProxy::SimCommunicator *>(comm);
    simCommunicator->topoModel_->GetLinks(netLayer, srcRank, dstRank, links, linkNum);
    printf("[HCCLCtrlStub][HcclGetLinks] Success\n");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcommRegOpInfo(const char* commId, void* opInfo, size_t size)
{
    printf("[HCCLCtrlStub]HcommRegOpInfo\n");
    return HcclResult::HCCL_SUCCESS;
}
using HcclGetOpInfoCallback = void (*)(const void *opInfo, char *outPut, size_t size);

HcclResult HcommRegOpTaskException(const char* commId, HcclGetOpInfoCallback callback)
{
    printf("[HCCLCtrlStub]HcommRegOpTaskException\n");
    return HcclResult::HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif  // __cplusplus