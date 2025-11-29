/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "hcomm_primitives.h"
#include "hccl_res.h"
#include "hcomm_primitives.h"
#include "hccl_rank_graph.h"
#include "hccl_rankgraph.h"
#include "acl/acl.h"
#include <memory>
#include <iostream>
#include <mutex>
#include "sim_communicator.h"
#include "sim_task.h"
#include "sim_world.h"
#include "sim_npu.h"
#include "sim_stream.h"
#include "sim_task_queue.h"
#include "sim_channel.h"
#include "transformer.h"

using namespace hccl;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

std::mutex lock_;

HcclResult HcclGetCommName(HcclComm commHandle, char *commName)
{
    CHK_PTR_NULL(commName);
    auto simComm = static_cast<HcclSim::SimCommunicator*>(commHandle);
    CHK_PTR_NULL(simComm);
    s32 ret = strncpy_s(commName, ROOTINFO_INDENTIFIER_MAX_LENGTH, simComm->GetIdentifier().c_str(),
        simComm->GetIdentifier().size());
    if (ret != EOK) {
        HCCL_ERROR("[%s] str copy fail. return %d", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    CHK_PTR_NULL(rankSize);
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    *rankSize = simComm->GetRankSize();
    HCCL_INFO("[%s] rankSize: %u", __func__, *rankSize);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    CHK_PTR_NULL(rank);
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    *rank = simComm->GetRankId();
    HCCL_INFO("[%s] rankId: %u", __func__, *rank);
    return HCCL_SUCCESS;
}

HcclResult HcclScatterInner(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    HCCL_ERROR("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclGetHcclBuffer(HcclComm comm, CommBuffer *buffer)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->GetHcclBuffer(buffer);
}

HcclResult CommChannelCreate(HcclComm comm, const char *channelTag, CommEngine engine,
    const ChannelDesc *channelDescList, uint32_t listNum, ChannelHandle *channelList)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->ChannelCommCreate(simComm->GetIdentifier(), std::string(channelTag), engine, channelDescList, listNum, channelList);
}

HcclResult CommCreateEngineCtx(HcclComm comm, const char *engineTag, CommEngine engine, HcclMem *engineCtx)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->contextManager_->CreateCommEngineCtx(std::string(engineTag), engine, engineCtx);
}

HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, CommBuffer *buffer)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->ChannelCommGetHcclBuffer(channel, buffer);
}

HcclResult CommGetRankGraph(HcclComm comm, GraphType type, void **graph, uint32_t *len)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->GetCommRankGraph(graph, len);
}

HcclResult HcclAllocThreadRes(
    HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle *thread)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->independentOpThreadMgr_->HcclAllocThreadRes(engine, threadNum, notifyNumPerThread, thread);
}

HcclResult CommAllocThreadResByStream(
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->independentOpThreadMgr_->CommAllocThreadResByStream(engine, stream, notifyNum, thread);
}

HcclResult CommGetEngineCtx(HcclComm comm, const char *engineTag, CommEngine engine, HcclMem *engineCtx)
{
    auto simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    return simComm->contextManager_->GetCommEngineCtx(std::string(engineTag), engine, engineCtx);
}

HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
{
    HCCL_ERROR("[%s] not support", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommDestroy(HcclComm comm)
{
    auto* simComm = static_cast<HcclSim::SimCommunicator*>(comm);
    CHK_PTR_NULL(simComm);
    delete simComm;
    return HCCL_SUCCESS;
}

HcclResult HcommInterThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout)
{
    std::lock_guard<std::mutex> lock(lock_);
    // timeout 暂时未使用
    static_cast<void>(timeout);

    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.从thread获得notifyId
    uint32_t notifyId = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetNotifyIdByIndex(notifyIdx);

    // 3.下发task
    auto task = std::make_shared<HcclSim::TaskStubLocalWaitFrom>(notifyId);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommInterThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    std::lock_guard<std::mutex> lock(lock_);
    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.从thread获得notifyId
    uint32_t notifyId = reinterpret_cast<HcclSim::SimHcclThread*>(dstThread)->GetNotifyIdByIndex(dstNotifyIdx);

    // 3.下发task
    auto task = std::make_shared<HcclSim::TaskStubLocalPostTo>(notifyId);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
{
    std::lock_guard<std::mutex> lock(lock_);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // 1.获取当前rankId和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.从模型用rankid查询NpuPos，从NpuPos获得SimNpu
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimNpu &npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curRank);

    // 3.查询DataSlice
    // 出参准备
    HcclSim::DataSlice srcSlice;
    HcclSim::DataSlice dstSlice;

    CHK_RET(npu.GetSlice(reinterpret_cast<uint64_t>(src), len, srcSlice));
    CHK_RET(npu.GetSlice(reinterpret_cast<uint64_t>(dst), len, dstSlice));

    auto task = std::make_shared<HcclSim::TaskStubLocalCopy>(srcSlice, dstSlice);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
{
    std::lock_guard<std::mutex> lock(lock_);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.获取远端和本地rankId
    uint32_t locRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLocRankId();
    uint32_t rmtRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtRankId();

    // 3.从模型用rankid查询NpuPos，从NpuPos获得SimNpu
    // src地址rank
    HcclSim::SimNpu &srcNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(locRank);
    // dst地址rank
    HcclSim::SimNpu &dstNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(rmtRank);

    // 4.查询DataSlice
    // 出参准备
    HcclSim::DataSlice srcSlice;
    HcclSim::DataSlice dstSlice;

    CHK_RET(srcNpu.GetSlice(reinterpret_cast<uint64_t>(src), len, srcSlice));
    CHK_RET(dstNpu.GetSlice(reinterpret_cast<uint64_t>(dst), len, dstSlice));

    // 5.通过抽象链接类型判断链接协议
    HcclSim::LinkInfo link(reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLinkType());

    // 6.下发task
    auto task = std::make_shared<HcclSim::TaskStubWrite>(rmtRank, link, srcSlice, dstSlice);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len)
{
    std::lock_guard<std::mutex> lock(lock_);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.获取远端和本地rankId
    uint32_t locRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLocRankId();
    uint32_t rmtRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtRankId();

    // 3.从模型用rankid查询NpuPos，从NpuPos获得SimNpu
    // src地址rank
    HcclSim::SimNpu &srcNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(rmtRank);
    // dst地址rank
    HcclSim::SimNpu &dstNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(locRank);

    // 4.查询DataSlice
    // 出参准备
    HcclSim::DataSlice srcSlice;
    HcclSim::DataSlice dstSlice;

    CHK_RET(srcNpu.GetSlice(reinterpret_cast<uint64_t>(src), len, srcSlice));
    CHK_RET(dstNpu.GetSlice(reinterpret_cast<uint64_t>(dst), len, dstSlice));

    // 5.通过抽象链接类型判断链接协议
    HcclSim::LinkInfo link(reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLinkType());

    // 6.下发task
    auto task = std::make_shared<HcclSim::TaskStubRead>(rmtRank, link, dstSlice, srcSlice);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult CommNotifyRecord(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    std::lock_guard<std::mutex> lock(lock_);
    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();

    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimNpu &npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curRank);

    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.获取远端和本地rankId
    uint32_t rmtRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtRankId();

    // 3.通过抽象链接类型判断链接协议
    HcclSim::LinkInfo link(reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLinkType());

    // 4.通过channel获得remoteNotify id
    uint32_t rmtNotifyId = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtNotifyIdByIndex(remoteNotifyIdx);

    // 5.下发task
    auto task = std::make_shared<HcclSim::TaskStubPost>(rmtRank, link, rmtNotifyId, HcclSim::NotifyTypeStub::READY, "POST");
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout)
{
    std::lock_guard<std::mutex> lock(lock_);
    //timeout 不参与 taskstubwait的构造
    static_cast<void>(timeout);

    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();

    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimNpu &npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curRank);

    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.获取远端和本地rankId
    uint32_t rmtRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtRankId();

    // 3.通过抽象链接类型判断链接协议
    HcclSim::LinkInfo link(reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLinkType());

    // 4.通过channel获得remoteNotify id
    uint32_t localNotifyId = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLocNotifyIdByIndex(localNotifyIdx);

    // 5.下发task
    auto task = std::make_shared<HcclSim::TaskStubWait>(rmtRank, link, localNotifyId, HcclSim::NotifyTypeStub::READY, "WAIT");
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommLocalReduceOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t count,
    HcclDataType dataType, HcclReduceOp reduceOp)
{
    std::lock_guard<std::mutex> lock(lock_);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // 1.获取当前rankId和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.从模型用rankid查询NpuPos，从NpuPos获得SimNpu
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimNpu &npu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(curRank);

    // 3.查询DataSlice
    // 出参准备
    HcclSim::DataSlice srcSlice;
    HcclSim::DataSlice dstSlice;

    CHK_RET(npu.GetSlice(reinterpret_cast<uint64_t>(src), count, dataType, srcSlice));
    CHK_RET(npu.GetSlice(reinterpret_cast<uint64_t>(dst), count, dataType, dstSlice));

    // 4.datatype转换成checker校验类型
    CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[dataType];
    CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[reduceOp];

    // 5.下发task
    auto task = std::make_shared<HcclSim::TaskStubLocalReduce>(srcSlice, dstSlice, checkerDataType, checkerReduceOp);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t count, HcclDataType dataType, HcclReduceOp reduceOp)
{
    std::lock_guard<std::mutex> lock(lock_);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();
    CHK_PTR_NULL(stream);

    // 2.获取远端和本地rankId
    uint32_t locRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLocRankId();
    uint32_t rmtRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtRankId();

    // 3.从模型用rankid查询NpuPos，从NpuPos获得SimNpu
    // src地址rank
    HcclSim::SimNpu &srcNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(locRank);
    // dst地址rank
    HcclSim::SimNpu &dstNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(rmtRank);

    // 4.查询DataSlice
    // 出参准备
    HcclSim::DataSlice srcSlice;
    HcclSim::DataSlice dstSlice;

    CHK_RET(srcNpu.GetSlice(reinterpret_cast<uint64_t>(src), count, dataType, srcSlice));
    CHK_RET(dstNpu.GetSlice(reinterpret_cast<uint64_t>(dst), count, dataType, dstSlice));

    // 5.datatype转换成checker校验类型
    CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[dataType];
    CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[reduceOp];

    // 6.通过抽象链接类型判断链接协议
    HcclSim::LinkInfo link(reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLinkType());

    // 7.下发task
    auto task = std::make_shared<HcclSim::TaskStubWriteReduce>(rmtRank, link, srcSlice, dstSlice, 
        checkerDataType, checkerReduceOp);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult HcommNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t count, HcclDataType dataType, HcclReduceOp reduceOp)
{
    std::lock_guard<std::mutex> lock(lock_);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // 1.获取当前rankId,NpuPos和stream
    uint32_t curRank = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetCurRank();
    NpuPos pos = HcclSim::SimWorld::Global()->GetNpuPosByRankId(curRank);
    HcclSim::SimStream *stream = reinterpret_cast<HcclSim::SimHcclThread*>(thread)->GetStream();

    // 2.获取远端和本地rankId
    uint32_t locRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLocRankId();
    uint32_t rmtRank = reinterpret_cast<HcclSim::SimChannel*>(channel)->GetRmtRankId();

    // 3.从模型用rankid查询NpuPos，从NpuPos获得SimNpu
    // src地址rank
    HcclSim::SimNpu &srcNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(rmtRank);
    // dst地址rank
    HcclSim::SimNpu &dstNpu = HcclSim::SimWorld::Global()->GetSimNpuByRankId(locRank);

    // 4.查询DataSlice
    // 出参准备
    HcclSim::DataSlice srcSlice;
    HcclSim::DataSlice dstSlice;

    CHK_RET(srcNpu.GetSlice(reinterpret_cast<uint64_t>(src), count, dataType, srcSlice));
    CHK_RET(dstNpu.GetSlice(reinterpret_cast<uint64_t>(dst), count, dataType, dstSlice));

    // 5.datatype转换成checker校验类型
    CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[dataType];
    CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[reduceOp];

    // 6.通过抽象链接类型判断链接协议
    HcclSim::LinkInfo link(reinterpret_cast<HcclSim::SimChannel*>(channel)->GetLinkType());

    // 7.下发task
    auto task = std::make_shared<HcclSim::TaskStubWriteReduce>(rmtRank, link, dstSlice, srcSlice,
        checkerDataType, checkerReduceOp);
    HcclSim::SimTaskQueue::Global()->AppendTask(pos, stream, task);

    return HCCL_SUCCESS;
}

HcclResult CommLocalBareNotifyRecord(ThreadHandle thread, uint64_t dstNotifyId) 
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommLocalBareNotifyWait(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommTaskPrepare(char *key, uint32_t keyLen)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommTaskLaunch(ThreadHandle *threads, uint32_t threadNum)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommWriteReduceWithNotify(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t count, HcclDataType dataType, HcclReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommFence(ThreadHandle thread, ChannelHandle channel)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommSetLaunchMode(int64_t launchId, LaunchMode mode)
{
    HCCL_ERROR("[%s] not support.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

#ifdef __cplusplus
}
#endif  // __cplusplus