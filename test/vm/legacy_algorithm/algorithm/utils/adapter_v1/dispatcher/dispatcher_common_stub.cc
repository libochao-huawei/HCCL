/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: dispatcher common api
 * Author: yanyefeng
 * Create: 2023-11-14
 */
#include "dispatcher.h"
#include "externalinput_pub.h"

using namespace hccl;

HcclResult HcclDispatcherInit(DispatcherType type, const s32 devicePhyId,
    HcclDispatcher *dispatcher)
{
    CHK_PTR_NULL(dispatcher);

    auto pDispatcher = std::make_unique<Dispatcher>(type, devicePhyId);
    CHK_RET(pDispatcher->Init());

    *dispatcher = pDispatcher.release();
    return HCCL_SUCCESS;
}

HcclResult HcclDispatcherDestroy(HcclDispatcher dispatcherPtr)
{
    if (dispatcherPtr != nullptr) {
        delete reinterpret_cast<Dispatcher*>(dispatcherPtr);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclSetGlobalWorkSpace(HcclDispatcher dispatcherPtr, std::vector<void *> &globalWorkSpaceAddr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->SetGlobalWorkSpace(globalWorkSpaceAddr);
}

HcclResult HcclSetQosCfg(HcclDispatcher dispatcherPtr, const u32 qosCfg)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->SetQosCfg(qosCfg);
}

HcclResult HcclResetQosCfg(HcclDispatcher dispatcherPtr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->ResetQosCfg();
}

HcclResult HcclGetQosCfg(HcclDispatcher dispatcherPtr, u32 *qosCfg)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(qosCfg);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->GetQosCfg(*qosCfg);
}

HcclResult HcclSetNotifyWaitMode(HcclDispatcher dispatcherPtr, const SyncMode notifyWaitMode)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->SetNotifyWaitMode(notifyWaitMode);
}

HcclResult HcclGetNotifyWaitMode(HcclDispatcher dispatcherPtr, SyncMode *notifyWaitMode)
{
    CHK_PTR_NULL(dispatcherPtr);
    CHK_PTR_NULL(notifyWaitMode);

    *notifyWaitMode = reinterpret_cast<Dispatcher*>(dispatcherPtr)->GetNotifyWaitMode();
    return HCCL_SUCCESS;
}

HcclResult HcclD2DMemcpyAsync(HcclDispatcher dispatcherPtr, DeviceMem &dst, const DeviceMem &src,
    Stream &stream, const u32 remoteUserRank, const LinkType linkType)
{
    CHK_PTR_NULL(dispatcherPtr);
    // 如果源和目的相等的话，给出告警
    if (dst == src) {
        if (dst.size() != 0 || src.size() != 0) {
            HCCL_WARNING("dst and src are same, dst[ptr=%p, size=%llu], src[ptr=%p, size=%llu]",
                dst.ptr(), dst.size(), src.ptr(), src.size());
            return HCCL_SUCCESS;
        }
    }
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->MemcpyAsync(dst, src, stream, remoteUserRank, linkType);
}

HcclResult HcclMemcpyAsync(HcclDispatcher dispatcherPtr, void *dst, const uint64_t destMax, const void *src,
    uint64_t count, const HcclRtMemcpyKind kind, Stream &stream, const u32 remoteUserRank,
    const LinkType linkType)
{
    CHK_PTR_NULL(dispatcherPtr);

    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->MemcpyAsync(dst, destMax, src, count, kind,
        stream, remoteUserRank, linkType);
}

HcclResult HcclReduceAsync(HcclDispatcher dispatcherPtr, void *src, uint64_t count, const HcclDataType datatype,
    const HcclReduceOp reduceOp, Stream &stream, void *dst, const u32 remoteUserRank,
    const LinkType linkType, const u64 reduceAttr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->ReduceAsync(src, count, datatype, reduceOp,
        stream, dst, remoteUserRank, linkType, reduceAttr);
}

HcclResult HcclDispatcherWaitValue(HcclDispatcher dispatcherPtr, hccl::Stream &stream, u64 waitAddr, u64 valueAddr, bool reset)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->WaitValue(stream, waitAddr, valueAddr, reset);
}
HcclResult HcclDispatcherWriteValue(HcclDispatcher dispatcherPtr, hccl::Stream &stream, u64 writeAddr, u64 valueAddr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->WriteValue(stream, writeAddr, valueAddr);
}

HcclResult HcclSignalRecord(HcclDispatcher dispatcherPtr, HcclRtNotify signal, Stream &stream, u32 userRank,
    u64 offset, s32 stage, bool inchip, u64 signalAddr)
{
    CHK_PTR_NULL(dispatcherPtr);

    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->SignalRecord(signal, stream, userRank,
        offset, stage, inchip, signalAddr);
}
HcclResult HcclSignalWait(HcclDispatcher dispatcherPtr, HcclRtNotify signal, Stream &stream, u32 userRank,
    u32 remoteUserRank, s32 stage, bool inchip)
{
    CHK_PTR_NULL(dispatcherPtr);

    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->SignalWait(signal, stream,
        userRank, remoteUserRank, stage, inchip);
}

HcclResult LaunchTask(HcclDispatcher dispatcherPtr, Stream &stream)
{
    CHK_PTR_NULL(dispatcherPtr);

    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->LaunchFftsTask(stream);
}

HcclResult LaunchTaskExtend(HcclDispatcher dispatcherPtr, Stream &stream, std::vector<Stream> &subStreams)
{
    CHK_PTR_NULL(dispatcherPtr);

    return reinterpret_cast<Dispatcher*>(dispatcherPtr)->LaunchFftsTask(stream);
}

HcclResult InitTask(HcclDispatcher dispatcherPtr, hccl::Stream &stream, const bool enableCache, const std::string &key)
{
    CHK_PTR_NULL(dispatcherPtr);

    CHK_RET(reinterpret_cast<Dispatcher*>(dispatcherPtr)->ResetFftsCtx(enableCache, key));
    return  HCCL_SUCCESS;
}

HcclResult IsCtxInitialized(HcclDispatcher dispatcherPtr, bool *fftsCtxInitFlag)
{
    CHK_PTR_NULL(dispatcherPtr);

    reinterpret_cast<Dispatcher*>(dispatcherPtr)->JudgeFftsCtxInitialized(*fftsCtxInitFlag);
    return HCCL_SUCCESS;
}

HcclResult SetNormalMode(HcclDispatcher dispatcherPtr)
{
    CHK_PTR_NULL(dispatcherPtr);
    return HCCL_SUCCESS;
}

