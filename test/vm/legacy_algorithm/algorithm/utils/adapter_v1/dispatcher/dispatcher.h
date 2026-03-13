#ifndef HCCL_INC_DISPATCHER_H
#define HCCL_INC_DISPATCHER_H

#include "hccl_types.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "hccl_common.h"
#include "task_stub.h"
#include "task_queue_stub.h"
#include "rank_info_recorder.h"
#include "dispatcher_task_types.h"

enum class DispatcherType {
    DISPATCHER_NORMAL = 0,
    DISPATCHER_FFTS,
    DISPATCHER_VIRTURAL,
};

constexpr u64 ATTR_POS_INLINE_REDUCE = 0x00;
constexpr u64 ATTR_POS_SUPPORT_RDMA = 0x01;
constexpr u64 ATTR_POS_SUPPORT_RDMA_ASYNC = 0x02;
constexpr u64 ATTR_POS_SUPPORT_RDMA_REDUCE = 0x03;
constexpr u64 INLINE_REDUCE_BITMASK = 0x01;
constexpr u64 RDMA_REDUCE_BITMASK = 0x08;
constexpr u64 INLINE_REDUCE_BIT = 0x01;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
    HcclResult HcclDispatcherInit(DispatcherType type, const s32 deviceLogicId, HcclDispatcher *dispatcher);

    HcclResult HcclDispatcherDestroy(HcclDispatcher dispatcher);

    HcclResult HcclSetGlobalWorkSpace(HcclDispatcher dispatcherPtr, std::vector<void *> &globalWorkSpaceAddr);

    HcclResult HcclSetQosCfg(HcclDispatcher dispatcherPtr, const u32 qosCfg);
    HcclResult HcclResetQosCfg(HcclDispatcher dispatcherPtr);
    HcclResult HcclGetQosCfg(HcclDispatcher dispatcherPtr, u32 *qosCfg);

    HcclResult HcclSetNotifyWaitMode(HcclDispatcher dispatcherPtr, const SyncMode notifyWaitMode);
    HcclResult HcclGetNotifyWaitMode(HcclDispatcher dispatcherPtr, SyncMode *notifyWaitMode);

    HcclResult HcclD2DMemcpyAsync(HcclDispatcher dispatcherPtr, hccl::DeviceMem &dst, const hccl::DeviceMem &src,
        hccl::Stream &stream, u32 remoteUserRank = INVALID_VALUE_RANKID,
        hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);

    HcclResult HcclMemcpyAsync(HcclDispatcher dispatcherPtr, void *dst, const uint64_t destMax, const void *src,
        const uint64_t count, const HcclRtMemcpyKind kind, hccl::Stream &stream, const u32 remoteUserRank,
        hccl::LinkType linkType);

    HcclResult HcclReduceAsync(HcclDispatcher dispatcherPtr, void *src, uint64_t count,
        const HcclDataType datatype, const HcclReduceOp reduceOp, hccl::Stream &stream, void *dst,
        const u32 remoteUserRank, const hccl::LinkType linkType, const u64 reduceAttr);

    HcclResult HcclDispatcherWaitValue(HcclDispatcher dispatcherPtr,
        hccl::Stream &stream, u64 waitAddr, u64 valueAddr, bool reset);
    HcclResult HcclDispatcherWriteValue(HcclDispatcher dispatcherPtr,
        hccl::Stream &stream, u64 writeAddr, u64 valueAddr);

    // 仅限task loader使用
    HcclResult HcclSignalRecord(HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream &stream,
        u32 userRank, u64 offset, s32 stage, bool inchip, u64 signalAddr);
    HcclResult HcclSignalWait(HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream &stream,
        u32 userRank, u32 remoteUserRank, s32 stage, bool inchip);

    HcclResult LaunchTask(HcclDispatcher dispatcherPtr, hccl::Stream &stream);
    HcclResult LaunchTaskExtend(
        HcclDispatcher dispatcherPtr, hccl::Stream &stream, std::vector<hccl::Stream> &subStreams);
    HcclResult InitTask(HcclDispatcher dispatcherPtr, hccl::Stream &stream, const bool enableCache, const std::string &key);
    HcclResult IsCtxInitialized(HcclDispatcher dispatcherPtr, bool *ctxInitFlag);
    HcclResult SetNormalMode(HcclDispatcher dispatcherPtr);
#ifdef __cplusplus
}
#endif // __cplusplus

namespace hccl {

using HcclRtSignal = void *;


class Dispatcher {
public:
    explicit Dispatcher(
        DispatcherType type, const s32 deviceLogicId);
    ~Dispatcher();

    HcclResult Init();  // 初始化必要信息

    HcclResult SetNotifyWaitMode(SyncMode notifyWaitMode);
    SyncMode GetNotifyWaitMode();

    // 算法下发task时，不要使用HcclRtStream参数类型接口，需要改为hccl::Stream参数类型的接口
    HcclResult MemcpySync(void *dst, uint64_t destMax, const void *src, uint64_t count,
        HcclRtMemcpyKind kind);
    HcclResult MemcpyAsync(void *dst, uint64_t destMax, const void *src, u64 count,
        HcclRtMemcpyKind kind, hccl::Stream &stream, u32 remoteUserRank = INVALID_VALUE_RANKID,
        hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);

    HcclResult MemcpyAsync(hccl::HostMem &dst, const hccl::DeviceMem &src, hccl::Stream &stream);
    HcclResult MemcpyAsync(hccl::HostMem &dst, const hccl::HostMem &src, hccl::Stream &stream);
    HcclResult MemcpyAsync(hccl::DeviceMem &dst, const hccl::HostMem &src, hccl::Stream &stream);

    HcclResult MemcpyAsync(hccl::DeviceMem &dst, const hccl::DeviceMem &src, hccl::Stream &stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);

    HcclResult InlineReduceAsync(const void *src, u64 count, const HcclDataType datatype, HcclReduceOp redOp,
        Stream& stream, void *dst, u32 remoteUserRank = INVALID_VALUE_RANKID,
        hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);
    HcclResult ReduceAsync(const void *src, u64 dataCount, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream, void *dst, const u32 remoteUserRank, const hccl::LinkType linkType,
        const u64 reduceAttr);

    HcclResult SignalRecord(HcclRtSignal signal, hccl::Stream &stream, u32 userRank, u64 offset = INVALID_U64,
        s32 stage = INVALID_VALUE_STAGE, bool inchip = false, u64 signalAddr = INVALID_U64);
    HcclResult SignalWait(HcclRtSignal signal, hccl::Stream &stream, u32 userRank, u32 remoteUserRank,
        s32 stage = INVALID_VALUE_STAGE, bool inchip = false, u32 timeOut = NOTIFY_INVALID_WAIT_TIME);

    HcclResult LaunchFftsTask(Stream &stream);
    HcclResult ResetFftsCtx(bool enableCache, const std::string &key);
    void JudgeFftsCtxInitialized(bool &fftsCtxInitFlag);

    HcclResult SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr);
    HcclResult SetQosCfg(const u32 qosCfg);
    HcclResult ResetQosCfg();
    HcclResult GetQosCfg(u32& qosCfg);

    u32 qosCfg_ = INVALID_QOSCFG;

    virtual HcclResult WaitValue(hccl::Stream &stream, u64 waitAddr, u64 valueAddr, bool reset);
    virtual HcclResult WriteValue(hccl::Stream &stream, u64 writeAddr, u64 valueAddr);
};
}
#endif