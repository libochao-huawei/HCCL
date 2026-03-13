#include "dispatcher.h"
#include "llt_common.h"
#include "checker_data_slice.h"
#include "mem_layout.h"
#include "rank_info_recorder.h"
#include "log.h"
#include "utils_stub.h"
#include "task_stub.h"
#include "task_queue_stub.h"
#include "checker_def.h"
#include "transformer.h"

namespace hccl {
Dispatcher::Dispatcher(
    DispatcherType type, const s32 deviceLogicId)
{
    return;
}

Dispatcher::~Dispatcher()
{
    return;
}

HcclResult Dispatcher::Init()
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Dispatcher::SetNotifyWaitMode(SyncMode notifyWaitMode)
{
    return HcclResult::HCCL_SUCCESS;
}

SyncMode Dispatcher::GetNotifyWaitMode()
{
    return SyncMode::DEFAULT_TIMEWAITSYNCMODE;
}

HcclResult Dispatcher::MemcpySync(void *dst, uint64_t destMax, const void *src, uint64_t count,
    HcclRtMemcpyKind kind)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult Dispatcher::MemcpyAsync(void *dst, uint64_t destMax, const void *src, u64 count,
    HcclRtMemcpyKind kind, hccl::Stream &stream, u32 remoteUserRank,
    hccl::LinkType inLinkType)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult Dispatcher::MemcpyAsync(hccl::HostMem &dst, const hccl::DeviceMem &src, hccl::Stream &stream)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult Dispatcher::MemcpyAsync(hccl::HostMem &dst, const hccl::HostMem &src, hccl::Stream &stream)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult Dispatcher::MemcpyAsync(hccl::DeviceMem &dst, const hccl::HostMem &src, hccl::Stream &stream)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult Dispatcher::MemcpyAsync(hccl::DeviceMem &dst, const hccl::DeviceMem &src, hccl::Stream &stream,
    u32 remoteUserRank, hccl::LinkType inLinkType)
{
    RankId srcRank;
    DataSlice srcSlice;
    RankId dstRank;
    DataSlice dstSlice;

    CHK_RET(MemLayout::Global()->GetSlice(src, srcSlice, &srcRank));
    CHK_RET(MemLayout::Global()->GetSlice(dst, dstSlice, &dstRank));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET(CheckCurRankId(curRank, srcRank, dstRank));

    // 忽略空拷贝操作
    if (srcRank == dstRank && srcSlice.GetSize() == 0 && dstSlice.GetSize() == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    LinkInfoStub link(LinkProtoStub::SDMA);
    std::shared_ptr<TaskStub> task = nullptr;
    if (srcRank == dstRank) { // 本地拷贝的场景
        task.reset(new TaskStubLocalCopy(srcSlice, dstSlice));
    } else if (curRank == srcRank) {  // 写操作
        task.reset(new TaskStubWrite(dstRank, link, srcSlice, dstSlice));
    } else if (curRank == dstRank) { // 读操作
        task.reset(new TaskStubRead(srcRank, link, dstSlice, srcSlice));
    }
    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Dispatcher::InlineReduceAsync(const void *src, u64 count, const HcclDataType datatype, HcclReduceOp redOp,
    Stream& stream, void *dst, u32 remoteUserRank,
    hccl::LinkType inLinkType)
{
    RankId srcRank;
    DataSlice srcSlice;
    RankId dstRank;
    DataSlice dstSlice;

    checker::CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[datatype];
    checker::CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[redOp];

    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, count, datatype, srcSlice, &srcRank));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)dst, count, datatype, dstSlice, &dstRank));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET(CheckCurRankId(curRank, srcRank, dstRank));

    LinkInfoStub link(LinkProtoStub::SDMA);
    std::shared_ptr<TaskStub> task = nullptr;
    if (srcRank == dstRank) { // 本地reduce的场景
        task.reset(new TaskStubLocalReduce(srcSlice, dstSlice, checkerDataType, checkerReduceOp));
    } else if (curRank == srcRank) {  // 写reduce操作
        task.reset(new TaskStubWriteReduce(dstRank, link, srcSlice, dstSlice, checkerDataType, checkerReduceOp));
    } else if (curRank == dstRank) { // 读reduce操作
        task.reset(new TaskStubReadReduce(srcRank, link, dstSlice, srcSlice, checkerDataType, checkerReduceOp));
    }
    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Dispatcher::ReduceAsync(const void *src, u64 dataCount, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream, void *dst, const u32 remoteUserRank, const hccl::LinkType linkType,
        const u64 reduceAttr)
{
    RankId srcRank;
    DataSlice srcSlice;
    RankId dstRank;
    DataSlice dstSlice;

    checker::CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[datatype];
    checker::CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[redOp];

    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, dataCount, datatype, srcSlice, &srcRank));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)dst, dataCount, datatype, dstSlice, &dstRank));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET(CheckCurRankId(curRank, srcRank, dstRank));

    LinkInfoStub link(LinkProtoStub::SDMA);
    if (srcRank == dstRank) {
        std::shared_ptr<TaskStub> task(new TaskStubLocalReduce(srcSlice, dstSlice, checkerDataType, checkerReduceOp));
        TaskQueueStub::AppendTask(curRank, &stream, task);
    } else if (curRank == srcRank) {
        std::shared_ptr<TaskStub> task(new TaskStubWriteReduce(dstRank, link, srcSlice, dstSlice, checkerDataType, checkerReduceOp));
        TaskQueueStub::AppendTask(curRank, &stream, task);
    } else if (curRank == dstRank) {
        std::shared_ptr<TaskStub> task(new TaskStubReadReduce(srcRank, link, dstSlice, srcSlice, checkerDataType, checkerReduceOp));
        TaskQueueStub::AppendTask(curRank, &stream, task);
    }
    return HcclResult::HCCL_SUCCESS;
}

// 应该没有算法编排逻辑是直接使用这个接口
HcclResult Dispatcher::SignalRecord(HcclRtSignal signal, hccl::Stream &stream, u32 userRank, u64 offset,
    s32 stage, bool inchip, u64 signalAddr)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

// 应该没有算法编排逻辑是直接使用这个接口
HcclResult Dispatcher::SignalWait(HcclRtSignal signal, hccl::Stream &stream, u32 userRank, u32 remoteUserRank,
    s32 stage, bool inchip, u32 timeOut)
{
    HCCL_ERROR("Not implemented stub API.");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult Dispatcher::LaunchFftsTask(Stream &stream)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Dispatcher::ResetFftsCtx(bool enableCache, const std::string &key)
{
    return HCCL_SUCCESS;
}

void Dispatcher::JudgeFftsCtxInitialized(bool &fftsCtxInitFlag)
{
    return;
}

HcclResult Dispatcher::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
{
    return HCCL_SUCCESS;
}

HcclResult Dispatcher::SetQosCfg(const u32 qosCfg)
{
    qosCfg_ = qosCfg;
    return HCCL_SUCCESS;
}

HcclResult Dispatcher::ResetQosCfg()
{
    qosCfg_ = INVALID_QOSCFG;
    return HCCL_SUCCESS;
}

HcclResult Dispatcher::GetQosCfg(u32& qosCfg)
{
    qosCfg = qosCfg_;
    return HCCL_SUCCESS;
}

HcclResult Dispatcher::WaitValue(hccl::Stream &stream, u64 waitAddr, u64 valueAddr, bool reset)
{
    return HCCL_SUCCESS;
}
HcclResult Dispatcher::WriteValue(hccl::Stream &stream, u64 writeAddr, u64 valueAddr)
{
    return HCCL_SUCCESS;
}

}
