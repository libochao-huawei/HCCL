
#include "transport_pub.h"
#include "checker_buffer_type.h"
#include "coll_alg_param.h"
#include "checker.h"
#include "mem_layout.h"
#include "utils_stub.h"
#include "transport.h"
#include "task_stub.h"
#include "task_queue_stub.h"
#include "rank_info_recorder.h"
#include "device_info_recorder.h"
#include "transformer.h"
#include "orchestrate.h"

namespace hccl {

std::map<TransportMemType, BufferType> transportMem2Buffer = {
    {TransportMemType::CCL_INPUT, BufferType::INPUT_CCL},
    {TransportMemType::CCL_OUTPUT, BufferType::OUTPUT_CCL},
    {TransportMemType::SCRATCH, BufferType::SCRATCH},
    {TransportMemType::PARAM_INPUT, BufferType::INPUT},
    {TransportMemType::PARAM_OUTPUT, BufferType::OUTPUT},
    {TransportMemType::AIV_INPUT, BufferType::INPUT_AIV},
    {TransportMemType::AIV_OUTPUT, BufferType::OUTPUT_AIV},
    {TransportMemType::RESERVED, BufferType::RESERVED}
    };

HcclResult GenLinkInfo(TransportType transportType, LinkInfoStub &linkInfo)
{
    if (transportType == TransportType::TRANS_TYPE_P2P) {
        linkInfo.linkProto = LinkProtoStub::SDMA;
    } else if (transportType == TransportType::TRANS_TYPE_IBV_EXP) {
        linkInfo.linkProto = LinkProtoStub::RDMA;
    } else {
        HCCL_ERROR("[GenLinkInfo] this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult GetRemoteBufferType(Transport* transport, UserMemType memType, BufferType &bufferType)
{
    if (memType == UserMemType::INPUT_MEM) {
        bufferType = transportMem2Buffer[links2TransportCompare_[transport]->remoteinputMemType];
    } else if (memType == UserMemType::OUTPUT_MEM) {
        bufferType = transportMem2Buffer[links2TransportCompare_[transport]->remoteoutputMemType];
    } else {
        HCCL_ERROR("memType is error");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

Transport::Transport(TransportType type, TransportPara& para, MachinePara &machinePara, LinkType linkType)
{
    transportType_ = type;
    machinePara_ = machinePara;
    linkType_ = linkType;
    return;
}

Transport::Transport(TransportType type, TransportPara& para,
              const HcclDispatcher dispatcher,
              const std::unique_ptr<NotifyPool> &notifyPool,
              MachinePara &machinePara)
{
    transportType_ = type;
    machinePara_ = machinePara;
    return;
}

Transport::~Transport()
{ }

HcclResult Transport::Init()
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::DeInit()
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxDataSignal(Stream &stream)
{
    // 同步量p2p与TxAck相同，ibverbs与PostFin相同
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXDATASIGNAL"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxDataSignal(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "RXDATASIGNAL"));
    TaskQueueStub::AppendTask(curRank, &stream, taskwait);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxAsync(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        LinkInfoStub link(LinkProtoStub::SDMA);
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        DataSlice srcSlice;
        DataSlice dstSlice;
        BufferType bufferType;
        if (dstMemType == UserMemType::INPUT_MEM) {
            bufferType = transportMem2Buffer[links2TransportCompare_[this]->remoteinputMemType];
        } else if (dstMemType == UserMemType::OUTPUT_MEM) {
            bufferType = transportMem2Buffer[links2TransportCompare_[this]->remoteoutputMemType];
        } else {
            HCCL_ERROR("srcMemType is error");
            return HCCL_E_NOT_SUPPORT;
        }
        LinkInfoStub link(LinkProtoStub::RDMA);
        if (src != nullptr) {
            CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, len, srcSlice));
            CHK_RET(MemLayout::Global()->GetSlice(bufferType, dstOffset, len, dstSlice));
            std::shared_ptr<TaskStub> taskwrite(new TaskStubWrite(remoteRank, link, srcSlice, dstSlice));
            TaskQueueStub::AppendTask(curRank, &stream, taskwrite);
        }

        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        LinkInfoStub link(LinkProtoStub::SDMA);
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        LinkInfoStub link(LinkProtoStub::RDMA);
        DataSlice srcSlice;
        DataSlice dstSlice;
        for(int i = 0; i < txMems.size(); i++) {
            const void *src = txMems[i].src;
            if (src == nullptr) {
                continue;
            }
            u64 dstOffset = txMems[i].dstOffset;
            u64 len = txMems[i].len;
            BufferType bufferType;
            UserMemType dstMemType = txMems[i].dstMemType;
            if (dstMemType == UserMemType::INPUT_MEM) {
                bufferType = transportMem2Buffer[links2TransportCompare_[this]->remoteinputMemType];
            } else if (dstMemType == UserMemType::OUTPUT_MEM) {
                bufferType = transportMem2Buffer[links2TransportCompare_[this]->remoteoutputMemType];
            } else {
                HCCL_ERROR("srcMemType is error");
                return HCCL_E_NOT_SUPPORT;
            }

            CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, len, srcSlice));
            CHK_RET(MemLayout::Global()->GetSlice(bufferType, dstOffset, len, dstSlice));
            std::shared_ptr<TaskStub> taskwrite(new TaskStubWrite(remoteRank, link, srcSlice, dstSlice));
            TaskQueueStub::AppendTask(curRank, &stream, taskwrite);
        }
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxData(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)
{
    if (src == nullptr) {
        HCCL_ERROR("src is null, do not need to tx");
        return HCCL_SUCCESS;
    }

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        DataSlice srcSlice;
        DataSlice dstSlice;
        BufferType bufferType;

        CHK_RET(GetRemoteBufferType(this, dstMemType, bufferType));
        LinkInfoStub link(LinkProtoStub::RDMA);

        CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, len, srcSlice));
        CHK_RET(MemLayout::Global()->GetSlice(bufferType, dstOffset, len, dstSlice));
        std::shared_ptr<TaskStub> taskwrite(new TaskStubWrite(remoteRank, link, srcSlice, dstSlice));
        TaskQueueStub::AppendTask(curRank, &stream, taskwrite);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxData(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)
{
    if (dst == nullptr) {
        HCCL_ERROR("dst is null, do not need to rx");
        return HCCL_SUCCESS;
    }

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        DataSlice srcSlice;
        DataSlice dstSlice;
        BufferType bufferType;

        CHK_RET(GetRemoteBufferType(this, srcMemType, bufferType));
        LinkInfoStub link(LinkProtoStub::SDMA);

        CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)dst, len, dstSlice));
        CHK_RET(MemLayout::Global()->GetSlice(bufferType, srcOffset, len, srcSlice));
        std::shared_ptr<TaskStub> taskresd(new TaskStubRead(remoteRank, link, dstSlice, srcSlice));
        TaskQueueStub::AppendTask(curRank, &stream, taskresd);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxPrepare(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::READY, "TXPREPARE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::READY, "TXPREPARE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxPrepare(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::READY, "RXPREPARE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::READY, "RXPREPARE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxDone(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "TXDONE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXDONE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);

        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN_ACK, "TXDONE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxDone(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));

    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "RXDONE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "RXDONE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);

        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN_ACK, "RXDONE"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

bool Transport::IsValid()
{
    auto compared = links2TransportCompare_.find(this);
    if (compared == links2TransportCompare_.end()) {
        // 模拟pimpl_ == nullptr的情况
        HCCL_ERROR("There is no such a link.");
        return false;
    } else {
        return links2TransportCompare_[this]->isValid;
    }
}


// 支持RDMA Reduce的场景
HcclResult Transport::TxWithReduce(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len,
    const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)
{
    if ((this->transportType_ == TransportType::TRANS_TYPE_P2P)) {
        // SDMA场景下实现为空
        return HCCL_SUCCESS;
    }
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    LinkInfoStub link(LinkProtoStub::RDMA);
    DataSlice srcSlice;
    DataSlice dstSlice;
    u32 unitSize = SIZE_TABLE[datatype];
    u64 dataCount = len / unitSize;

    BufferType bufferType;
    CHK_RET(GetRemoteBufferType(this, dstMemType, bufferType));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, dataCount, datatype, srcSlice));
    CHK_RET(MemLayout::Global()->GetSlice(bufferType, dstOffset, len, dstSlice));

    checker::CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[datatype];
    checker::CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[redOp];

    std::shared_ptr<TaskStub> taskwritereduce(new TaskStubWriteReduce(remoteRank, link, srcSlice, dstSlice, checkerDataType, checkerReduceOp));
    TaskQueueStub::AppendTask(curRank, &stream, taskwritereduce);
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXWITHREDUCE"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxWithReduce(const std::vector<TxMemoryInfo> &txWithReduceMems,
    const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)
{
    // 只有RDMA场景下有
    if (txWithReduceMems.size() == 0) {
        HCCL_ERROR("txWithReduceMems is null");
        return HCCL_E_PARA;
    }
    if ((this->transportType_ == TransportType::TRANS_TYPE_P2P)) {
        // SDMA场景下实现为空
        return HCCL_SUCCESS;
    }
    LinkInfoStub link(LinkProtoStub::RDMA);
    DataSlice srcSlice;
    DataSlice dstSlice;
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    checker::CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[datatype];
    checker::CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[redOp];

    for(int i = 0; i < txWithReduceMems.size(); i++) {
        u32 unitSize = SIZE_TABLE[datatype];
        const void *src = txWithReduceMems[i].src;
        u64 dstOffset = txWithReduceMems[i].dstOffset;
        u64 len = txWithReduceMems[i].len;
        u64 dataCount = len / unitSize;
        UserMemType dstMemType = txWithReduceMems[i].dstMemType;
        BufferType bufferType;
        CHK_RET(GetRemoteBufferType(this, dstMemType, bufferType));
        CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)src, dataCount, datatype, srcSlice));
        CHK_RET(MemLayout::Global()->GetSlice(bufferType, dstOffset, len, dstSlice));
        std::shared_ptr<TaskStub> taskwritereduce(new TaskStubWriteReduce(remoteRank, link, srcSlice, dstSlice, checkerDataType, checkerReduceOp));
        TaskQueueStub::AppendTask(curRank, &stream, taskwritereduce);
    }
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "TXWITHREDUCE"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

bool Transport::IsSupportTransportWithReduce()
{
    DevType devType = g_CheckerDevType2HcclDevType[DeviceInfoRecorder::Global()->rankId2devType[this->machinePara_.localUserrank]];
    if (this->linkType_ == LinkType::LINK_ROCE) {
        if (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_910_93) {
            return true;
        } else {
            return false;
        }
    }
    return false;
}

HcclResult Transport::RxAsync(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        DataSlice srcSlice;
        DataSlice dstSlice;
        BufferType bufferType;

        if (srcMemType == UserMemType::INPUT_MEM) {
            bufferType = transportMem2Buffer[links2TransportCompare_[this]->remoteinputMemType];
        } else if (srcMemType == UserMemType::OUTPUT_MEM) {
            bufferType = transportMem2Buffer[links2TransportCompare_[this]->remoteoutputMemType];
        } else {
            HCCL_ERROR("srcMemType is error");
            return HCCL_E_NOT_SUPPORT;
        }
        LinkInfoStub link(LinkProtoStub::SDMA);
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "RXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
        if (dst != nullptr) {
            CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)dst, len, dstSlice));
            CHK_RET(MemLayout::Global()->GetSlice(bufferType, srcOffset, len, srcSlice));
            std::shared_ptr<TaskStub> taskresd(new TaskStubRead(remoteRank, link, dstSlice, srcSlice));
            TaskQueueStub::AppendTask(curRank, &stream, taskresd);
        }
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        LinkInfoStub link(LinkProtoStub::RDMA);
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "RXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    if (rxMems.size() == 0) {
        HCCL_ERROR("TxMemoryInfo is a null vector");
        return HCCL_E_PARA;
    }

    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        DataSlice srcSlice;
        DataSlice dstSlice;
        BufferType bufferType;

        LinkInfoStub link(LinkProtoStub::SDMA);
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "RXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);

        for(int i = 0; i < rxMems.size(); i++) {
            const void *dst = rxMems[i].dst;
            if (dst == nullptr) {
                continue;
            }
            u64 srcOffset = rxMems[i].srcOffset;
            u64 len = rxMems[i].len;
            UserMemType memType = rxMems[i].srcMemType;
            // 假设本端和远端的inputmemtype是一样的
            CHK_RET(GetRemoteBufferType(this, memType, bufferType));
            CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)dst, len, dstSlice));
            CHK_RET(MemLayout::Global()->GetSlice(bufferType, srcOffset, len, srcSlice));
            std::shared_ptr<TaskStub> taskresd(new TaskStubRead(remoteRank, link, dstSlice, srcSlice));
            TaskQueueStub::AppendTask(curRank, &stream, taskresd);
        }
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        LinkInfoStub link(LinkProtoStub::RDMA);
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "RXASYNC"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else {
        HCCL_ERROR("this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxAck(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::READY, "TXACK"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxAck(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::READY, "RXACK"));
    TaskQueueStub::AppendTask(curRank, &stream, taskwait);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::TxWaitDone(Stream &stream)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::RxWaitDone(Stream &stream)
{
    return HcclResult::HCCL_SUCCESS;
}

// 可以通过MemLayout来打桩实现
HcclResult Transport::GetRemoteMem(UserMemType memType, void **remotePtr)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    u32 phyId = RankInfoRecorder::Global()->rankId2phyId[remoteRank];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[remoteRank];
    u32 superpodId = RankInfoRecorder::Global()->rankId2superpodId[remoteRank];

    BufferType bufferType;
    CHK_RET(GetRemoteBufferType(this, memType, bufferType));

    *remotePtr = (void*)MemLayout::Global()->allSuperPodLayout[superpodId][serverId][phyId][bufferType].startAddr;
    return HcclResult::HCCL_SUCCESS;
}

hccl::LinkType Transport::GetLinkType() const
{
    return linkType_;
}

bool Transport::IsSpInlineReduce() const
{
    bool isSpInlineReduce = linkType_ == LinkType::LINK_HCCS ||
                            linkType_ == LinkType::LINK_PCIE ||
                            linkType_ == LinkType::LINK_SIO ||
                            linkType_ == LinkType::LINK_HCCS_SW;
    return isSpInlineReduce;
}

HcclResult Transport::Write(const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    BufferType bufferType;
    CHK_RET(GetRemoteBufferType(this, remoteMemType, bufferType));

    DataSlice srcSlice;
    DataSlice dstSlice;

    RankId dstRank;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localAddr, len, srcSlice));
    CHK_RET(MemLayout::Global()->GetSlice(bufferType, remoteOffset, len, dstSlice));

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));

    std::shared_ptr<TaskStub> task(new TaskStubWrite(remoteRank, link, srcSlice, dstSlice));
    TaskQueueStub::AppendTask(curRank, &stream, task);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::Read(const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;
    BufferType bufferType;
    CHK_RET(GetRemoteBufferType(this, remoteMemType, bufferType));

    DataSlice srcSlice;
    DataSlice dstSlice;

    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localAddr, len, srcSlice));
    CHK_RET(MemLayout::Global()->GetSlice(bufferType, remoteOffset, len, dstSlice));

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> task(new TaskStubRead(remoteRank, link, srcSlice, dstSlice));
    TaskQueueStub::AppendTask(curRank, &stream, task);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::PostReady(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::READY, "POSTREADY"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::WaitReady(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::READY, "WAITREADY"));
    TaskQueueStub::AppendTask(curRank, &stream, taskwait);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::PostFin(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "POSTFIN"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::WaitFin(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "WAITFIN"));
    TaskQueueStub::AppendTask(curRank, &stream, taskwait);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::PostFinAck(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN_ACK, "POSTFINACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);
    } else {
        // 实际代码中SDMA场景下为空实现
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::WaitFinAck(Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN_ACK, "WAITFINACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else {
        // 实际代码中SDMA场景下为空实现
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::ConnectAsync(u32& status)
{
    return HCCL_SUCCESS;
}

HcclResult Transport::ConnectQuerry(u32& status)
{
    return HCCL_SUCCESS;
}

HcclResult Transport::RxWithReduce(
    UserMemType recvSrcMemType, u64 recvSrcOffset, void *recvDst, u64 recvLen,
    void *reduceSrc, void *reduceDst, u64 reduceDataCount, HcclDataType reduceDatatype,
    HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr)
{
    return HCCL_SUCCESS;
}

HcclResult Transport::RxWithReduce(
    const std::vector<RxWithReduceMemoryInfo> &rxWithReduceMems,
    HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream,
    const u64 reduceAttr)
{
    return HCCL_SUCCESS;
}

u32 Transport::GetRemoteRank()
{
    return machinePara_.remoteUserrank;
}

bool Transport::IsTransportRoce()
{
    return false;
}

HcclResult Transport::TxEnv(const void *ptr, const u64 len, Stream &stream)
{
    HCCL_ERROR("This task is not mocked.");
    return HCCL_SUCCESS;
}

HcclResult Transport::RxEnv(Stream &stream)
{
    HCCL_ERROR("This task is not mocked.");
    return HCCL_SUCCESS;
}

HcclResult Transport::DataReceivedAck(Stream &stream)
{
    if (this->transportType_ == TransportType::TRANS_TYPE_P2P) {
        RankId curRank = RankInfoRecorder::Global()->GetRankId();
        RankId remoteRank = links2TransportCompare_[this]->remoteRank;
        LinkInfoStub link(LinkProtoStub::SDMA);

        std::shared_ptr<TaskStub> taskpost1(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::READY, "DATARECEIVEDACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost1);

        std::shared_ptr<TaskStub> taskwait1(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::READY, "DATARECEIVEDACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait1);

        std::shared_ptr<TaskStub> taskpost2(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN, "DATARECEIVEDACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost2);

        std::shared_ptr<TaskStub> taskwait2(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN, "DATARECEIVEDACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait2);
    } else if (this->transportType_ == TransportType::TRANS_TYPE_IBV_EXP) {
        RankId curRank = RankInfoRecorder::Global()->GetRankId();
        RankId remoteRank = links2TransportCompare_[this]->remoteRank;
        LinkInfoStub link(LinkProtoStub::RDMA);

        std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, 0, NotifyTypeStub::FIN_ACK, "DATARECEIVEDACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskpost);

        std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, 0, NotifyTypeStub::FIN_ACK, "DATARECEIVEDACK"));
        TaskQueueStub::AppendTask(curRank, &stream, taskwait);
    } else {
        HCCL_ERROR("[GenLinkInfo] this transportType not support");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

bool Transport::GetSupportDataReceivedAck() const
{
    return machinePara_.supportDataReceivedAck;
}

HcclResult Transport::ReadSync(struct Transport::Buffer &localBuf, struct Transport::Buffer &remoteBuf, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    DataSlice srcSlice;
    DataSlice dstSlice;

    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)remoteBuf.addr, remoteBuf.size, srcSlice));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localBuf.addr, localBuf.size, dstSlice));

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> task(new TaskStubRead(remoteRank, link, dstSlice, srcSlice));
    TaskQueueStub::AppendTask(curRank, &stream, task);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::ReadAsync(struct Transport::Buffer &localBuf, struct Transport::Buffer &remoteBuf, Stream &stream)
{
    CHK_RET(ReadSync(localBuf, remoteBuf, stream));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::WriteSync(struct Transport::Buffer &remoteBuf, struct Transport::Buffer &localBuf, Stream &stream)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    DataSlice srcSlice;
    DataSlice dstSlice;

    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)remoteBuf.addr, remoteBuf.size, dstSlice));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localBuf.addr, localBuf.size, srcSlice));

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> task(new TaskStubWrite(remoteRank, link, srcSlice, dstSlice));
    TaskQueueStub::AppendTask(curRank, &stream, task);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::WriteAsync(struct Transport::Buffer &remoteBuf, struct Transport::Buffer &localBuf, Stream &stream)
{
    CHK_RET(WriteSync(remoteBuf, localBuf, stream));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::Fence()
{
    return HcclResult::HCCL_SUCCESS;
}


HcclResult Transport::ReadReduceSync(struct Transport::Buffer &localBuf, struct Transport::Buffer &remoteBuf,
        const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)
{
    RankId srcRank;
    DataSlice srcSlice;
    RankId dstRank;
    DataSlice dstSlice;

    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)remoteBuf.addr,
        remoteBuf.size / SIZE_TABLE[datatype], datatype, srcSlice, &srcRank));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localBuf.addr,
        localBuf.size / SIZE_TABLE[datatype], datatype, dstSlice, &dstRank));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET(CheckCurRankId(curRank, srcRank, dstRank));

    LinkInfoStub link(LinkProtoStub::SDMA);
    std::shared_ptr<TaskStub> task = nullptr;
    // 读reduce操作
    checker::CheckerDataType checkerDataType = g_HcclDataType2CheckerDataType[datatype];
    checker::CheckerReduceOp checkerReduceOp = g_HcclReduceOp2CheckerReduceOp[redOp];
    task.reset(new TaskStubReadReduce(srcRank, link, dstSlice, srcSlice, checkerDataType, checkerReduceOp));

    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::Post(u32 notifyIdx, Stream &stream)
{
    if (notifyIdx >= this->machinePara_.notifyNum) {
        HCCL_ERROR("[CreatePostNode] the notifyIdx is bigger than notifyNum");
        return HcclResult::HCCL_E_PARA;
    }
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskpost(new TaskStubPost(remoteRank, link, notifyIdx, NotifyTypeStub::READY, "POST"));
    TaskQueueStub::AppendTask(curRank, &stream, taskpost);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::Wait(u32 notifyIdx, Stream &stream)
{
    if (notifyIdx >= this->machinePara_.notifyNum) {
        HCCL_ERROR("[CreateWaitNode] the notifyIdx is bigger than notifyNum");
        return HcclResult::HCCL_E_PARA;
    }
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    RankId remoteRank = links2TransportCompare_[this]->remoteRank;

    LinkInfoStub link(LinkProtoStub::INVALID_A);
    CHK_RET(GenLinkInfo(transportType_, link));
    std::shared_ptr<TaskStub> taskwait(new TaskStubWait(remoteRank, link, notifyIdx, NotifyTypeStub::READY, "WAIT"));
    TaskQueueStub::AppendTask(curRank, &stream, taskwait);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Transport::GetRemoteMem(std::vector<void *> *remotePtr)
{
    return HCCL_SUCCESS;
}

}