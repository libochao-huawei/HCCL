/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: hccL 2.0 transform function 实现
 * Create: 2025-04-11
 */

#include "host_task_transform.h"
#include "task_stub.h"
#include "task_ccu.h"
#include "task_queue_stub.h"
#include "coll_service_stub.h"

namespace Hccl {

using InstructionFunc = HcclResult (*)(const Instruction &ins, RankId rankId, QId qId);
extern std::unordered_map<uint16_t, InstructionFunc> g_transformInstrTypeMap;

HcclResult GenCollAlgOperator(CollAlgOperator &op, CheckerOpParam &checkerOpParam)
{
    // ReduceOp
    if (g_CheckerReduceOp2ReduceOp_aicpu.count(checkerOpParam.reduceType) != 0) {
        op.reduceOp = g_CheckerReduceOp2ReduceOp_aicpu[checkerOpParam.reduceType];
    }
    //opType
    if (g_CheckerOpType2OpType_aicpu.count(checkerOpParam.opType) != 0) {
        op.opType = g_CheckerOpType2OpType_aicpu[checkerOpParam.opType];
    }
    if (checkerOpParam.opType == CheckerOpType::ALLTOALL) {
        op.all2AllDataDes.sendType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.sendType];
        op.all2AllDataDes.recvType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.recvType];
        op.all2AllDataDes.sendCount = checkerOpParam.All2AllDataDes.sendCount;
        op.all2AllDataDes.recvCount = checkerOpParam.All2AllDataDes.recvCount;
    } else if (checkerOpParam.opType == CheckerOpType::ALLTOALLV) {
        op.all2AllVDataDes.sendType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.sendType];
        op.all2AllVDataDes.recvType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.recvType];
        op.all2AllVDataDes.sendCounts = static_cast<void *>(checkerOpParam.All2AllDataDes.sendCounts.data());
        op.all2AllVDataDes.recvCounts = static_cast<void *>(checkerOpParam.All2AllDataDes.recvCounts.data());
        op.all2AllVDataDes.sdispls = static_cast<void *>(checkerOpParam.All2AllDataDes.sdispls.data());
        op.all2AllVDataDes.rdispls = static_cast<void *>(checkerOpParam.All2AllDataDes.rdispls.data());
    } else if (checkerOpParam.opType == CheckerOpType::ALLTOALLVC) {
        op.all2AllVCDataDes.sendType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.sendType];
        op.all2AllVCDataDes.recvType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.recvType];
        op.all2AllVCDataDes.sendCountMatrix = static_cast<void *>(checkerOpParam.All2AllDataDes.sendCountMatrix.data());
    } else {
        //datacount datatype
        op.dataCount = checkerOpParam.DataDes.count;
        op.dataType = g_CheckerDataType2DataType_aicpu[checkerOpParam.DataDes.dataType];
    }
    // root
    op.root = checkerOpParam.root;
    // opMode
    op.opMode = g_CheckerOpMode2OpMode_aicpu[checkerOpParam.opMode];
    return HCCL_SUCCESS;
}

HcclResult GenCollAlgParams(CollAlgParams &params, CheckerOpParam &checkerOpParam, DavidAlgConfig &config)
{
    params.opMode = g_CheckerOpMode2OpMode_aicpu[checkerOpParam.opMode];
    params.maxTmpMemSize = config.maxTmpMemSize;
    return HCCL_SUCCESS;
}

HcclResult HcclDataSlice2CheckerDataSlice(const Hccl::DataSlice &dataSlice, checker::DataSlice &checkerDataSlice)
{
    checkerDataSlice.SetSize(dataSlice.GetSize());
    checkerDataSlice.SetOffset(dataSlice.GetOffset());
    checkerDataSlice.SetBufferType(g_HcclBufferType2CheckerBufferType_aicpu[dataSlice.GetType()]);

    return HCCL_SUCCESS;
}

HcclResult TransformInsLocalCopy (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insLocalCopy = static_cast<const InsLocalCopy &>(ins);
    checker::DataSlice srcSlice;
    checker::DataSlice dstSlice;
    HcclDataSlice2CheckerDataSlice(insLocalCopy.GetSrcSlice(), srcSlice);
    HcclDataSlice2CheckerDataSlice(insLocalCopy.GetDstSlice(), dstSlice);
    std::shared_ptr<TaskStubLocalCopy> tasLocalCopy(new TaskStubLocalCopy(srcSlice, dstSlice));
    TaskQueueStub::AppendTask(rankId, qId, tasLocalCopy);
    return HCCL_SUCCESS;
}

HcclResult TransformInsLocalReduce (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insLocalReduce = static_cast<const InsLocalReduce &>(ins);
    checker::DataSlice srcSlice;
    checker::DataSlice dstSlice;
    HcclDataSlice2CheckerDataSlice(insLocalReduce.GetSrcSlice(), srcSlice);
    HcclDataSlice2CheckerDataSlice(insLocalReduce.GetDstSlice(), dstSlice);

    CheckerDataType dataType;
    dataType = g_DataType2CheckerDataType_aicpu[insLocalReduce.GetDataType()];
    CheckerReduceOp reduceOp;
    reduceOp = g_ReduceOp2CheckerReduceOp_aicpu[insLocalReduce.GetReduceOp()];
    std::shared_ptr<TaskStubLocalReduce> taskLocalReduce(new TaskStubLocalReduce(srcSlice, dstSlice, dataType, reduceOp));
    TaskQueueStub::AppendTask(rankId, qId, taskLocalReduce);
    return HCCL_SUCCESS;
}

HcclResult TransformInsLocalPostTo (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insLocalPostTo = static_cast<const InsLocalPostTo &>(ins);
    u32 topicID = insLocalPostTo.GetTopicId();
    std::shared_ptr<TaskStub> taskLocalPostTo(new TaskStubLocalPostTo(
        topicID, insLocalPostTo.GetPostQid(), insLocalPostTo.GetWaitQid()));
    TaskQueueStub::AppendTask(rankId, qId, taskLocalPostTo);
    return HCCL_SUCCESS;
}

HcclResult TransformInsLocalWaitFrom (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insLocalWaitFrom = static_cast<const InsLocalWaitFrom &>(ins);
    u32 topicId = insLocalWaitFrom.GetTopicId();
    std::shared_ptr<TaskStub> taskLocalWaitFrom(new TaskStubLocalWaitFrom(
        topicId, insLocalWaitFrom.GetPostQid(), insLocalWaitFrom.GetWaitQid()));
    TaskQueueStub::AppendTask(rankId, qId, taskLocalWaitFrom);
    return HCCL_SUCCESS;
}

HcclResult TransformInsLocalWaitGroup (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insLocalWaitGroup = static_cast<const InsLocalWaitGroup &>(ins);
    u32 topicId = insLocalWaitGroup.GetTopicId();
    for (auto insIter = insLocalWaitGroup.Iter(); insIter.HasNext(); ++insIter) {
        std::shared_ptr<TaskStub> taskLocalWaitGroup(new TaskStubLocalWaitFrom(
            topicId, (*insIter), insLocalWaitGroup.GetWaitQid()));
        TaskQueueStub::AppendTask(rankId, qId, taskLocalWaitGroup);
    }
    return HCCL_SUCCESS;
}

HcclResult TransformInsLocalBcastPost (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insLocalBcastPost = static_cast<const InsLocalBcastPost &>(ins);
    u32 topicId = insLocalBcastPost.GetTopicId();
    for (auto insIter = insLocalBcastPost.Iter(); insIter.HasNext(); ++insIter) {
        std::shared_ptr<TaskStub> taskLocalBcastPost(new TaskStubLocalPostTo(
            topicId, insLocalBcastPost.GetPostQid(), (*insIter)));
        TaskQueueStub::AppendTask(rankId, qId, taskLocalBcastPost);
    }
    return HCCL_SUCCESS;
}

HcclResult TransformInsPostReady (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insPostReady = static_cast<const InsPostReady &>(ins);
    RankId remoteRank;
    remoteRank = insPostReady.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insPostReady.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::READY;
    std::string tag = "InsPostReady";
    std::shared_ptr<TaskStub> taskPostReady(new TaskStubPost(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskPostReady);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWaitReady (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWaitReady = static_cast<const InsWaitReady &>(ins);
    RankId remoteRank;
    remoteRank = insWaitReady.GetRemoteRank();
    LinkProtoType linkType;
    linkType =  LinkProtocol2LinkProtoType(insWaitReady.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::READY;
    std::string tag = "InsWaitReady";
    std::shared_ptr<TaskStub> taskWaitReady(new TaskStubWait(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskWaitReady);
    return HCCL_SUCCESS;
}

HcclResult TransformInsPostFin (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insPostFin = static_cast<const InsPostFin &>(ins);
    RankId remoteRank;
    remoteRank = insPostFin.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insPostFin.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::FIN;
    std::string tag = "InsPostFin";
    std::shared_ptr<TaskStub> taskPostFin(new TaskStubPost(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskPostFin);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWaitFin (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWaitFin = static_cast<const InsWaitFin &>(ins);
    RankId remoteRank;
    remoteRank = insWaitFin.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insWaitFin.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::FIN;
    std::string tag = "InsWaitFin";
    std::shared_ptr<TaskStub> taskWaitFin(new TaskStubWait(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskWaitFin);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWaitGroupFin (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWaitGroupFin = static_cast<const InsWaitGroupFin &>(ins);
    for (auto insIter = insWaitGroupFin.Iter(); insIter.HasNext(); ++insIter) {
        RankId remoteRank;
        remoteRank = (*insIter).GetRemoteRankId();
        LinkProtoType linkType;
        linkType = LinkProtocol2LinkProtoType((*insIter).GetLinkProtocol());
        LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
        u32 topicId = 0;
        NotifyTypeStub notifyType;
        notifyType = NotifyTypeStub::FIN;
        std::string tag = "InsWaitGroupFin";
        std::shared_ptr<TaskStub> taskWaitGroupFin(new TaskStubWait(remoteRank, link, topicId, notifyType, tag));
        TaskQueueStub::AppendTask(rankId, qId, taskWaitGroupFin);
    }
    return HCCL_SUCCESS;
}

HcclResult TransformInsPostFinAck (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insPostFinAck = static_cast<const InsPostFinAck &>(ins);
    RankId remoteRank;
    remoteRank = insPostFinAck.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insPostFinAck.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::FIN_ACK;
    std::string tag = "InsPostFinAck";
    std::shared_ptr<TaskStub> taskPostFinAck(new TaskStubPost(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskPostFinAck);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWaitFinAck (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWaitFinAck = static_cast<const InsWaitFinAck &>(ins);
    RankId remoteRank;
    remoteRank = insWaitFinAck.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insWaitFinAck.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::FIN_ACK;
    std::string tag = "InsWaitFinAck";
    std::shared_ptr<TaskStub> taskWaitFinAck(new TaskStubWait(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskWaitFinAck);
    return HCCL_SUCCESS;
}

HcclResult TransformInsRead (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insRead = static_cast<const InsRead &>(ins);
    RankId remoteRank;
    remoteRank = insRead.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insRead.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);

    checker::DataSlice localSlice;
    checker::DataSlice remoteSlice;
    HcclDataSlice2CheckerDataSlice(insRead.GetLocalSlice(), localSlice);
    HcclDataSlice2CheckerDataSlice(insRead.GetRemoteSlice(), remoteSlice);

    std::shared_ptr<TaskStub> taskRead(new TaskStubRead(remoteRank, link, localSlice, remoteSlice));
    TaskQueueStub::AppendTask(rankId, qId, taskRead);
    return HCCL_SUCCESS;
}

HcclResult TransformInsReadReduce (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insReadReduce = static_cast<const InsReadReduce &>(ins);
    RankId remoteRank;
    remoteRank = insReadReduce.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insReadReduce.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);

    checker::DataSlice localSlice;
    checker::DataSlice remoteSlice;
    HcclDataSlice2CheckerDataSlice(insReadReduce.GetLocalSlice(), localSlice);
    HcclDataSlice2CheckerDataSlice(insReadReduce.GetRemoteSlice(), remoteSlice);

    CheckerDataType dataType;
    dataType = g_DataType2CheckerDataType_aicpu[insReadReduce.GetDataType()];
    CheckerReduceOp reduceOp;
    reduceOp = g_ReduceOp2CheckerReduceOp_aicpu[insReadReduce.GetReduceOp()];
    std::shared_ptr<TaskStub> taskReadReduce(new TaskStubReadReduce(remoteRank, link, localSlice, remoteSlice, dataType, reduceOp));
    TaskQueueStub::AppendTask(rankId, qId, taskReadReduce);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWrite (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWrite = static_cast<const InsWrite &>(ins);
    RankId remoteRank;
    remoteRank = insWrite.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insWrite.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);

    checker::DataSlice localSlice;
    checker::DataSlice remoteSlice;
    HcclDataSlice2CheckerDataSlice(insWrite.GetLocalSlice(), localSlice);
    HcclDataSlice2CheckerDataSlice(insWrite.GetRemoteSlice(), remoteSlice);
    std::shared_ptr<TaskStub> taskWrite(new TaskStubWrite(remoteRank, link, localSlice, remoteSlice));
    TaskQueueStub::AppendTask(rankId, qId, taskWrite);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWriteReduce (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWriteReaduce = static_cast<const InsWriteReduce &>(ins);
    RankId remoteRank;
    remoteRank = insWriteReaduce.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insWriteReaduce.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);

    checker::DataSlice localSlice;
    checker::DataSlice remoteSlice;
    HcclDataSlice2CheckerDataSlice(insWriteReaduce.GetLocalSlice(), localSlice);
    HcclDataSlice2CheckerDataSlice(insWriteReaduce.GetRemoteSlice(), remoteSlice);

    CheckerDataType dataType;
    dataType = g_DataType2CheckerDataType_aicpu[insWriteReaduce.GetDataType()];
    CheckerReduceOp reduceOp;
    reduceOp = g_ReduceOp2CheckerReduceOp_aicpu[insWriteReaduce.GetReduceOp()];
    std::shared_ptr<TaskStub> taskWriteReaduce(new TaskStubWriteReduce(remoteRank, link, localSlice, remoteSlice, dataType, reduceOp));
    TaskQueueStub::AppendTask(rankId, qId, taskWriteReaduce);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWriteReduceWithFin (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWriteReaduceWithFin = static_cast<const InsWriteReduceWithFin &>(ins);
    //获取参数writereduce
    RankId remoteRank;
    remoteRank = insWriteReaduceWithFin.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insWriteReaduceWithFin.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);

    checker::DataSlice localSlice;
    checker::DataSlice remoteSlice;
    HcclDataSlice2CheckerDataSlice(insWriteReaduceWithFin.GetLocalSlice(), localSlice);
    HcclDataSlice2CheckerDataSlice(insWriteReaduceWithFin.GetRemoteSlice(), remoteSlice);

    CheckerDataType dataType;
    dataType = g_DataType2CheckerDataType_aicpu[insWriteReaduceWithFin.GetDataType()];
    CheckerReduceOp reduceOp;
    reduceOp = g_ReduceOp2CheckerReduceOp_aicpu[insWriteReaduceWithFin.GetReduceOp()];
    std::shared_ptr<TaskStub> taskWriteReaduce(new TaskStubWriteReduce(remoteRank, link, localSlice, remoteSlice, dataType, reduceOp));
    TaskQueueStub::AppendTask(rankId, qId, taskWriteReaduce);
    // 生成postFin task
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::FIN;
    std::string tag = "InsWriteReduceWithFin";
    std::shared_ptr<TaskStub> taskPostFin(new TaskStubPost(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskPostFin);
    return HCCL_SUCCESS;
}

HcclResult TransformInsWriteWithFin (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &insWriteWithFin = static_cast<const InsWriteWithFin &>(ins);
    RankId remoteRank;
    remoteRank = insWriteWithFin.GetRemoteRank();
    LinkProtoType linkType;
    linkType = LinkProtocol2LinkProtoType(insWriteWithFin.GetLink()->GetLinkProtocol());
    LinkInfoStub link(g_LinkProtoType2LinkProtoStub_aicpu[linkType]);

    checker::DataSlice localSlice;
    checker::DataSlice remoteSlice;
    HcclDataSlice2CheckerDataSlice(insWriteWithFin.GetLocalSlice(), localSlice);
    HcclDataSlice2CheckerDataSlice(insWriteWithFin.GetRemoteSlice(), remoteSlice);

    std::shared_ptr<TaskStub> taskWrite(new TaskStubWrite(remoteRank, link, localSlice, remoteSlice));
    TaskQueueStub::AppendTask(rankId, qId, taskWrite);
    u32 topicId = 0;
    NotifyTypeStub notifyType;
    notifyType = NotifyTypeStub::FIN;
    std::string tag = "InsWriteWithFin";
    std::shared_ptr<TaskStub> taskWriteWithFin(new TaskStubPost(remoteRank, link, topicId, notifyType, tag));
    TaskQueueStub::AppendTask(rankId, qId, taskWriteWithFin);
    return HCCL_SUCCESS;
}

HcclResult TransformInsBatchWith (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &batchIns = static_cast<const InsBatchWrite &>(ins);
    for (auto iter = batchIns.Iter(); iter.HasNext(); ++iter) {
        TransformIns2Task(*iter, rankId, qId);
    }
    return HCCL_SUCCESS;
}

HcclResult TransformInsBatchRead (const Instruction &ins, RankId rankId, QId qId)
{
    const auto &batchIns = static_cast<const InsBatchRead &>(ins);
    for (auto iter = batchIns.Iter(); iter.HasNext(); ++iter) {
        TransformIns2Task(*iter, rankId, qId);
    }
    return HCCL_SUCCESS;
}

HcclResult TransformInsCcuIns (const Instruction &ins, RankId rankId, QId qId)
{
    std::shared_ptr<TaskStub> taskCCU(new TaskStubCcuGraph(ins, rankId));
    TaskQueueStub::AppendTask(rankId, qId, taskCCU);
    return HCCL_SUCCESS;
}

HcclResult TransformInsAivIns (const Instruction &ins, RankId rankId, QId qId)
{
    return HCCL_SUCCESS;
}

std::unordered_map<uint16_t, InstructionFunc> g_transformInstrTypeMap = {
    {InstructionType::LOCAL_COPY, &TransformInsLocalCopy},
    {InstructionType::LOCAL_REDUCE, &TransformInsLocalReduce},
    {InstructionType::LOCAL_POST_TO, &TransformInsLocalPostTo},
    {InstructionType::LOCAL_WAIT_FROM, &TransformInsLocalWaitFrom},
    {InstructionType::LOCAL_WAIT_GROUP, &TransformInsLocalWaitGroup},
    {InstructionType::LOCAL_BCAST_POST, &TransformInsLocalBcastPost},
    {InstructionType::POST_READY, &TransformInsPostReady},
    {InstructionType::WAIT_READY, &TransformInsWaitReady},
    {InstructionType::POST_FIN, &TransformInsPostFin},
    {InstructionType::WAIT_FIN, &TransformInsWaitFin},
    {InstructionType::WAIT_GROUP_FIN, &TransformInsWaitGroupFin},
    {InstructionType::POST_FIN_ACK, &TransformInsPostFinAck},
    {InstructionType::WAIT_FIN_ACK, &TransformInsWaitFinAck},
    {InstructionType::READ, &TransformInsRead},
    {InstructionType::READ_REDUCE, &TransformInsReadReduce},
    {InstructionType::WRITE, &TransformInsWrite},
    {InstructionType::WRITE_REDUCE, &TransformInsWriteReduce},
    {InstructionType::WRITE_REDUCE_WITH_FIN, &TransformInsWriteReduceWithFin},
    {InstructionType::WRITE_WITH_FIN, &TransformInsWriteWithFin},
    {InstructionType::BATCH_WRITE, &TransformInsBatchWith},
    {InstructionType::BATCH_READ, &TransformInsBatchRead},
    {InstructionType::CCU_INS, &TransformInsCcuIns},
    {InstructionType::AIV_INS, &TransformInsAivIns},
};

HcclResult TransformIns2Task(const Instruction &ins, RankId rankId, QId qId)
{
    auto insTypeId = g_transformInstrTypeMap.find(ins.GetType());
    if (insTypeId == g_transformInstrTypeMap.end()) {
        HCCL_ERROR("Ins type %s not supported.", ins.GetType().Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return insTypeId->second(ins, rankId, qId);
}

} // namespace hccl