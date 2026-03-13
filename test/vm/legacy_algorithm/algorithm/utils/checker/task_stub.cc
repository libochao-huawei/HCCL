/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: task stub
 * Author: shenyutian
 * Create: 2024-07-27
 */

#include "task_stub.h"

namespace checker {
std::string TaskStubLocalCopy::Describe() const
{
    return StringFormat("LocalCopy[srcSlice=%s, dstSlice=%s]", srcSlice.Describe().c_str(),
                        dstSlice.Describe().c_str());
}

const DataSlice &TaskStubLocalCopy::GetSrcSlice() const
{
    return srcSlice;
}

const DataSlice &TaskStubLocalCopy::GetDstSlice() const
{
    return dstSlice;
}

bool TaskStubLocalCopy::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubLocalReduce::Describe() const
{
    return StringFormat("LocalReduce[dataType=%d, reduceOp=%d, srcSlice=%s, dstSlice=%s]",
                        dataType, reduceOp, srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
}

const DataSlice &TaskStubLocalReduce::GetSrcSlice() const
{
    return srcSlice;
}

const DataSlice &TaskStubLocalReduce::GetDstSlice() const
{
    return dstSlice;
}

const CheckerDataType TaskStubLocalReduce::GetDataType() const
{
    return dataType;
}

const CheckerReduceOp TaskStubLocalReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubLocalReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubLocalBatchReduce::Describe() const
{
    std::string srcDes = "[";
    for (auto& slice : srcSlices) {
        srcDes += slice.Describe();
        srcDes += ",";
    }
    srcDes[srcDes.size() - 1] = ']';

    return StringFormat("LocalBatchReduce[dataType=%d, reduceOp=%d, srcSlice=%s, dstSlice=%s]",
                        dataType, reduceOp, srcDes.c_str(), dstSlice.Describe().c_str());
}

const std::vector<DataSlice>& TaskStubLocalBatchReduce::GetSrcSlices() const
{
    return srcSlices;
}

const DataSlice& TaskStubLocalBatchReduce::GetSrcSlice(u32 index) const
{
    return srcSlices[index];
}

const DataSlice& TaskStubLocalBatchReduce::GetDstSlice() const
{
    return dstSlice;
}

const CheckerDataType TaskStubLocalBatchReduce::GetDataType() const
{
    return dataType;
}

const CheckerReduceOp TaskStubLocalBatchReduce::GetReduceOp() const
{
    return reduceOp;
}

std::string TaskStubLocalPostTo::Describe() const
{
    return StringFormat("LocalPostTo[topicId=%d]", topicIdBack);
}

void TaskStubLocalPostTo::SetPostQid(QId qid)
{
    postQid = qid;
}

void TaskStubLocalPostTo::SetWaitQid(QId qid)
{
    waitQid = qid;
}

QId TaskStubLocalPostTo::GetPostQid() const
{
    return postQid;
}

QId TaskStubLocalPostTo::GetWaitQid() const
{
    return waitQid;
}

u32 TaskStubLocalPostTo::GetTopicId() const
{
    return topicId;
}

void TaskStubLocalPostTo::SetTopicId(u32 id)
{
    topicId = id;
}

u32 TaskStubLocalPostTo::GetTopicIdBack() const
{
    return topicIdBack;
}

bool TaskStubLocalPostTo::IsInvalidPost() const
{
    return invalidPost_;
}

std::string TaskStubLocalWaitFrom::Describe() const
{
    return StringFormat("LocalWaitFrom[topicId=%d]", topicId);
}

void TaskStubLocalWaitFrom::SetPostQid(QId qid)
{
    postQid = qid;
}

void TaskStubLocalWaitFrom::SetWaitQid(QId qid)
{
    waitQid = qid;
}

QId TaskStubLocalWaitFrom::GetPostQid() const
{
    return postQid;
}

QId TaskStubLocalWaitFrom::GetWaitQid() const
{
    return waitQid;
}

u32 TaskStubLocalWaitFrom::GetTopicId() const
{
    return topicId;
}

std::string TaskStubPost::Describe() const
{
    if (notifyType != NotifyTypeStub::CCU) {
        return StringFormat("Post[remoteRank=%d, link=%s, notifyType=%s, originated tag=%s]", remoteRank,
                            link.Describe().c_str(), notifyType.Describe().c_str(), tag.c_str());
    }
    return StringFormat("Post[remoteRank=%d, link=%s, topicId=%u, originated tag=%s]", remoteRank,
                        link.Describe().c_str(), topicIdBack, tag.c_str());
}

RankId TaskStubPost::GetRemoteRank() const
{
    return remoteRank;
}
const LinkProtoStub TaskStubPost::GetLinkType() const
{
    return link.linkProto;
}
const u32 TaskStubPost::GetTopicId() const
{
    return topicId;
}

void TaskStubPost::SetTopicId(u32 id)
{
    topicId = id;
}

const NotifyTypeStub TaskStubPost::GetNotifyType() const
{
    return notifyType;
}

const std::string TaskStubPost::GetTag() const
{
    return tag;
}

std::string TaskStubWait::Describe() const
{
    if (notifyType != NotifyTypeStub::CCU) {
        return StringFormat("Wait[remoteRank=%d, link=%s, notifyType=%s, originated tag=%s]", remoteRank,
                            link.Describe().c_str(), notifyType.Describe().c_str(), tag.c_str());
    }
    return StringFormat("Wait[remoteRank=%d, link=%s, topicId=%u, originated tag=%s]", remoteRank,
                        link.Describe().c_str(), topicId, tag.c_str());
}

RankId TaskStubWait::GetRemoteRank() const
{
    return remoteRank;
}

void TaskStubWait::SetRemoteRank(u32 rankId)
{
    remoteRank = rankId;
    return;
}

const LinkProtoStub TaskStubWait::GetLinkType() const
{
    return link.linkProto;
}

const u32 TaskStubWait::GetTopicId() const
{
    return topicId;
}

const NotifyTypeStub TaskStubWait::GetNotifyType() const
{
    return notifyType;
}

const std::string TaskStubWait::GetTag() const
{
    return tag;
}

std::string TaskStubRead::Describe() const
{
    return StringFormat("Read[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank,
                        link.Describe().c_str(), localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubRead::GetRemoteRank() const
{
    return remoteRank;
}
const LinkProtoStub TaskStubRead::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubRead::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubRead::GetRemoteSlice() const
{
    return remoteSlice;
}

const LinkInfoStub TaskStubRead::GetLinkInfo() const
{
    return link;
}

bool TaskStubRead::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubReadReduce::Describe() const
{
    return StringFormat(
        "ReadReduce[remoteRank=%d, link=%s, dataType=%d, reduceOp=%d, localSlice=%s, remoteSlice=%s]",
        remoteRank, link.Describe().c_str(), dataType, reduceOp,
        localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

const LinkInfoStub TaskStubReadReduce::GetLinkInfo() const
{
    return link;
}

RankId TaskStubReadReduce::GetRemoteRank() const
{
    return remoteRank;
}

const LinkProtoStub TaskStubReadReduce::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubReadReduce::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubReadReduce::GetRemoteSlice() const
{
    return remoteSlice;
}

const CheckerDataType TaskStubReadReduce::GetDataType() const
{
    return dataType;
}

const CheckerReduceOp TaskStubReadReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubReadReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubWrite::Describe() const
{
    return StringFormat("Write[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank,
                        link.Describe().c_str(), localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubWrite::GetRemoteRank() const
{
    return remoteRank;
}

const LinkProtoStub TaskStubWrite::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubWrite::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubWrite::GetRemoteSlice() const
{
    return remoteSlice;
}

const LinkInfoStub TaskStubWrite::GetLinkInfo() const
{
    return link;
}

bool TaskStubWrite::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubWriteReduce::Describe() const
{
    return StringFormat(
        "WriteReduce[remoteRank=%d, link=%s, dataType=%d, reduceOp=%d, localSlice=%s, remoteSlice=%s]",
        remoteRank, link.Describe().c_str(), dataType, reduceOp,
        localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubWriteReduce::GetRemoteRank() const
{
    return remoteRank;
}

const LinkInfoStub TaskStubWriteReduce::GetLinkInfo() const
{
    return link;
}

const LinkProtoStub TaskStubWriteReduce::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubWriteReduce::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubWriteReduce::GetRemoteSlice() const
{
    return remoteSlice;
}

const CheckerDataType TaskStubWriteReduce::GetDataType() const
{
    return dataType;
}

const CheckerReduceOp TaskStubWriteReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubWriteReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubLocalPostToShadow::Describe() const
{
    return StringFormat("LocalPostToShadow[neighborRank=%d, postQid=%d, topicId=%d]",
                        neighborRank, curQueId, peerQueId);
}

RankId TaskStubLocalPostToShadow::GetNeighborRank() const
{
    return neighborRank;
}

std::string TaskStubLocalWaitFromShadow::Describe() const
{
    return StringFormat("LocalWaitFromShadow[neighborRank=%d, waitQid=%d, topicId=%d]",
                        neighborRank, curQueId, peerQueId);
}

RankId TaskStubLocalWaitFromShadow::GetNeighborRank() const
{
    return neighborRank;
}

std::string TaskStubBeingRead::Describe() const
{
    return StringFormat("BeingRead[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank,
                        link.Describe().c_str(), localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubBeingRead::GetRemoteRank() const
{
    return remoteRank;
}
const LinkProtoStub TaskStubBeingRead::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubBeingRead::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubBeingRead::GetRemoteSlice() const
{
    return remoteSlice;
}

bool TaskStubBeingRead::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubBeingReadReduce::Describe() const
{
    return StringFormat(
        "BeingReadReduce[remoteRank=%d, link=%s, dataType=%d, reduceOp=%d, localSlice=%s, remoteSlice=%s]",
        remoteRank, link.Describe().c_str(), dataType, reduceOp,
        localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubBeingReadReduce::GetRemoteRank() const
{
    return remoteRank;
}

const LinkProtoStub TaskStubBeingReadReduce::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubBeingReadReduce::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubBeingReadReduce::GetRemoteSlice() const
{
    return remoteSlice;
}

const CheckerDataType TaskStubBeingReadReduce::GetDataType() const
{
    return dataType;
}

const CheckerReduceOp TaskStubBeingReadReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubBeingReadReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubBeingWritten::Describe() const
{
    return StringFormat("BeingWritten[remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank,
                        link.Describe().c_str(), localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubBeingWritten::GetRemoteRank() const
{
    return remoteRank;
}

const LinkProtoStub TaskStubBeingWritten::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubBeingWritten::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubBeingWritten::GetRemoteSlice() const
{
    return remoteSlice;
}

bool TaskStubBeingWritten::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubBeingWrittenReduce::Describe() const
{
    return StringFormat(
        "BeingWrittenReduce[remoteRank=%d, link=%s, dataType=%d, reduceOp=%d, localSlice=%s, remoteSlice=%s]",
        remoteRank, link.Describe().c_str(), dataType, reduceOp,
        localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubBeingWrittenReduce::GetRemoteRank() const
{
    return remoteRank;
}

const LinkProtoStub TaskStubBeingWrittenReduce::GetLinkType() const
{
    return link.linkProto;
}

const DataSlice &TaskStubBeingWrittenReduce::GetLocalSlice() const
{
    return localSlice;
}

const DataSlice &TaskStubBeingWrittenReduce::GetRemoteSlice() const
{
    return remoteSlice;
}

const CheckerDataType TaskStubBeingWrittenReduce::GetDataType() const
{
    return dataType;
}

const CheckerReduceOp TaskStubBeingWrittenReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubBeingWrittenReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubLoopStart::Describe() const
{
    return StringFormat("LoopStart[loopGroupIdx=%d, loopIdx=%d]", loopGroupIdx, loopIdx);
}

std::string TaskStubLoopEnd::Describe() const
{
    return StringFormat("LoopEnd[loopGroupIdx=%d, loopIdx=%d]", loopGroupIdx, loopIdx);
}

} // namespace hccl