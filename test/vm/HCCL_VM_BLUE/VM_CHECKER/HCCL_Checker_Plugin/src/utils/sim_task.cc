/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "sim_task.h"


namespace HcclSim {

std::string TaskStubLocalCopy::Describe() const
{
    // return "";
    return StringFormat("[LocalCopy] : [srcSlice=%s, dstSlice=%s]", srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
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
    // return "";
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

const HcclDataType TaskStubLocalReduce::GetDataType() const
{
    return dataType;
}

const HcclReduceOp TaskStubLocalReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubLocalReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubRead::Describe() const
{
    // return "";
    return StringFormat("[Read] : [remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank,
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

const LinkInfo TaskStubRead::GetLinkInfo() const
{
    return link;
}

bool TaskStubRead::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubReadReduce::Describe() const
{
    // return "";
    return StringFormat(
        "ReadReduce[remoteRank=%d, link=%s, dataType=%d, reduceOp=%d, localSlice=%s, remoteSlice=%s]",
        remoteRank, link.Describe().c_str(), dataType, reduceOp,
        localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

const LinkInfo TaskStubReadReduce::GetLinkInfo() const
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

const HcclDataType TaskStubReadReduce::GetDataType() const
{
    return dataType;
}

const HcclReduceOp TaskStubReadReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubReadReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubWrite::Describe() const
{
    // return "";
    return StringFormat("[Write] : [remoteRank=%d, link=%s, localSlice=%s, remoteSlice=%s]", remoteRank,
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

const LinkInfo TaskStubWrite::GetLinkInfo() const
{
    return link;
}

bool TaskStubWrite::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubWriteReduce::Describe() const
{
    // return "";
    return StringFormat(
        "WriteReduce[remoteRank=%d, link=%s, dataType=%d, reduceOp=%d, localSlice=%s, remoteSlice=%s]",
        remoteRank, link.Describe().c_str(), dataType, reduceOp,
        localSlice.Describe().c_str(), remoteSlice.Describe().c_str());
}

RankId TaskStubWriteReduce::GetRemoteRank() const
{
    return remoteRank;
}

const LinkInfo TaskStubWriteReduce::GetLinkInfo() const
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

const HcclDataType TaskStubWriteReduce::GetDataType() const
{
    return dataType;
}

const HcclReduceOp TaskStubWriteReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubWriteReduce::IsGenFromSync()
{
	return isGenFromSync;
}

std::string TaskStubPost::Describe() const
{
    // return "";
    return StringFormat("[Post] : [remoteRank=%u, notifyId=%lu, link=%s]", remoteRank, topicId, link.Describe().c_str());
}

void TaskStubPost::SetRemoteRank(RankId rankId)
{
    remoteRank = rankId;
}

RankId TaskStubPost::GetRemoteRank() const
{
    return remoteRank;
}

const LinkProtoStub TaskStubPost::GetLinkType() const
{
    return link.linkProto;
}

const uint64_t TaskStubPost::GetNotifyId() const
{
    return topicId;
}

void TaskStubPost::SetNotifyId(uint64_t id)
{
    topicId = id;
}

const u32 TaskStubPost::GetTopicId() const
{
    return topicId;
}

void TaskStubPost::SetTopicId(u32 id)
{
    topicId = id;
}

std::string TaskStubWait::Describe() const
{
    // return "";
    return StringFormat("[Wait] : [remoteRank=%u, notifyId=%lu, link=%s]", remoteRank, notifyId, link.Describe().c_str());
}

RankId TaskStubWait::GetRemoteRank() const
{
    return remoteRank;
}

void TaskStubWait::SetRemoteRank(RankId rankId)
{
    remoteRank = rankId;
    return;
}

const LinkProtoStub TaskStubWait::GetLinkType() const
{
    return link.linkProto;
}

const uint64_t TaskStubWait::GetNotifyId() const
{
    return notifyId;
}

void TaskStubWait::SetNotifyId(uint64_t id)
{
    notifyId = id;
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

const uint64_t TaskStubLocalPostTo::GetNotifyId() const
{
    return topicId;
}

const uint64_t TaskStubLocalPostTo::GetNotifyIdBack() const
{
    return topicIdBack;
}

void TaskStubLocalPostTo::SetNotifyId(uint64_t id)
{
    topicId = id;
}

std::string TaskStubLocalWaitFrom::Describe() const
{
    // return "";
    return StringFormat("[LocalWaitFrom] : [notifyId=%lu]", notifyId);
}

const uint64_t TaskStubLocalWaitFrom::GetNotifyId() const
{
    return notifyId;
}

void TaskStubLocalWaitFrom::SetNotifyId(uint64_t id)
{
    notifyId = id;
}

u32 TaskStubLocalWaitFrom::GetPostQid() const
{
    return postQid;
}

u32 TaskStubLocalWaitFrom::GetWaitQid() const
{
    return waitQid;
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

const HcclDataType TaskStubBeingReadReduce::GetDataType() const
{
    return dataType;
}

const HcclReduceOp TaskStubBeingReadReduce::GetReduceOp() const
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

const HcclDataType TaskStubBeingWrittenReduce::GetDataType() const
{
    return dataType;
}

const HcclReduceOp TaskStubBeingWrittenReduce::GetReduceOp() const
{
    return reduceOp;
}

bool TaskStubBeingWrittenReduce::IsGenFromSync()
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

u32 TaskStubLocalPostToShadow::GetCurQueId() const
{
    return curQueId;
}

u32 TaskStubLocalPostToShadow::GetPeerQueId() const
{
    return peerQueId;
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

u32 TaskStubLocalWaitFromShadow::GetCurQueId() const
{
    return curQueId;
}

u32 TaskStubLocalWaitFromShadow::GetPeerQueId() const
{
    return peerQueId;
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

const HcclDataType TaskStubLocalBatchReduce::GetDataType() const
{
    return dataType;
}

const HcclReduceOp TaskStubLocalBatchReduce::GetReduceOp() const
{
    return reduceOp;
}

std::string TaskStubLoopStart::Describe() const
{
    return StringFormat("LoopStart[loopGroupIdx=%d, loopIdx=%d]", loopGroupIdx, loopIdx);
}

uint32_t TaskStubLoopStart::GetLoopIdx() const
{
    return loopIdx;
}

uint32_t TaskStubLoopStart::GetLoopGroupIdx() const
{
    return loopGroupIdx;
}

std::string TaskStubLoopEnd::Describe() const
{
    return StringFormat("LoopEnd[loopGroupIdx=%d, loopIdx=%d]", loopGroupIdx, loopIdx);
}

uint32_t TaskStubLoopEnd::GetLoopIdx() const
{
    return loopIdx;
}

uint32_t TaskStubLoopEnd::GetLoopGroupIdx() const
{
    return loopGroupIdx;
}

} //namespace HcclSim