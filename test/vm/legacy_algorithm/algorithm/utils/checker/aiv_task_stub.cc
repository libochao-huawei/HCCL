/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: aiv task stub
 * Author: luyongshu
 * Create: 2025-05-30
 */
 #include "aiv_task_stub.h"

namespace checker {

TaskNode* AivTaskStub::GetAivStart() 
{
    return aivStart;
}

TaskNode* AivTaskStub::GetAivEnd() 
{
    return aivEnd;
}

void AivTaskStub::SetAivStart(TaskNode* aivStart)
{
    this->aivStart = aivStart;
}

void AivTaskStub::SetAivEnd(TaskNode *aivEnd)
{
    this->aivEnd = aivEnd;
}

std::string AivTaskStub::Describe() const
{
	return StringFormat("AivTaskStub[rankId=%d, aivStart=%p, rankPos=%d, mainStreamPos = %d]", rankId, aivStart, rankPos, mainStreamPos);
}

u32 AivTaskStub::GetRankId() const
{
    return rankId;
}

u32 AivTaskStub::GetRankPos() const
{
    return rankPos;
}

u32 AivTaskStub::GetMainStreamPos() const {
    return mainStreamPos;
}

std::string TaskStubAivStart::Describe() const
{
	return StringFormat("AivStart[rankId=%d, rankPos=%d]", rankId, rankPos);
}

RankId TaskStubAivStart::GetRankId() const
{
    return rankId;
}

u32 TaskStubAivStart::GetRankPos() const
{
    return rankPos;
}

std::string TaskStubBlockStart::Describe() const
{
	return StringFormat("BlockStart[rankId=%d, blockId=%d]", rankId, blockId);
}

RankId TaskStubBlockStart::GetRankId() const
{
    return rankId;
}

BlockId TaskStubBlockStart::GetBlockId() const
{
    return blockId;
}

std::string TaskStubAivEnd::Describe() const
{
	return StringFormat("AivEnd[rankId=%d, rankPos=%d]", rankId, rankPos);
}

RankId TaskStubAivEnd::GetRankId() const
{
    return rankId;
}

u32 TaskStubAivEnd::GetRankPos() const
{
    return rankPos;
}

std::string TaskStubSetValue::Describe() const
{
	return StringFormat("SetValue[flagValue=%d]", flagValue);
}

pipe_t TaskStubSetFlag::GetSrcPipe() const 
{
    return srcPipe;
}

pipe_t TaskStubSetFlag::GetDstPipe() const 
{
    return dstPipe;
}

u32 TaskStubSetFlag::GetEventId() const
{
    return eventId;
}

BlockId TaskStubSetFlag::GetBlockId() const
{
    return blockId;
}

bool TaskStubSetFlag::IsGenFromFree() const
{
    return isGenFromFree;
}

std::string TaskStubSetFlag::Describe() const
{
	return StringFormat("SetFlag[src=%s, dst=%s, event=%d, block=%d, isGenFromFree=%s]", 
                        GetPipeName(srcPipe).c_str(), GetPipeName(dstPipe).c_str(), eventId, blockId, isGenFromFree ? "true" : "false");
}


pipe_t TaskStubWaitFlag::GetSrcPipe() const 
{
    return srcPipe;
}

pipe_t TaskStubWaitFlag::GetDstPipe() const 
{
    return dstPipe;
}

u32 TaskStubWaitFlag::GetEventId() const
{
    return eventId;
}

BlockId TaskStubWaitFlag::GetBlockId() const
{
    return blockId;
}

bool TaskStubWaitFlag::IsGenFromAlloc() const
{
    return isGenFromAlloc;
}

std::string TaskStubWaitFlag::Describe() const
{
	return StringFormat("WaitFlag[src=%s, dst=%s, event=%d, block=%d, IsGenFromAlloc=%s]",
                        GetPipeName(srcPipe).c_str(), GetPipeName(dstPipe).c_str(), eventId, blockId, isGenFromAlloc ? "True" : "False");
}

std::string TaskStubSendSync::Describe() const
{
	return StringFormat("SendSync[flagValue=%d, flagAddr=%p]", flagValue, flagAddr);
}

int32_t* TaskStubSendSync::GetFlagAddr() const
{
    return flagAddr;
}

int32_t TaskStubSendSync::GetFlagValue() const
{
    return flagValue;
}

std::string TaskStubRecvSync::Describe() const
{
	return StringFormat("RecvSync[flagValue=%d, flagAddr=%p]", flagValue, flagAddr);
}

int32_t* TaskStubRecvSync::GetFlagAddr() const
{
    return flagAddr;
}

int32_t TaskStubRecvSync::GetFlagValue() const
{
    return flagValue;
}

std::string TaskStubSendSyncReduce::Describe() const
{
	return StringFormat("SendSyncReduce[flagValue=%d, flagAddr=%p]", flagValue, flagAddr);
}

int32_t* TaskStubSendSyncReduce::GetFlagAddr() const
{
    return flagAddr;
}

int32_t TaskStubSendSyncReduce::GetFlagValue() const
{
    return flagValue;
}

std::string TaskStubCompValue::Describe() const
{
	return StringFormat("CompValue[flagValue=%d]", flagValue);
}

int32_t TaskStubCompValue::GetFlagValue() const
{
    return flagValue;
}

std::string TaskStubPipeBarrier::Describe() const
{
    return StringFormat("PipeBarrier[pipeType=%s]", GetPipeName(pipeBarrierType).c_str());
}

bool TaskStubPipeBarrier::IsPipeBarrierAll()
{
    return this->pipeBarrierType == pipe_t::PIPE_ALL;
}

void TaskStubPipeBarrier::SetPipeToPos(pipe_t pipeName, u32 pos)
{
    this->pipeToPos[pipeName] = pos;
}

u32 TaskStubPipeBarrier::GetPos(pipe_t pipeName)
{
    return this->pipeToPos[pipeName];
}

std::string TaskStubSetFlagShadow::Describe() const
{
    return StringFormat("SetFlagShadow[neighborRank=%d, srcBlock=%d, dstBlock=%d, srcpipe=%d, dstPipe=%d, isInVirtual=%s]", 
                        neighborRank, srcBlock, dstBlock, srcpipe, dstPipe, isInVirtual ? "true" : "false");
}

RankId TaskStubSetFlagShadow::GetNeighborRank() const
{
    return neighborRank;
}

u32 TaskStubSetFlagShadow::GetSrcBlock() const
{
    return srcBlock;
}

u32 TaskStubSetFlagShadow::GetDstBlock() const
{
    return dstBlock;
}

u32 TaskStubSetFlagShadow::GetSrcPipe() const
{
    return srcpipe;
}

u32 TaskStubSetFlagShadow::GetDstPipe() const
{
    return dstPipe;    
}

bool TaskStubSetFlagShadow::IsInVirtual() const
{
    return isInVirtual;
}

std::string TaskStubWaitFlagShadow::Describe() const
{
    return StringFormat("WaitFlagShadow[neighborRank=%d, srcBlock=%d, dstBlock=%d, srcpipe=%d, dstPipe=%d, isInVirtual=%s]", 
                        neighborRank, srcBlock, dstBlock, srcpipe, dstPipe, isInVirtual ? "true" : "false");
}

RankId TaskStubWaitFlagShadow::GetNeighborRank() const
{
    return neighborRank;
}

u32 TaskStubWaitFlagShadow::GetSrcBlock() const
{
    return srcBlock;
}

u32 TaskStubWaitFlagShadow::GetDstBlock() const
{
    return dstBlock;
}

u32 TaskStubWaitFlagShadow::GetSrcPipe() const
{
    return srcpipe;
}

u32 TaskStubWaitFlagShadow::GetDstPipe() const
{
    return dstPipe;    
}

bool TaskStubWaitFlagShadow::IsInVirtual() const
{
    return isInVirtual;
}

std::string TaskStubVirtualRankStart::Describe() const
{
    return StringFormat("VirtualRankStart[neighborRank=%d]",  neighborRank);
}

RankId TaskStubVirtualRankStart::GetNeighborRank() const
{
    return neighborRank;
}

std::string GetPipeName(pipe_t pipe) {
    switch (pipe) {
        case pipe_t::PIPE_ALL:
            return "PIPE_ALL";

        case pipe_t::PIPE_MTE2:
            return "PIPE_MTE2";

        case pipe_t::PIPE_MTE3:
            return "PIPE_MTE3";

        case pipe_t::PIPE_S:
            return "PIPE_SCALAR";

        default:
            return "UNKNOWN";
    }
}

std::string GetTaskName(TaskTypeStub taskType) {
    switch (taskType) {
        case TaskTypeStub::SET_VALUE: 
            return "SetValue";

        case TaskTypeStub::SET_FLAG: 
            return "SetFlag";

        case TaskTypeStub::WAIT_FLAG: 
            return "WaitFlag";

        case TaskTypeStub::SEND_SYNC: 
            return "SendSync";

        case TaskTypeStub::RECV_SYNC:
            return "RecvSync";

        case TaskTypeStub::SEND_SYNC_REDUCE: 
            return "SendSyncReduce";

        case TaskTypeStub::COMP_VALUE: 
            return "CompValue";

        case TaskTypeStub::PIPE_BARRIER: 
            return "PipeBarrier";

        case TaskTypeStub::LOCAL_COPY: 
            return "LocalCopy";

        case TaskTypeStub::LOCAL_REDUCE: 
            return "LocalReduce";

        case TaskTypeStub::READ: 
            return "Read";
                        
        case TaskTypeStub::READ_REDUCE: 
            return "ReadReduce";

        case TaskTypeStub::WRITE: 
            return "Write";

        case TaskTypeStub::WRITE_REDUCE: 
            return "WriteReduce";

        case TaskTypeStub::SET_FLAG_SHADOW:
            return "SetFlagShadow";

        case TaskTypeStub::WAIT_FLAG_SHADOW:
            return "WaitFlagShadow";

        default:
            return "unknown task";
        }
}

bool IsGenFromSync(TaskStub* task)
{
    bool isGenFromSync = false;
    if (task->GetType() == TaskTypeStub::LOCAL_COPY) {
        TaskStubLocalCopy *candLocalCopy = dynamic_cast<TaskStubLocalCopy *>(task);
        isGenFromSync = candLocalCopy->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::LOCAL_REDUCE) {
        TaskStubLocalReduce *candLocalReduce = dynamic_cast<TaskStubLocalReduce *>(task);
        isGenFromSync = candLocalReduce->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::READ) {
        TaskStubRead *candRead = dynamic_cast<TaskStubRead *>(task);
        isGenFromSync = candRead->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::WRITE) {
        TaskStubWrite *candWrite = dynamic_cast<TaskStubWrite *>(task);
        isGenFromSync = candWrite->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::READ_REDUCE) {
        TaskStubReadReduce *candReadReduce = dynamic_cast<TaskStubReadReduce *>(task);
        isGenFromSync = candReadReduce->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::WRITE_REDUCE) {
        TaskStubWriteReduce *candWriteReduce = dynamic_cast<TaskStubWriteReduce *>(task);
        isGenFromSync = candWriteReduce->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::BEING_READ) {
        TaskStubBeingRead *candBeingRead = dynamic_cast<TaskStubBeingRead *>(task);
        isGenFromSync = candBeingRead->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::BEING_WRITTEN) {
        TaskStubBeingWritten *candBeingWritten = dynamic_cast<TaskStubBeingWritten *>(task);
        isGenFromSync = candBeingWritten->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::BEING_READ_REDUCE) {
        TaskStubBeingReadReduce *candBeingReadReduce = dynamic_cast<TaskStubBeingReadReduce *>(task);
        isGenFromSync = candBeingReadReduce->IsGenFromSync();
    } else if (task->GetType() == TaskTypeStub::BEING_WRITTEN_REDUCE) {
        TaskStubBeingWrittenReduce *candBeingWrittenReduce = dynamic_cast<TaskStubBeingWrittenReduce *>(task);
        isGenFromSync = candBeingWrittenReduce->IsGenFromSync();
    } 

    return isGenFromSync;
}

}
