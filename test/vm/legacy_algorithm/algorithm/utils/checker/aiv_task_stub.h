/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: aiv task stub header file
 * Author: luyongshu
 * Create: 2025-05-30
 */

#ifndef HCCLV1_AIV_TASK_STUB_H
#define HCCLV1_AIV_TASK_STUB_H

#include "task_queue_stub.h"
#include "aiv_base_stub.h"

using namespace AscendC;

namespace checker {

struct TaskNode;
class AivTaskStub : public TaskStub {
public:
    AivTaskStub(const RankId rankId, const u32 rankPos, u32 mainStreamPos)
        : TaskStub(TaskTypeStub::AIV_TASK), rankId(rankId), rankPos(rankPos), mainStreamPos(mainStreamPos)
    {
    }
    TaskNode* GetAivStart();
    TaskNode* GetAivEnd();
    void SetAivStart(TaskNode* aivStart);
    void SetAivEnd(TaskNode* aivEnd);
    std::string Describe() const override;
    u32 GetRankId() const;
    u32 GetRankPos() const;
    u32 GetMainStreamPos() const;

private:
    RankId rankId;
    TaskNode* aivStart = nullptr;
    TaskNode* aivEnd = nullptr;
    u32 rankPos;
    u32 mainStreamPos;
};

class TaskStubAivStart : public TaskStub {
public:
    TaskStubAivStart(const RankId rankId, const u32 rankPos)
        : TaskStub(TaskTypeStub::AIV_START), rankId(rankId), rankPos(rankPos)
    {
    }
    std::string Describe() const override;
    RankId GetRankId() const;
    u32 GetRankPos() const;

private:
    RankId rankId;
    u32 rankPos;
};

class TaskStubBlockStart : public TaskStub {
public:
    TaskStubBlockStart(const RankId rankId, const BlockId blockId)
        : TaskStub(TaskTypeStub::BLOCK_START), rankId(rankId), blockId(blockId)
    {
    }
    std::string Describe() const override;
    RankId GetRankId() const;
    BlockId GetBlockId() const;

private:
    RankId rankId;
    BlockId blockId;
};

class TaskStubAivEnd : public TaskStub {
public:
    TaskStubAivEnd(const RankId rankId, const u32 rankPos)
        : TaskStub(TaskTypeStub::AIV_END), rankId(rankId), rankPos(rankPos)
    {
    }
    std::string Describe() const override;
    RankId GetRankId() const;
    u32 GetRankPos() const;

private:
    RankId rankId;
    u32 rankPos;
};

class TaskStubSetFlag : public TaskStub {
public:
    TaskStubSetFlag(pipe_t srcPipe, pipe_t dstPipe, u32 eventId, BlockId blockId, bool isGenFromFree = false)
        : TaskStub(TaskTypeStub::SET_FLAG), srcPipe(srcPipe), dstPipe(dstPipe), eventId(eventId), blockId(blockId), isGenFromFree(isGenFromFree)
    {
    }
    pipe_t GetSrcPipe() const;
    pipe_t GetDstPipe() const;
    u32 GetEventId() const;
    BlockId GetBlockId() const;
    std::string Describe() const override;
    bool IsGenFromFree() const;

private:
    pipe_t srcPipe;
    pipe_t dstPipe;
    u32 eventId;
    BlockId blockId;
    bool isGenFromFree; 
};

class TaskStubWaitFlag : public TaskStub {
public:
    TaskStubWaitFlag(pipe_t srcPipe, pipe_t dstPipe, u32 eventId, BlockId blockId, bool isGenFromAlloc = false)
        : TaskStub(TaskTypeStub::WAIT_FLAG), srcPipe(srcPipe), dstPipe(dstPipe), eventId(eventId), blockId(blockId), isGenFromAlloc(isGenFromAlloc)
    {
    }
    pipe_t GetSrcPipe() const;
    pipe_t GetDstPipe() const;
    u32 GetEventId() const;
    BlockId GetBlockId() const;
    std::string Describe() const override;
    bool IsGenFromAlloc() const;

private:
    pipe_t srcPipe;
    pipe_t dstPipe;
    u32 eventId;
    BlockId blockId;
    bool isGenFromAlloc; 
};

class TaskStubSetValue : public TaskStub {
public:
    TaskStubSetValue(int32_t flagValue)
        : TaskStub(TaskTypeStub::SET_VALUE), flagValue(flagValue)
    {
    }
    std::string Describe() const override;
private:
    int32_t flagValue;
};

class TaskStubSendSync : public TaskStub {
public:
    TaskStubSendSync(int32_t* flagAddr, int32_t flagValue)
        : TaskStub(TaskTypeStub::SEND_SYNC), flagAddr(flagAddr), flagValue(flagValue)
    {
    }
    std::string Describe() const override;
    int32_t* GetFlagAddr() const;
    int32_t GetFlagValue() const;

private:
    int32_t* flagAddr = nullptr;
    int32_t flagValue = 0;
};


class TaskStubRecvSync : public TaskStub {
public:
    TaskStubRecvSync(int32_t* flagAddr, int32_t flagValue)
        : TaskStub(TaskTypeStub::RECV_SYNC), flagAddr(flagAddr), flagValue(flagValue)
    {
    }
    std::string Describe() const override;
    int32_t* GetFlagAddr() const;
    int32_t GetFlagValue() const;

private:
    int32_t* flagAddr = nullptr;
    int32_t flagValue = 0;
};

class TaskStubSendSyncReduce : public TaskStub {
public:
    TaskStubSendSyncReduce(int32_t* flagAddr, int32_t flagValue)
        : TaskStub(TaskTypeStub::SEND_SYNC_REDUCE), flagAddr(flagAddr), flagValue(flagValue)
    {
    }
    std::string Describe() const override;
    int32_t* GetFlagAddr() const;
    int32_t GetFlagValue() const;

private:
    int32_t* flagAddr = nullptr;
    int32_t flagValue = 0;
};

class TaskStubCompValue : public TaskStub {
public:
    TaskStubCompValue(int32_t flagValue)
        : TaskStub(TaskTypeStub::COMP_VALUE), flagValue(flagValue)
    {
    }
    std::string Describe() const override;
    int32_t GetFlagValue() const;

private:
    int32_t flagValue = 0;
};

class TaskStubPipeBarrier : public TaskStub {
public:
    TaskStubPipeBarrier(pipe_t barrierName) 
        : TaskStub(TaskTypeStub::PIPE_BARRIER), pipeBarrierType(barrierName)
    {
    }
    std::string Describe() const override;
    bool IsPipeBarrierAll();
    void SetPipeToPos(pipe_t pipeName, u32 pos);
    u32 GetPos(pipe_t pipeName);

private:
    pipe_t pipeBarrierType;
    std::map<pipe_t, u32> pipeToPos = {};
};

class TaskStubSetFlagShadow : public TaskStub {
public :
    TaskStubSetFlagShadow(RankId neighborRank, u32 srcpipe, u32 dstPipe, u32 srcBlock, u32 dstBlock,  bool isInVirtual)
        : TaskStub(TaskTypeStub::SET_FLAG_SHADOW), neighborRank(neighborRank), srcpipe(srcpipe), dstPipe(dstPipe), 
                   srcBlock(srcBlock), dstBlock(dstBlock), isInVirtual(isInVirtual)
    {
    }
    std::string Describe() const override;
    RankId GetNeighborRank() const;
    bool IsInVirtual() const;
    u32 GetSrcPipe() const;
    u32 GetDstPipe() const;
    u32 GetSrcBlock() const;
    u32 GetDstBlock() const;

private:
    RankId neighborRank;
    u32 srcpipe;
    u32 dstPipe;

    //当不在虚拟流上时， 代表实际block， 反之代表是代表neighborRank的虚拟流
    u32 srcBlock;
    u32 dstBlock;
    bool isInVirtual; //是否在虚拟流上标志位
};

class TaskStubWaitFlagShadow : public TaskStub {
public :
    TaskStubWaitFlagShadow(RankId neighborRank, u32 srcpipe, u32 dstPipe, u32 srcBlock, u32 dstBlock,  bool isInVirtual)
        : TaskStub(TaskTypeStub::WAIT_FLAG_SHADOW), neighborRank(neighborRank), srcpipe(srcpipe), dstPipe(dstPipe), 
                   srcBlock(srcBlock), dstBlock(dstBlock), isInVirtual(isInVirtual)
    {
    }
    std::string Describe() const override;
    RankId GetNeighborRank() const;
    bool IsInVirtual() const;
    u32 GetSrcPipe() const;
    u32 GetDstPipe() const;
    u32 GetSrcBlock() const;
    u32 GetDstBlock() const;

private:
    RankId neighborRank;
    u32 srcpipe;
    u32 dstPipe;

    //当不在虚拟流上时， 代表实际block， 反之代表是代表neighborRank的虚拟流
    u32 srcBlock;
    u32 dstBlock;
    bool isInVirtual; //是否在虚拟流上标志位
};

class TaskStubVirtualRankStart : public TaskStub {
public:
    TaskStubVirtualRankStart(RankId neighborRank)
        : TaskStub(TaskTypeStub::VIRTUAL_RANK_START), neighborRank(neighborRank)
    {
    }
    std::string Describe() const override;
    RankId GetNeighborRank() const;

private:
    RankId neighborRank;
};

std::string GetPipeName(pipe_t pipe);

std::string GetTaskName(TaskTypeStub taskType);
bool IsGenFromSync(TaskStub* task);

}
#endif
