/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_TASK_STUB_H
#define SIM_TASK_STUB_H
#include <map>
#include <vector>
#include <sstream>
#include <memory>

#include "enum_factory.h"
#include "sim_common.h"
#include "data_slice.h"
#include "hccl_types.h"
#include "sim_task.h"

namespace HcclSim {
MAKE_ENUM(TaskTypeStub, LOCAL_COPY, LOCAL_REDUCE, LOCAL_BATCH_REDUCE, LOCAL_POST_TO, LOCAL_WAIT_FROM, POST, WAIT, READ,
          READ_REDUCE, WRITE, WRITE_REDUCE, BEING_READ, BEING_READ_REDUCE, BEING_WRITTEN, BEING_WRITTEN_REDUCE,
          LOCAL_POST_TO_SHADOW, LOCAL_WAIT_FROM_SHADOW, AIV_TASK, SET_VALUE, SET_FLAG, WAIT_FLAG, SEND_SYNC, RECV_SYNC,
          SEND_SYNC_REDUCE, COMP_VALUE, PIPE_BARRIER, CCU_GRAPH, LOOP_START, LOOP_END, SUB_GRAPH_END, SET_FLAG_SHADOW, WAIT_FLAG_SHADOW, 
          AIV_START, BLOCK_START, AIV_END, VIRTUAL_RANK_START, GRAPH_SEPARATE)
          
MAKE_ENUM(NotifyTypeStub, READY, FIN, FIN_ACK, CCU, INVALID_A)
MAKE_ENUM(LinkProtoStub, SDMA, RDMA, CCU, INVALID_A)

using QId = u32;
using BlockId = u32;

struct LinkInfo {
    LinkProtoStub linkProto;

    LinkInfo(LinkProtoStub proto)
    {
        linkProto = proto;
    }

    std::string Describe() const
    {
        return StringFormat("link prototyp=%s", linkProto.Describe().c_str());
    }
};

class TaskStub {
public:
    explicit TaskStub(TaskTypeStub type) : type(type)
    {
    }
    virtual ~TaskStub()                  = default;
    virtual std::string Describe() const = 0;

    const TaskTypeStub GetType() const
    {
        return type;
    }

    virtual const LinkProtoStub GetLinkType() const
    {
        return LinkProtoStub::INVALID_A;
    }

protected:
    TaskTypeStub type;
};

class TaskStubLocalCopy : public TaskStub {
public:
    TaskStubLocalCopy(const DataSlice &srcSlice, const DataSlice &dstSlice, bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::LOCAL_COPY), srcSlice(srcSlice), dstSlice(dstSlice), isGenFromSync(isGenFromSync)
    {
    }
    ~TaskStubLocalCopy() override = default;
    std::string Describe() const override;

    const DataSlice &GetSrcSlice() const;
    const DataSlice &GetDstSlice() const;
    bool IsGenFromSync();

private:
    DataSlice srcSlice;
    DataSlice dstSlice;
    bool isGenFromSync;
};

class TaskStubLocalReduce : public TaskStub {
public:
    TaskStubLocalReduce(const DataSlice &srcSlice, const DataSlice &dstSlice, HcclDataType dataType,
                        HcclReduceOp reduceOp, bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::LOCAL_REDUCE), srcSlice(srcSlice), dstSlice(dstSlice), dataType(dataType),
          reduceOp(reduceOp), isGenFromSync(isGenFromSync)
    {
    }
    ~TaskStubLocalReduce() override = default;
    std::string Describe() const override;

    const DataSlice &GetSrcSlice() const;
    const DataSlice &GetDstSlice() const;
    const HcclDataType   GetDataType() const;
    const HcclReduceOp   GetReduceOp() const;
    bool IsGenFromSync();

private:
    DataSlice srcSlice;
    DataSlice dstSlice;
    HcclDataType  dataType;
    HcclReduceOp  reduceOp;
    bool isGenFromSync;
};

// LocalBatchReduce后续应该要支持低精度模式，即输入的数据类型与输出的数据类型不一致
class TaskStubLocalBatchReduce : public TaskStub {
public:
    TaskStubLocalBatchReduce(const std::vector<DataSlice> &srcSlices, const DataSlice &dstSlice, HcclDataType dataType,
        HcclReduceOp reduceOp)
        : TaskStub(TaskTypeStub::LOCAL_BATCH_REDUCE), srcSlices(srcSlices), dstSlice(dstSlice), dataType(dataType),
          reduceOp(reduceOp)
    {
    }
    ~TaskStubLocalBatchReduce() override = default;
    std::string Describe() const override;

    const std::vector<DataSlice>& GetSrcSlices() const;
    const DataSlice& GetSrcSlice(u32 index) const;
    const DataSlice& GetDstSlice() const;
    const HcclDataType GetDataType() const;
    const HcclReduceOp GetReduceOp() const;

private:
    std::vector<DataSlice> srcSlices;
    DataSlice dstSlice;
    HcclDataType  dataType;
    HcclReduceOp  reduceOp;
};


class TaskStubRead : public TaskStub {
public:
    TaskStubRead(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                 const DataSlice &remoteSlice, bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::READ), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), isGenFromSync(isGenFromSync)
    {
    }
    ~TaskStubRead() override = default;
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    const LinkInfo     GetLinkInfo() const;
    bool IsGenFromSync();

private:
    RankId  remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    bool isGenFromSync;
};

class TaskStubReadReduce : public TaskStub {
public:
    TaskStubReadReduce(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                       const DataSlice &remoteSlice, HcclDataType dataType, HcclReduceOp reduceOp,
                       bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::READ_REDUCE), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), dataType(dataType), reduceOp(reduceOp), isGenFromSync(isGenFromSync)
    {
    }
    ~TaskStubReadReduce() override = default;
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    const HcclDataType      GetDataType() const;
    const HcclReduceOp      GetReduceOp() const;
    const LinkInfo     GetLinkInfo() const;
    bool IsGenFromSync();

private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    HcclDataType  dataType;
    HcclReduceOp  reduceOp;
    bool isGenFromSync;
};

class TaskStubWrite : public TaskStub {
public:
    TaskStubWrite(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                  const DataSlice &remoteSlice, bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::WRITE), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), isGenFromSync(isGenFromSync)
    {
    }
    ~TaskStubWrite() override = default;
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    const LinkInfo     GetLinkInfo() const;
    bool IsGenFromSync();

private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    bool isGenFromSync;
};

class TaskStubWriteReduce : public TaskStub {
public:
    TaskStubWriteReduce(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                        const DataSlice &remoteSlice, HcclDataType dataType, HcclReduceOp reduceOp,
                        bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::WRITE_REDUCE), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), dataType(dataType), reduceOp(reduceOp), isGenFromSync(isGenFromSync)
    {
    }
    ~TaskStubWriteReduce() override = default;
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    const HcclDataType      GetDataType() const;
    const HcclReduceOp      GetReduceOp() const;
    const LinkInfo     GetLinkInfo() const;
    bool IsGenFromSync();

private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    HcclDataType  dataType;
    HcclReduceOp  reduceOp;
    bool isGenFromSync;
};

class TaskStubPost : public TaskStub {
public:
    TaskStubPost(const RankId remoteRank, const LinkInfo &link, u32 topicId,
                 NotifyTypeStub notifyType = NotifyTypeStub::INVALID_A, std::string tag = "INVALID", void *curCcuTask = nullptr)
        : TaskStub(TaskTypeStub::POST), remoteRank(remoteRank), link(link), topicId(topicId), topicIdBack(topicId),
          notifyType(notifyType), tag(tag)
    {
        ccuTaskPtr_ = reinterpret_cast<uint64_t>(curCcuTask);
    }
    ~TaskStubPost() override = default;
    std::string Describe() const override;

    std::string Describe(bool isdeadlock);
    RankId               GetRemoteRank() const;
    void                 SetRemoteRank(RankId rankId);
    const LinkProtoStub  GetLinkType() const override;
    const uint64_t       GetNotifyId() const;
    void                 SetNotifyId(uint64_t id);
    const u32            GetTopicId() const;
    void                 SetTopicId(u32 id);
public:
    uint64_t ccuTaskPtr_{0}; // 保存所属ccu子图首节点（用于获取queNum）

private:
    RankId         remoteRank;
    LinkInfo       link;
    uint64_t       topicId;
    u32            topicIdBack;
    NotifyTypeStub notifyType;
    std::string    tag;
};

class TaskStubWait : public TaskStub {
public:
    TaskStubWait(const RankId remoteRank, const LinkInfo &link, u32 topicId,
                 NotifyTypeStub notifyType = NotifyTypeStub::INVALID_A, std::string tag = "INVALID", void *curCcuTask = nullptr)
        : TaskStub(TaskTypeStub::WAIT), remoteRank(remoteRank), link(link), notifyId(topicId),
          notifyType(notifyType), tag(tag)
    {
        ccuTaskPtr_ = reinterpret_cast<uint64_t>(curCcuTask);
    }
    ~TaskStubWait() override = default;
    std::string Describe() const override;

    std::string Describe(bool isdeadlock);
    RankId               GetRemoteRank() const;
    void                 SetRemoteRank(RankId rankId);
    const LinkProtoStub  GetLinkType() const override;
    const uint64_t       GetNotifyId() const;
    void                 SetNotifyId(uint64_t id);
public:
    uint64_t ccuTaskPtr_{0}; // 保存所属ccu子图首节点（用于获取queNum）

private:
    RankId          remoteRank;
    LinkInfo        link;
    uint64_t        notifyId;
    NotifyTypeStub notifyType;
    std::string    tag;
};

constexpr uint32_t INVALID_QID = 0xffffffff; // 无效的指令队列
class TaskStubLocalPostTo : public TaskStub {
public:
    TaskStubLocalPostTo(u32 topicId, QId postQid = INVALID_QID, QId waitQid = INVALID_QID)
        : TaskStub(TaskTypeStub::LOCAL_POST_TO), topicId(topicId), topicIdBack(topicId), postQid(postQid), waitQid(waitQid)
    {
    }
    ~TaskStubLocalPostTo() override = default;
    std::string Describe() const override;

    const uint64_t       GetNotifyId() const;
    const uint64_t       GetNotifyIdBack() const;
    void                 SetNotifyId(uint64_t id);

    void SetPostQid(QId qid);
    void SetWaitQid(QId qid);

    QId GetPostQid() const;
    QId GetWaitQid() const;
    u32 GetTopicId() const;
    void SetTopicId(u32 id);
    u32 GetTopicIdBack() const;

private:
    u32 topicId;
    u32 topicIdBack;
    QId postQid{INVALID_QID};
    QId waitQid{INVALID_QID};
};

class TaskStubLocalWaitFrom : public TaskStub {
public:
    TaskStubLocalWaitFrom(u32 notifyId, u32 postQid = INVALID_QID, u32 waitQid = INVALID_QID)
        : TaskStub(TaskTypeStub::LOCAL_WAIT_FROM), notifyId(notifyId), postQid(postQid), waitQid(waitQid)
    {
    }
    ~TaskStubLocalWaitFrom() override = default;
    std::string Describe() const override;

    const uint64_t       GetNotifyId() const;
    void                 SetNotifyId(uint64_t id);

private:
    uint64_t notifyId;
    u32 postQid{INVALID_QID};
    u32 waitQid{INVALID_QID};
};

class TaskStubLocalPostToShadow : public TaskStub {
public:
    TaskStubLocalPostToShadow(const RankId neighborRank, u32 curQueId, u32 peerQueId)
        : TaskStub(TaskTypeStub::LOCAL_POST_TO_SHADOW), neighborRank(neighborRank), curQueId(curQueId), peerQueId(peerQueId)
    {
    }
    std::string Describe() const override;
    RankId GetNeighborRank() const;
private:
    RankId neighborRank;
    u32 curQueId;
    u32 peerQueId;
};

class TaskStubLocalWaitFromShadow : public TaskStub {
public:
    TaskStubLocalWaitFromShadow(const RankId neighborRank, u32 curQueId, u32 peerQueId)
        : TaskStub(TaskTypeStub::LOCAL_WAIT_FROM_SHADOW), neighborRank(neighborRank), curQueId(curQueId), peerQueId(peerQueId)
    {
    }
    std::string Describe() const override;
    RankId GetNeighborRank() const;
private:
    RankId neighborRank;
    u32 curQueId;
    u32 peerQueId;
};

class TaskStubBeingRead : public TaskStub {
public:
    TaskStubBeingRead(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                      const DataSlice &remoteSlice, bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::BEING_READ), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), isGenFromSync(isGenFromSync)
    {
    }
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    bool IsGenFromSync();
    
private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    bool isGenFromSync;
};

class TaskStubBeingReadReduce : public TaskStub {
public:
    TaskStubBeingReadReduce(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                            const DataSlice &remoteSlice, HcclDataType dataType, HcclReduceOp reduceOp,
                            bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::BEING_READ_REDUCE), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), dataType(dataType), reduceOp(reduceOp), isGenFromSync(isGenFromSync)
    {
    }
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    const HcclDataType      GetDataType() const;
    const HcclReduceOp      GetReduceOp() const;
    bool IsGenFromSync();

private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    HcclDataType  dataType;
    HcclReduceOp  reduceOp;
    bool isGenFromSync;
};

class TaskStubBeingWritten : public TaskStub {
public:
    TaskStubBeingWritten(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                         const DataSlice &remoteSlice, bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::BEING_WRITTEN), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), isGenFromSync(isGenFromSync)
    {
    }
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    bool IsGenFromSync();

private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    bool isGenFromSync;
};

class TaskStubBeingWrittenReduce : public TaskStub {
public:
    TaskStubBeingWrittenReduce(const RankId remoteRank, const LinkInfo &link, const DataSlice &localSlice,
                               const DataSlice &remoteSlice, HcclDataType dataType, HcclReduceOp reduceOp,
                               bool isGenFromSync = false)
        : TaskStub(TaskTypeStub::BEING_WRITTEN_REDUCE), remoteRank(remoteRank), link(link), localSlice(localSlice),
          remoteSlice(remoteSlice), dataType(dataType), reduceOp(reduceOp), isGenFromSync(isGenFromSync)
    {
    }
    std::string Describe() const override;

    RankId              GetRemoteRank() const;
    const LinkProtoStub GetLinkType() const override;
    const DataSlice    &GetLocalSlice() const;
    const DataSlice    &GetRemoteSlice() const;
    const HcclDataType      GetDataType() const;
    const HcclReduceOp      GetReduceOp() const;
    bool IsGenFromSync();

private:
    RankId    remoteRank;
    LinkInfo  link;
    DataSlice localSlice;
    DataSlice remoteSlice;
    HcclDataType  dataType;
    HcclReduceOp  reduceOp;
    bool isGenFromSync;
};

// 标识展开的Loop指令序列开始
class TaskStubLoopStart : public TaskStub {
public:
    TaskStubLoopStart(uint32_t loopIdx, uint32_t loopGroupIdx)
        : TaskStub(TaskTypeStub::LOOP_START), loopIdx(loopIdx), loopGroupIdx(loopGroupIdx)
    {}
    std::string Describe() const override;

public:
    uint32_t loopIdx{0};       // loop序号：loopGroup内唯一
    uint32_t loopGroupIdx{0};  // loopGroup序号：ccuInsGroup内唯一
};

// 标识展开的Loop指令序列结束
class TaskStubLoopEnd : public TaskStub {
public:
    TaskStubLoopEnd(uint32_t loopIdx, uint32_t loopGroupIdx)
        : TaskStub(TaskTypeStub::LOOP_END), loopIdx(loopIdx), loopGroupIdx(loopGroupIdx)
    {}
    std::string Describe() const override;

public:
    uint32_t loopIdx{0};       // loop序号：loopGroup内唯一
    uint32_t loopGroupIdx{0};  // loopGroup序号：ccuInsGroup内唯一
};

// 子图分隔节点：防止广度优先搜索错误执行，生成错误语义
class TaskStubGraphSeparate : public TaskStub {
public:
    TaskStubGraphSeparate()
        : TaskStub(TaskTypeStub::GRAPH_SEPARATE)
    {}
    std::string Describe() const override;
};

using SingleTaskQueue = std::vector<std::vector<std::shared_ptr<TaskStub>>>;
using AllRankTaskQueues = std::map<RankId, SingleTaskQueue>;

}
#endif