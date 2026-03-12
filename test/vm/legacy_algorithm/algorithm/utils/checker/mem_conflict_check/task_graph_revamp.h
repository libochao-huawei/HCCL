/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: graph revamp header file
 * Author: huangweihao
 * Create: 2024-10-29
 */
#ifndef HCCLV1_GRAPH_REVAMP_H
#define HCCLV1_GRAPH_REVAMP_H

#include <map>
#include <set>
#include <vector>
#include <memory>
#include <queue>

#include "log.h"
#include "task_stub.h"
#include "task_queue_stub.h"
#include "task_graph_generator.h"
#include "llt_common.h"
#include "task_graph_revamp_base.h"

namespace checker {
// vector struct is preserved for multi-QP
// 只维护最后一个TaskNodePtr的信息
using VirtQueMgr = std::map<RankId, std::map<RankId, std::vector<TaskNodePtr>>>;
using VirtAivBlockMgr = std::map<std::pair<RankId, u32>, std::map<std::pair<RankId, u32>, std::vector<TaskNodePtr>>>;

class GraphRevampBilateralSemantics : public GraphRevampBase {
public:
    // GraphRevampBilateralSemantics(uint32_t rankSize) : GraphRevampBase(rankSize) {}
    HcclResult Revamp(TaskNodePtr dummyStart) override;

private:
    HcclResult InitRankHead(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head);
    HcclResult RevampGraph(TaskNodePtr dummyStart, std::map<RankId, TaskNodePtr> &rank2Head);
    HcclResult ProceedNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &visited);

    HcclResult ProcReadNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult ProcSdmaRWNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchBackwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchForwardSdmaRW(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    bool IsVirtualTask(TaskNodePtr node);
    bool IsReadWriteWithSameRank(RankId peerRank, TaskNodePtr candNode);
    bool IsBeingReadOrWrittenTask(TaskNodePtr candNode);
    bool IsProceedParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue, std::set<TaskNodePtr> &visited);

    HcclResult AddBeingRWNodeToVirtualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddBeingRWNodeToVirtualQueWithWait(TaskNodePtr waitNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult InsertNode(TaskNodePtr headNode, TaskNodePtr insertNode);
    HcclResult AddTerminalNodePeerRankVirtualQue(TaskNodePtr candNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult PrepAvailVirtQueTail(const RankId myRank, const RankId remRank, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddVirtQueTail(const RankId myRank, const RankId remRank, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, TaskNodePtr waitFromHeadQueNode);

    HcclResult ProcWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult ProcRdmaWriteNode(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchBackwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult SearchForwardRdmaWrite(TaskNodePtr dummyStart, TaskNodePtr currNode,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddBeingWrittenRdmaWithWait(TaskNodePtr candNode, TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddTerminalNodePeerRankVirtualQue(TaskNodePtr candNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddWaitToCurRankVitualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult TransferCurNodeToVitualQue(TaskNodePtr currNode, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    HcclResult AddTerminalNodeCurRankVirtualQue(TaskNodePtr candNode, RankId peerRank, TaskNodePtr dummyStart,
        std::map<RankId, TaskNodePtr> &rank2Head, VirtQueMgr &virtQueManager);

    //for Aiv revamp
    std::map<RankId, std::map<u32, TaskNodePtr>> rank2AivStart_;
    std::map<RankId, std::map<u32, u32>> rank2AivStartSize_;
    HcclResult ProcAivNode(TaskNodePtr aivTaskNode);
    HcclResult ProcAivRWNode(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult ProcSdmaAivRWNode(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult SearchBackwardSdmaAivRW(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult SearchForwardSdmaAivRW(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult AddBeingAivRWNodeToVirtualQueWithRecvSync(TaskNodePtr recvNode, TaskNodePtr sendNode, 
        TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult AddBeingAivRWNodeToVirtualQue(TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult AddAivTerminalNodePeerRankVirtualQue(TaskNodePtr sendNode, TaskNodePtr recvNode, 
        TaskNodePtr currNode, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult PrepAvailAivVirtQueTail(std::pair<RankId, u32> peerRankAndPosPair, 
        std::pair<RankId, u32> currRankAndPosPair, VirtAivBlockMgr& virtAivBlockManager);
    HcclResult InsertAivNode(TaskNodePtr headNode, TaskNodePtr insertNode);
    HcclResult ProceedAivNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                      std::set<TaskNodePtr> &isVisited);
    bool IsProceedAivParentNode(TaskNodePtr currNode, std::queue<TaskNodePtr> &graphNodeQue,
                                                        std::set<TaskNodePtr> &isVisited);
};
} // namespace Hccl

#endif