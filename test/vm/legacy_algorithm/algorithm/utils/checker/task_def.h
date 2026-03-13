#ifndef HCCLV1_CHECKER_TASK_DEF_H
#define HCCLV1_CHECKER_TASK_DEF_H

#include <vector>
#include "task_stub.h"
#include "checker_string_util.h"

namespace checker {

using namespace std;

struct TaskNode {
    TaskStub                              *task;
    RankId                                 rankIdx;
    u32                                    queIdx;
    u32                                    pos;
    std::vector<TaskNode*>                 parents  = {};
    std::vector<TaskNode*>                 children = {};
    TaskNode*                              realPeerNode = nullptr;
    u32                                    realqueId = -1;
    u32                                    globalStep = 0;
    u32                                    localStep = 0;
    bool                                   execFlag = false; // for GenGraph, true => executable
    bool                                   travFlag = false; // for GenGraph, true => traversed
    bool                                   procFlag = false; // for Graphrevamp, true => processed
#ifdef HCCL_ALG_ANALYZER_DAVID
    bool                                   ccuBilateral = false; // for GraphRevampBilateralCcu, true => processed
    bool                                   loopParallel = false; // for GraphRevampParallel, true => processed
    bool                                   asynParallel = false; // for GraphRevampParallel, true => processed
#endif
    bool                                   unmatch = false;
    bool                                   genSemanticError = false;

    TaskNode(TaskStub *x, RankId rId, u32 qId, u32 pos) : task(x), rankIdx(rId), queIdx(qId), pos(pos)
    {
    }

    std::string GenPosInfo()
    {
        if (isAivNode) {
            bool isVir = realPeerNode != nullptr;
            u32 rId = isVir ? realPeerNode->rankIdx : rankIdx;
            s32 rPos = isVir ? realPeerNode->rankPos : rankPos;
            s32 bId = isVir ? realPeerNode->blockIdx : blockIdx;
            s32 pId= isVir ? realPeerNode->pipeIdx : pipeIdx;
            s32 pPos = isVir ? realPeerNode->pipePos : pipePos; 
            return StringFormat("[rankId:%d, rankPos:%d, BlockId:%d, pipeId:%d, pipePos:%d]", rId, rPos, bId, pId, pPos);
        }

        u32 rId = 0;
        u32 qId = 0;
        u32 pId = 0;
        if (realPeerNode == nullptr) {
            if (realqueId != -1) {
                qId = realqueId;
            } else {
                qId = queIdx;
            }
            rId = rankIdx;
            pId = pos;
        } else {
            rId = realPeerNode->rankIdx;
            qId = realPeerNode->queIdx;
            pId = realPeerNode->pos;
        }
        return StringFormat("[rankId:%u, queueId:%u, index:%u]", rId, qId, pId);
    }

    //for aiv taskNode
    s32                                    rankPos = -1;
    s32                                    blockIdx = -1;
    s32                                    pipeIdx = -1;
    s32                                    pipePos = -1; 
    bool                                   hasAivTask = false;
    bool                                   isAivNode = false;
    bool                                   hasCcuTask = false;

    TaskNode(TaskStub *x, RankId rId, s32 rPos, s32 bId, s32 pId, u32 pos) 
        : task(x), rankIdx(rId), rankPos(rPos), blockIdx(bId), pipeIdx(pId), pipePos(pos), isAivNode(true)
    {
    }

};

using TaskNodePtr = TaskNode*;

}

#endif