/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: TaskGraphGenerator
 * Author: shenyutian
 * Create: 2024-07-29
 */

#include "task_graph_generator.h"
#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#endif

using namespace Hccl;

namespace checker {

HcclResult TaskGraphGenerator::GenGraph(const TaskQueueStub &taskQueues, TaskNodePtr dummyStart)
{
    for (RankId rankId = 0; rankId < taskQueues.GetRankSize(); rankId++) {
        CHK_RET(GenGraph4Rank(taskQueues.GetTaskQueueOfRank(rankId), rankId, dummyStart));
        HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], local dependency graph generation done.", rankId);
    }
    HCCL_DEBUG("[TaskGraphGenerator] rankSize [%u] and numChildren of dummyStart [%u].", taskQueues.GetRankSize(),
               dummyStart->children.size());

    /*
    Mismatch may occur when: 1) fail to generate local dependency graph correctly --> ERROR
                             2) a group prim is placed at the begining of the primitive queue
    */

    CHK_RET(GenGraphInterRanks(dummyStart));

    //Handling the synchronization relationship between AIV, if exist.
    if (dummyStart->hasAivTask) {
        CHK_RET(GenGraphInterAivs(AivTaskQueueStub::Global()->GetAllAivTasks().rank2AivTask));
    }
#ifdef HCCL_ALG_ANALYZER_DAVID
    // 将CCU微码序列转换为Task子图
    CHK_RET(Hccl::GenCcuGraph(dummyStart));
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::GenGraph4Rank(const SingleRankTaskQueues *rankTaskQueues, const RankId rankIdx,
                                             TaskNodePtr dummyStart)
{
    std::vector<TaskNodePtr> rankNodeQue;      // executable task nodes
    SeenLocalPost            seenLocalPosts;   // seen Local Posts
    u64                      unmatchedCnt = 0; // for deadlock checking

    CHK_PRT_RET(InitRankNodeQue(rankTaskQueues, rankIdx, dummyStart, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to  init rankNodeQue.", rankIdx),
                HcclResult::HCCL_E_INTERNAL);

    while (!rankNodeQue.empty()) {
        if (unmatchedCnt >= rankNodeQue.size()) {
            // DeadLocking
            for (auto &rankNodeUnmatch : rankNodeQue) {
                rankNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of LOCAL_POST_TO and LOCAL_WAIT_FROM.");
            LocateUnmatchedNode(rankNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = rankNodeQue[0];
        rankNodeQue.erase(rankNodeQue.begin());

        switch (currNode->task->GetType()) {
            case TaskTypeStub::LOCAL_COPY:
            case TaskTypeStub::LOCAL_REDUCE:
            case TaskTypeStub::POST:
            case TaskTypeStub::WAIT:
            case TaskTypeStub::READ:
            case TaskTypeStub::READ_REDUCE:
            case TaskTypeStub::WRITE:
            case TaskTypeStub::WRITE_REDUCE:
            case TaskTypeStub::CCU_GRAPH:
                dummyStart->hasCcuTask =  true;
                CHK_RET(ExecFlitPrim(rankTaskQueues, currNode, rankNodeQue, unmatchedCnt));
                break;

            case TaskTypeStub::LOCAL_POST_TO:
                CHK_RET(ExecLocalPostPrim(rankTaskQueues, currNode, rankNodeQue, seenLocalPosts, unmatchedCnt));
                break;

            case TaskTypeStub::LOCAL_WAIT_FROM:
                CHK_RET(ExecLocalWaitPrim(rankTaskQueues, currNode, rankNodeQue, seenLocalPosts, unmatchedCnt));
                break;

            case TaskTypeStub::AIV_TASK:
                dummyStart->hasAivTask =  true;
                CHK_RET(ExecAivTaskPrim(rankTaskQueues, currNode, rankNodeQue, unmatchedCnt));
                break;

            default:
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], taskType not supported.", rankIdx);
                return HcclResult::HCCL_E_INTERNAL;
        }
    }
    if (!seenLocalPosts.empty()){
        for (auto &localPost : seenLocalPosts) {
            localPost->unmatch = true;
            HCCL_ERROR("[TaskGraphGenerator] unmatched local_post: %s.",
                       localPost->GenPosInfo().c_str());
            return HcclResult::HCCL_E_INTERNAL;
	    }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::InitRankNodeQue(const SingleRankTaskQueues *rankTaskQueues, const RankId rankIdx,
                                               TaskNodePtr dummyStart, std::vector<TaskNodePtr> &rankNodeQue)
{
    // first task of master queue
    auto currNode = std::make_shared<TaskNode>(rankTaskQueues->GetTask(0, 0).get(), rankIdx, 0, 0);
    CHK_PTR_NULL(currNode);
    nodes_.push_back(currNode);

    dummyStart->children.push_back(currNode.get());
    currNode->parents.push_back(dummyStart);
    rankNodeQue.push_back(currNode.get());
    HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], connect dummyStart -> first taskNode of master queue, put taskNode in "
               "rankNodeQue",
               rankIdx);

    // first task of slave queues: first task should be local wait
    for (u32 qIdx = 1; qIdx < rankTaskQueues->taskQueues.size(); qIdx++) {
        // 表明这条流未下发任何Task
        if (rankTaskQueues->GetQueTaskNum(qIdx) == 0) {
            continue;
        }
        auto currNode = std::make_shared<TaskNode>(rankTaskQueues->GetTask(qIdx, 0).get(), rankIdx, qIdx, 0);
        CHK_PTR_NULL(currNode);
        nodes_.push_back(currNode);
        CHK_PRT_RET(
            currNode->task->GetType() != TaskTypeStub::LOCAL_WAIT_FROM,
            HCCL_ERROR("[TaskGraphGenerator] Rank[%d], Que [%u], first task of slave queue should be localWaitFrom.",
                       currNode->rankIdx, currNode->queIdx),
            HcclResult::HCCL_E_INTERNAL);
        rankNodeQue.push_back(currNode.get());
        HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], put first taskNode of slave queues in rankNodeQue", rankIdx);
    }

    return HcclResult::HCCL_SUCCESS;
}

void TaskGraphGenerator::LocateUnmatchedNode(const std::vector<TaskNodePtr> &rankNodeQue)
{
    auto rankNodeIter = rankNodeQue.begin();
    for (; rankNodeIter != rankNodeQue.end(); rankNodeIter++) {
        HCCL_ERROR("[TaskGraphGenerator] unmatched task locates in: %s", (*rankNodeIter)->GenPosInfo().c_str());
        return;
    }
    HCCL_ERROR("[TaskGraphGenerator] Checker internal error, deadlock is not due to mismatch of local sync.");
    return;
}

HcclResult TaskGraphGenerator::ExecFlitPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                            std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt)
{
    // curr -> its nxt, push nxt to nodeQue
    CHK_PRT_RET(ConnectNextAndPushInQue(rankTaskQueues, currNode, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);
    unmatchedCnt = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ConnectNextAndPushInQue(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                                       std::vector<TaskNodePtr> &rankNodeQue)
{
    if (currNode->pos < rankTaskQueues->GetQueTaskNum(currNode->queIdx) - 1) {
        auto nxtNode = std::make_shared<TaskNode>(rankTaskQueues->GetTask(currNode->queIdx, currNode->pos + 1).get(),
                                                  currNode->rankIdx, currNode->queIdx, currNode->pos + 1);
        CHK_PTR_NULL(nxtNode);
        nodes_.push_back(nxtNode);
        nxtNode->parents.push_back(currNode);
        currNode->children.push_back(nxtNode.get());
        rankNodeQue.push_back(nxtNode.get());
    } else {
        HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], end of current Que [%u]: TrimType [%s].", currNode->rankIdx,
                   currNode->queIdx, currNode->task->GetType().Describe().c_str());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecLocalPostPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                                 u64 &unmatchedCnt)
{
    // curr -> its nxt, nxt in nodeQue
    CHK_PRT_RET(ConnectNextAndPushInQue(rankTaskQueues, currNode, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, "LocalPostTo"),
                HcclResult::HCCL_E_INTERNAL);

    seenLocalPosts.push_back(currNode);
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecLocalWaitPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                                 std::vector<TaskNodePtr> &rankNodeQue, SeenLocalPost &seenLocalPosts,
                                                 u64 &unmatchedCnt)
{
    std::vector<TaskNodePtr>::iterator postIter;
    for (postIter = seenLocalPosts.begin(); postIter != seenLocalPosts.end(); postIter++) {
        if (IsSemPeer((*postIter), currNode)) {
            (*postIter)->children.push_back(currNode);
            currNode->parents.push_back((*postIter));    // local_post_to -> curr local_wait_from
            (seenLocalPosts).erase(postIter); // remove local_post_to from seenLocalPosts
            // curr local_wait_from -> its nxt
            CHK_PRT_RET(
                ConnectNextAndPushInQue(rankTaskQueues, currNode, rankNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency graph: TaskType [%s].",
                            currNode->rankIdx, "LocalWaitFrom"),
                HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;
        }
    }

    // corresponding post not seen yet
    rankNodeQue.push_back(currNode); // local_wait_from cannot be executed, push back to node que
    unmatchedCnt++;
    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsSemPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode)
{
    // check type
    if ((postNode->task->GetType() != TaskTypeStub::LOCAL_POST_TO)
        || (waitNode->task->GetType() != TaskTypeStub::LOCAL_WAIT_FROM)) {
        return false;
    }

    // check queId
    TaskStubLocalPostTo   *localPostTo   = dynamic_cast<TaskStubLocalPostTo *>(postNode->task);
    TaskStubLocalWaitFrom *localWaitFrom = dynamic_cast<TaskStubLocalWaitFrom *>(waitNode->task);

    return ((localPostTo->GetTopicId() == localWaitFrom->GetTopicId()) &&
        (localPostTo->GetPostQid() == localWaitFrom->GetPostQid()) &&
        (localPostTo->GetWaitQid() == localWaitFrom->GetWaitQid()));
}

HcclResult TaskGraphGenerator::GenGraphInterRanks(TaskNodePtr dummyStart)
{
    std::vector<TaskNodePtr> graphNodeQue;       // executable primnodes
    SeenInterRankPosts       seenInterRankPosts; // seen inter-rank Posts
    u64                      unmatchedCnt = 0;   // for deadlock checking

    CHK_PRT_RET(ExecNode4Graph(dummyStart, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Fail to init graphNodeQue."), HcclResult::HCCL_E_INTERNAL);

    while (!graphNodeQue.empty()) {
        if (unmatchedCnt >= graphNodeQue.size()) {
            for (auto &graphNodeUnmatch : graphNodeQue) {
                graphNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of inter-rank Post/Wait.");
            LocateUnmatchedNode(graphNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = graphNodeQue[0];
        graphNodeQue.erase(graphNodeQue.begin());

        CHK_PRT_RET(ProcNode4Graph(currNode, graphNodeQue, seenInterRankPosts, unmatchedCnt)
                        != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to proceed taskNode.", currNode->rankIdx),
                    HcclResult::HCCL_E_INTERNAL);
    }

    bool hasChanged = false;
    if (!seenInterRankPosts.empty()) {
        for (auto &curRankPosts : seenInterRankPosts) {
            for (auto &peerRankPosts : curRankPosts.second) {
                for (auto &post : peerRankPosts.second) {
                    post->unmatch = true;
                    HCCL_ERROR("[TaskGraphGenerator] unmatched inter-rank post: %s, PeerRank [%d],  ",
                               post->GenPosInfo().c_str(), peerRankPosts.first);
                    hasChanged = true;
                }
            }
        }
    }
    if (hasChanged) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecNode4Graph(TaskNodePtr node, std::vector<TaskNodePtr> &graphNodeQue)
{
    node->execFlag = true;
    std::vector<TaskNodePtr>::iterator childIter = node->children.begin();
    for (; childIter != node->children.end(); childIter++) {
        if (!(*childIter)->travFlag) {
            (*childIter)->travFlag = true;
            graphNodeQue.push_back((*childIter));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                              SeenInterRankPosts &seenInterRankPosts, u64 &unmatchedCnt)
{
    // taskNode not executable
    if (!IsExecutable(currNode)) {
        graphNodeQue.push_back(currNode);
        unmatchedCnt++;
        HCCL_DEBUG("[TaskGraphGenerator] taskNode not executable, push back to the queue.");
        return HcclResult::HCCL_SUCCESS;
    }

    switch (currNode->task->GetType()) {
        case TaskTypeStub::LOCAL_COPY:
        case TaskTypeStub::LOCAL_REDUCE:
        case TaskTypeStub::LOCAL_POST_TO:
        case TaskTypeStub::LOCAL_WAIT_FROM:
        case TaskTypeStub::READ:
        case TaskTypeStub::READ_REDUCE:
        case TaskTypeStub::WRITE:
        case TaskTypeStub::WRITE_REDUCE:
        case TaskTypeStub::AIV_TASK:
        case TaskTypeStub::CCU_GRAPH:
            CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to execute taskNode.", currNode->rankIdx),
                        HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::POST:
            // put in seenInterRankPosts
            CHK_RET(ProcInterRankPostNode4Graph(currNode, graphNodeQue, seenInterRankPosts, unmatchedCnt));
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::WAIT:
            // check if peer task in seenInterRankPosts
            CHK_RET(ProcInterRankWaitNode4Graph(currNode, graphNodeQue, seenInterRankPosts, unmatchedCnt));
            break;

        default:
            HCCL_ERROR("[TaskGraphGenerator] taskType not supported.");
            return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsExecutable(TaskNodePtr currNode)
{
    std::vector<TaskNodePtr>::iterator parentIter = currNode->parents.begin();
    for (; parentIter != currNode->parents.end(); parentIter++) {
        TaskNodePtr tmpParent = *parentIter;
        if (!tmpParent->execFlag) {
            return false;
        }
    }
    return true;
}

HcclResult TaskGraphGenerator::ProcInterRankPostNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankPosts &seenInterRankPosts,
                                                           u64                &unmatchedCnt)
{
    RankId        currRank = currNode->rankIdx;
    TaskStubPost *post     = dynamic_cast<TaskStubPost *>(currNode->task);
    RankId        peerRank = post->GetRemoteRank();

    if (seenInterRankPosts.find(currRank) == seenInterRankPosts.end()) {
        std::vector<TaskNodePtr> tmpPosts;
        tmpPosts.push_back(currNode);
        std::map<RankId, std::vector<TaskNodePtr>> tmpRankPosts;
        tmpRankPosts.insert(std::make_pair(peerRank, tmpPosts));
        seenInterRankPosts.insert(std::make_pair(currRank, tmpRankPosts));
    } else {
        if (seenInterRankPosts[currRank].find(peerRank) == seenInterRankPosts[currRank].end()) {
            std::vector<TaskNodePtr> tmpPosts;
            tmpPosts.push_back(currNode);
            seenInterRankPosts[currRank].insert(std::make_pair(peerRank, tmpPosts));
        } else {
            seenInterRankPosts[currRank][peerRank].push_back(currNode);
        }
    }
    CHK_PRT_RET(
        ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
        HCCL_ERROR("[TaskGraphGenerator] Fail to execute node %s: TaskType [%s].",
                   currNode->GenPosInfo().c_str(), currNode->task->GetType().Describe().c_str()),
        HcclResult::HCCL_E_INTERNAL);
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcInterRankWaitNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankPosts &seenInterRankPosts,
                                                           u64                &unmatchedCnt)
{
    RankId        currRank = currNode->rankIdx;
    TaskStubWait *wait     = dynamic_cast<TaskStubWait *>(currNode->task);
    RankId        peerRank = wait->GetRemoteRank();
    if ((seenInterRankPosts.find(peerRank) != seenInterRankPosts.end())
        && (seenInterRankPosts[peerRank].find(currRank) != seenInterRankPosts[peerRank].end())) {
        std::vector<TaskNodePtr>::iterator postIter = seenInterRankPosts[peerRank][currRank].begin();
        for (; postIter != seenInterRankPosts[peerRank][currRank].end(); postIter++) {
            if (IsPostWaitPeer((*postIter), currNode)) {
                HCCL_DEBUG("[TaskGraphGenerator] peer PostNode of current WaitNode has already been seen.");
                (*postIter)->children.push_back(currNode);
                currNode->parents.push_back((*postIter));
                seenInterRankPosts[peerRank][currRank].erase(postIter); // remove post from seenInterRankPosts
                CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                            HCCL_ERROR("[TaskGraphGenerator] Fail to execute node %s: TaskType [%s].",
                                       currNode->GenPosInfo().c_str(),
                                       currNode->task->GetType().Describe().c_str()),
                            HcclResult::HCCL_E_INTERNAL);
                unmatchedCnt = 0;
                return HcclResult::HCCL_SUCCESS;
            }
        }
    }

    HCCL_DEBUG("[TaskGraphGenerator] peer PostNode of current WaitNode has not been seen yet.");
    graphNodeQue.push_back(currNode);
    unmatchedCnt++;
    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsPostWaitPeer(const TaskNodePtr postNode, const TaskNodePtr waitNode)
{
    TaskStubPost *post = dynamic_cast<TaskStubPost *>(postNode->task);
    TaskStubWait *wait = dynamic_cast<TaskStubWait *>(waitNode->task);

    // check rankId
    if ((postNode->rankIdx != wait->GetRemoteRank()) || (waitNode->rankIdx != post->GetRemoteRank())) {
        return false;
    }

    // check LinkType
    if (post->GetLinkType() != wait->GetLinkType()) {
        return false;
    }

    // check topicId
    if (post->GetTopicId() != wait->GetTopicId()) {
        return false;
    }

    return post->GetNotifyType() == wait->GetNotifyType();
}

HcclResult TaskGraphGenerator::ExecAivTaskPrim(const SingleRankTaskQueues *rankTaskQueues, TaskNodePtr currNode,
                                               std::vector<TaskNodePtr> &rankNodeQue, u64 &unmatchedCnt)
{
    //init aivStart for rank, record aiv tasknode to map
    auto curRankPos = ((AivTaskStub*)(currNode->task))->GetRankPos();
    std::shared_ptr<TaskStub> aivStartTask = std::make_shared<TaskStubAivStart>(currNode->rankIdx, curRankPos);
    auto aivStart = std::make_shared<TaskNode>(aivStartTask.get(), currNode->rankIdx, curRankPos, -1, -1, -2);
    nodes_.push_back(aivStart);
    ((AivTaskStub *)currNode->task)->SetAivStart(aivStart.get());
    AivTaskQueueStub::Global()->SetRank2AivStart(currNode->rankIdx, aivStart.get());
    AivTaskQueueStub::Global()->GetAllAivTasks().headAndTailResource.push_back(aivStartTask);

    std::shared_ptr<TaskStub> aivEndTask = std::make_shared<TaskStubAivEnd>(currNode->rankIdx, curRankPos);
    auto aivEnd = std::make_shared<TaskNode>(aivEndTask.get(), currNode->rankIdx, curRankPos, -1, -1, -3);
    nodes_.push_back(aivEnd);
    ((AivTaskStub *)currNode->task)->SetAivEnd(aivEnd.get());
    AivTaskQueueStub::Global()->GetAllAivTasks().headAndTailResource.push_back(aivEndTask);
    

    //match the dependency relationships between the same pipe and different pipes in the same AIV
    for (auto& curRankAivTask : AivTaskQueueStub::Global()->GetAllAivTasks().rsb2AivTaskQueues[currNode->rankIdx]) {
        auto curBlock = curRankAivTask.first;
        AivSingleBlockTaskQues* curBlockAivTaskQueuesPtr = curRankAivTask.second[curRankPos];
        std::shared_ptr<TaskStub> blockStartTask = std::make_shared<TaskStubBlockStart>(currNode->rankIdx, curBlock);
        auto blockStart = std::make_shared<TaskNode>(blockStartTask.get(), currNode->rankIdx, curRankPos, curBlock, -1, -1);
        nodes_.push_back(blockStart);
        AivTaskQueueStub::Global()->GetAllAivTasks().headAndTailResource.push_back(blockStartTask);
        aivStart->children.push_back(blockStart.get());
        blockStart->parents.push_back(aivStart.get());
        CHK_PRT_RET(GenGraph4Aiv(curBlockAivTaskQueuesPtr, currNode->rankIdx, curBlock, blockStart.get(), aivEnd.get()) != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to generate dependency aiv graph: TaskType [%s].",
                               currNode->rankIdx, currNode->task->GetType().Describe().c_str()), HcclResult::HCCL_E_INTERNAL);
    }

    //merge PipeBarrierAll Node
    std::map<TaskStub*, std::vector<TaskNode*>> *barrierRecord = &AivTaskQueueStub::Global()->GetAllAivTasks().pipeBarrierAllRecord;
    for (auto& taskNodes : *barrierRecord) {
        TaskNode* firstNode = taskNodes.second[0];
        TaskStubPipeBarrier* firstStub = (TaskStubPipeBarrier*)firstNode->task;
        firstStub->SetPipeToPos((pipe_t)firstNode->pipeIdx, firstNode->pipePos);

        for (int idx = 1; idx < taskNodes.second.size(); idx++) {
            firstStub->SetPipeToPos((pipe_t)taskNodes.second[idx]->pipeIdx, taskNodes.second[idx]->pipePos);

            for (auto& taskNode : taskNodes.second[idx]->parents) {
                firstNode->parents.push_back(taskNode);
                taskNode->children.push_back(firstNode);
                taskNode->children.erase(std::remove(taskNode->children.begin(), taskNode->children.end(), taskNodes.second[idx]), taskNode->children.end());
            }

            for (auto& taskNode : taskNodes.second[idx]->children) {
                firstNode->children.push_back(taskNode);
                taskNode->parents.push_back(firstNode);
                taskNode->parents.erase(std::remove(taskNode->parents.begin(), taskNode->parents.end(), taskNodes.second[idx]), taskNode->parents.end());
            }

        }
    }

    barrierRecord->clear();

    CHK_RET(ExecFlitPrim(rankTaskQueues, currNode, rankNodeQue, unmatchedCnt));
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::GenGraph4Aiv(const AivSingleBlockTaskQues* aivTaskQueues, RankId rankIdx,
                                            BlockId blockIdx, TaskNodePtr blockStart, TaskNodePtr aivEnd)
{
    std::vector<TaskNodePtr> aivNodeQue;
    std::vector<TaskNodePtr> SeenSetFlag;
    u64 unmatchedCnt = 0;

    //init the first tasknode of each pipe in the same block
    auto pipeNum = aivTaskQueues->taskQueues.size();
    for (int currPipe = 0; currPipe < pipeNum; currPipe++) {
        if(aivTaskQueues->GetPipeTaskNum((pipe_t)currPipe) == 0) {
            continue;
        }

        auto currNode = std::make_shared<TaskNode>(aivTaskQueues->GetTask((pipe_t)currPipe, 0).get(), rankIdx, blockStart->rankPos, blockIdx, currPipe, 0);
        CHK_PTR_NULL(currNode);
        nodes_.push_back(currNode);
        blockStart->children.push_back(currNode.get());
        currNode->parents.push_back(blockStart);
        aivNodeQue.push_back(currNode.get());
    }

    while (!aivNodeQue.empty()) {
        if (unmatchedCnt >= aivNodeQue.size()) {
            for (auto& aivNodeUnmatch : aivNodeQue) {
                aivNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of setFlag and WaitFlag.");
            LocateUnmatchedNode(aivNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = aivNodeQue[0];
        aivNodeQue.erase(aivNodeQue.begin());

        switch (currNode->task->GetType()) {
            case TaskTypeStub::LOCAL_COPY:
            case TaskTypeStub::LOCAL_REDUCE:
            case TaskTypeStub::READ:
            case TaskTypeStub::READ_REDUCE:
            case TaskTypeStub::WRITE:
            case TaskTypeStub::WRITE_REDUCE:
            case TaskTypeStub::SEND_SYNC:
            case TaskTypeStub::SEND_SYNC_REDUCE:
            case TaskTypeStub::RECV_SYNC:
            case TaskTypeStub::COMP_VALUE:
            case TaskTypeStub::SET_VALUE:
                CHK_RET(ExecAivFlitPrim(aivTaskQueues, currNode, aivNodeQue, unmatchedCnt, aivEnd));
                break;

            case TaskTypeStub::SET_FLAG:
                CHK_RET(ExecSetFlagPrim(aivTaskQueues, currNode, aivNodeQue, SeenSetFlag, unmatchedCnt, aivEnd));
                break;

            case TaskTypeStub::WAIT_FLAG:
                CHK_RET(ExecWaitFlagPrim(aivTaskQueues, currNode, aivNodeQue, SeenSetFlag, unmatchedCnt, aivEnd));
                break;

            case TaskTypeStub::PIPE_BARRIER:
                CHK_RET(ExecPipeBarrierPrim(aivTaskQueues, currNode, aivNodeQue, unmatchedCnt, aivEnd));
                break;

            default:
                HCCL_ERROR("[TaskGraphGenerator] taskType not supported.");
                return HcclResult::HCCL_E_INTERNAL;

        }
    }

    if (!SeenSetFlag.empty()) {
        for (auto& setFlag : SeenSetFlag) {
            bool isGenFromfree = ((TaskStubSetFlag*)(setFlag->task))->IsGenFromFree();
            if (!isGenFromfree) {
                setFlag->unmatch = true;
                HCCL_ERROR("[TaskGraphGenerator] unmatched setFlag: rankId=%d, blockId=%d, pipeId=%s, pipePOs=%d, %s", setFlag->rankIdx, 
                            setFlag->blockIdx, GetPipeName((pipe_t)(setFlag->pipeIdx)).c_str(), setFlag->pipePos, setFlag->task->Describe().c_str());
                return HcclResult::HCCL_E_INTERNAL;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecAivFlitPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                            std::vector<TaskNodePtr> &aivNodeQue, u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
    // curr -> its nxt, push nxt to nodeQue
    CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);
    unmatchedCnt = 0;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ConnectNextAivTaskNodeAndPushInQue(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                                                  std::vector<TaskNodePtr> &aivNodeQue, TaskNodePtr aivEnd)
{
    if (currNode->pipePos < aivTaskQueues->GetPipeTaskNum((pipe_t)(currNode->pipeIdx)) - 1) {
        auto nxtNode = std::make_shared<TaskNode>(aivTaskQueues->GetTask((pipe_t)(currNode->pipeIdx), currNode->pipePos + 1).get(),
                                                  currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->pipePos + 1);
        CHK_PTR_NULL(nxtNode);
        nodes_.push_back(nxtNode);
        nxtNode->parents.push_back(currNode);
        currNode->children.push_back(nxtNode.get());
        aivNodeQue.push_back(nxtNode.get());
        return HcclResult::HCCL_SUCCESS;
    }

    if (currNode->pipePos == aivTaskQueues->GetPipeTaskNum((pipe_t)(currNode->pipeIdx)) - 1) {
        currNode->children.push_back(aivEnd);
        aivEnd->parents.push_back(currNode);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_DEBUG("[TaskGraphGenerator] Rank [%d], Block [%d], Pipe [%d], end of current Pipe [%d]: TrimType [%s].", currNode->rankIdx,
                   currNode->blockIdx, currNode->pipeIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecSetFlagPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                               std::vector<TaskNodePtr> &aivNodeQue, std::vector<TaskNodePtr> &SeenSetFlag,
                                               u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
     CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);

    SeenSetFlag.push_back(currNode);
    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecWaitFlagPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                               std::vector<TaskNodePtr> &aivNodeQue, std::vector<TaskNodePtr> &SeenSetFlag,
                                               u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
    std::vector<TaskNodePtr>::iterator setFlagIter;
    for (setFlagIter = SeenSetFlag.begin(); setFlagIter != SeenSetFlag.end(); setFlagIter++) {
        if(IsSetWaitPeer((*setFlagIter), currNode)) {
            (*setFlagIter)->children.push_back(currNode);
            currNode->parents.push_back((*setFlagIter));
            (SeenSetFlag).erase(setFlagIter);
             CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;
        }
    }

    aivNodeQue.push_back(currNode);
    unmatchedCnt++;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecPipeBarrierPrim(const AivSingleBlockTaskQues* aivTaskQueues, TaskNodePtr currNode,
                                                   std::vector<TaskNodePtr> &aivNodeQue, u64 &unmatchedCnt, TaskNodePtr aivEnd)
{
    TaskStubPipeBarrier* currBarrierStub = (TaskStubPipeBarrier*)(currNode->task);
    if (currBarrierStub->IsPipeBarrierAll()) {
        map<TaskStub*, std::vector<TaskNode*>> *pipeRecord = &(AivTaskQueueStub::Global()->GetAllAivTasks().pipeBarrierAllRecord);
        (*pipeRecord)[currNode->task].push_back(currNode);
    }

     CHK_PRT_RET(ConnectNextAivTaskNodeAndPushInQue(aivTaskQueues, currNode, aivNodeQue, aivEnd) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Rank [%d], RankPos [%d], Block [%d], Pipe [%d], fail to generate dependency graph: TaskType [%s].",
                           currNode->rankIdx, currNode->rankPos, currNode->blockIdx, currNode->pipeIdx, currNode->task->GetType().Describe().c_str()),
                HcclResult::HCCL_E_INTERNAL);

    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

bool TaskGraphGenerator::IsSetWaitPeer(const TaskNodePtr setFlagNode, const TaskNodePtr waitFlagNode)
{
    if ((setFlagNode->task->GetType() != TaskTypeStub::SET_FLAG)
        || (waitFlagNode->task->GetType() != TaskTypeStub::WAIT_FLAG)) {
        return false;
    }

    TaskStubSetFlag *setFlag = dynamic_cast<TaskStubSetFlag*>(setFlagNode->task);
    TaskStubWaitFlag *waitFlag = dynamic_cast<TaskStubWaitFlag*>(waitFlagNode->task);

    if (setFlag->GetBlockId() != waitFlag->GetBlockId()) {
        return false;
    }

    if (setFlag->GetSrcPipe() != waitFlag->GetSrcPipe()) {
        return false;
    }

    if (setFlag->GetDstPipe() != waitFlag->GetDstPipe()) {
        return false;
    }

    if (setFlag->IsGenFromFree() != waitFlag->IsGenFromAlloc()) {
        return false;
    }

    return (setFlag->GetEventId() == waitFlag->GetEventId());
}

HcclResult TaskGraphGenerator::GenGraphInterAivs(std::map<RankId, std::vector<TaskNode*>> &rank2AivTask)
{
    std::vector<TaskNodePtr> graphNodeQue;
    SeenInterRankSendSync seenInterAivSendSync;
    u64 unmatchedCnt = 0;

    CHK_PRT_RET(ExecAivNode4Graph(rank2AivTask, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);

    while (!graphNodeQue.empty()) {
        if (unmatchedCnt >= graphNodeQue.size()) {
            for (auto& aivNodeUnmatch : graphNodeQue) {
                aivNodeUnmatch->unmatch = true;
            }
            HCCL_ERROR("[TaskGraphGenerator] deadLocking occurs due to mismatch of SendSync/Reduce and RecvSync.");
            LocateUnmatchedNode(graphNodeQue);
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr currNode = graphNodeQue[0];
        graphNodeQue.erase(graphNodeQue.begin());

        CHK_PRT_RET(ProcAivNode4Graph(currNode, graphNodeQue, seenInterAivSendSync, unmatchedCnt)
                        != HcclResult::HCCL_SUCCESS,
                    HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to proceed taskNode.", currNode->rankIdx),
                    HcclResult::HCCL_E_INTERNAL);
    }

    if (!seenInterAivSendSync.empty()) {
        for (auto& curGmAddrRecord : seenInterAivSendSync) {
            if (!std::get<2>(curGmAddrRecord.second)) {
                for (auto& sendSyncNode : std::get<1>(curGmAddrRecord.second)) {
                    sendSyncNode->unmatch = true;
                    HCCL_WARNING("[TaskGraphGenerator] unmatched inter-aiv sendSyncNode: GmAddr [%d],  ",
                               curGmAddrRecord.first);
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ExecAivNode4Graph(std::map<RankId, std::vector<TaskNode*>> &rank2AivTask, std::vector<TaskNodePtr> &graphNodeQue)
{
    for (auto& aivPair : rank2AivTask) {
        for (auto& aivStart : aivPair.second) {
            aivStart->execFlag = true;
            for (auto& blockStart : aivStart->children) {
                CHK_PRT_RET(ExecNode4Graph(blockStart, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                            HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS; 
}

HcclResult TaskGraphGenerator::ProcAivNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                 SeenInterRankSendSync &seenInterRankSendSync,
                                                 u64 &unmatchedCnt)
{
    if (!IsExecutable(currNode)) {
        graphNodeQue.push_back(currNode);
        unmatchedCnt++;
        HCCL_DEBUG("[TaskGraphGenerator] taskNode not executable, push back to the queue.");
        return HcclResult::HCCL_SUCCESS;
    }
    if(currNode->task == nullptr){
        currNode->execFlag = true;
        HCCL_DEBUG("[TaskGraphGenerator] aiv end node.");
        return HcclResult::HCCL_SUCCESS;
    }
    switch (currNode->task->GetType()) {
        case TaskTypeStub::LOCAL_COPY:
        case TaskTypeStub::LOCAL_REDUCE:
        case TaskTypeStub::READ:
        case TaskTypeStub::READ_REDUCE:
        case TaskTypeStub::WRITE:
        case TaskTypeStub::WRITE_REDUCE:
        case TaskTypeStub::COMP_VALUE:
        case TaskTypeStub::SET_VALUE:
        case TaskTypeStub::SET_FLAG:
        case TaskTypeStub::WAIT_FLAG:
        case TaskTypeStub::PIPE_BARRIER:
        case TaskTypeStub::AIV_END:
            CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Rank [%d], fail to execute taskNode.", currNode->rankIdx),
                        HcclResult::HCCL_E_INTERNAL);
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::SEND_SYNC:
        case TaskTypeStub::SEND_SYNC_REDUCE:
            CHK_RET(ProcInterAivSendSyncNode4Graph(currNode, graphNodeQue, seenInterRankSendSync, unmatchedCnt));
            return HcclResult::HCCL_SUCCESS;

        case TaskTypeStub::RECV_SYNC:
            CHK_RET(ProcInterAivRecvSyncNode4Graph(currNode, graphNodeQue, seenInterRankSendSync, unmatchedCnt));
            return HcclResult::HCCL_SUCCESS;

        default:
            HCCL_ERROR("[TaskGraphGenerator] taskType not supported.");
            return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcInterAivSendSyncNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                           SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt)
{
    int32_t* flagAddr;
    int32_t  flagValue;
    switch (currNode->task->GetType()) {
        case TaskTypeStub::SEND_SYNC:
            flagAddr = ((TaskStubSendSync*)(currNode->task))->GetFlagAddr();
            flagValue = ((TaskStubSendSync*)(currNode->task))->GetFlagValue();
            break;
        case TaskTypeStub::SEND_SYNC_REDUCE:
            flagAddr = ((TaskStubSendSyncReduce*)(currNode->task))->GetFlagAddr();
            flagValue = ((TaskStubSendSyncReduce*)(currNode->task))->GetFlagValue();
            break;
        default:
            HCCL_ERROR("Node is not expected tpye, SendSync or SendSyncReduce.");
            return HcclResult::HCCL_E_INTERNAL;
    }
    if (seenInterRankSendSync.find(flagAddr) == seenInterRankSendSync.end()) {
        std::pair<int32_t*, std::tuple<int32_t, std::vector<TaskNodePtr>, bool>> sendAndRecvFlag
            = std::make_pair(flagAddr, std::make_tuple(flagValue, std::vector<TaskNodePtr>(), false));
        seenInterRankSendSync.insert(sendAndRecvFlag);
    } else {
        if (std::get<2>(seenInterRankSendSync[flagAddr])) {
            std::get<0>(seenInterRankSendSync[flagAddr]) = flagValue;
            std::get<1>(seenInterRankSendSync[flagAddr]).clear();
            std::get<2>(seenInterRankSendSync[flagAddr]) = false;
        } else {
            if (currNode->task->GetType() == TaskTypeStub::SEND_SYNC) {
                std::get<0>(seenInterRankSendSync[flagAddr]) = flagValue;
            } else if (currNode->task->GetType() == TaskTypeStub::SEND_SYNC_REDUCE) {
                std::get<0>(seenInterRankSendSync[flagAddr]) += flagValue;
            }
        }
    }
    std::get<1>(seenInterRankSendSync[flagAddr]).push_back(currNode);

    CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);

    unmatchedCnt = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskGraphGenerator::ProcInterAivRecvSyncNode4Graph(TaskNodePtr currNode, std::vector<TaskNodePtr> &graphNodeQue,
                                                              SeenInterRankSendSync &seenInterRankSendSync, u64 &unmatchedCnt)
{
    int32_t* flagAddr = ((TaskStubRecvSync*)(currNode->task))->GetFlagAddr();
    int32_t  flagValue = ((TaskStubRecvSync*)(currNode->task))->GetFlagValue();

    if (seenInterRankSendSync.find(flagAddr) == seenInterRankSendSync.end()) {
        graphNodeQue.push_back(currNode);
        unmatchedCnt++;
        return HcclResult::HCCL_SUCCESS;
    } else {
        auto sendSyncFlagValue = std::get<0>(seenInterRankSendSync[flagAddr]);
        if (sendSyncFlagValue == flagValue) {
            for (auto& sendNodePtr : std::get<1>(seenInterRankSendSync[flagAddr])) {
                sendNodePtr->children.push_back(currNode);
                currNode->parents.push_back(sendNodePtr);
            }
            CHK_PRT_RET(ExecNode4Graph(currNode, graphNodeQue) != HcclResult::HCCL_SUCCESS,
                        HCCL_ERROR("[TaskGraphGenerator] Fail to init graphAivNodeQue."), HcclResult::HCCL_E_INTERNAL);
            std::get<2>(seenInterRankSendSync[flagAddr]) = true;
            unmatchedCnt = 0;
            return HcclResult::HCCL_SUCCESS;
        }    
    }

    graphNodeQue.push_back(currNode);
    unmatchedCnt++;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hccl