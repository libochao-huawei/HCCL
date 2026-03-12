/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: HccL checker主控流程实现
 * Create: 2024-04-18
 */
#include <unordered_map>
#include "checker.h"
#include "rank_info_recorder.h"
#include "task_stub.h"
#include "externalinput.h"
#include "utils_stub.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "task_graph_revamp.h"
#include "check_rank_mem.h"
#include "log.h"
#include "check_utils.h"
#include "singletask_check.h"
#include "task_check_op_semantics.h"
#include "link_type_recorder.h"
#include "mem_layout.h"
#include "alg_adapter_v1_interface.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#include "task_graph_revamp_parallel.h"
#include "task_graph_revamp_bilateral_ccu.h"
#endif

using namespace std;
using namespace hccl;

namespace checker {

Checker::Checker()
{
}

Checker::~Checker()
{
    // 清理上次执行残留的数据
    TaskQueueStub::Global()->GetAllRankTasks().Clear();
    AivTaskQueueStub::Global()->GetAllAivTasks().Clear();
    MemLayout::Global()->Reset();

    DataDumper::Global()->Close();
    DataDumper::Global()->ClearData();

    RankInfoRecorder::Global()->Reset();

    for (auto& ele : toDeleteCopyTaskNodeResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }

    for (auto& ele : toDeleteCopyTaskResource_) {
        if (ele == nullptr) {
            continue;
        }
        delete ele;
    }
}

void Checker::EnableTaskPrint()
{
    enablePrimQuePrint_ = true;
    return;
}

void Checker::EnableGraphicDump()
{
    enableGraphicDump_ = true;
    return;
}

void Checker::EnableGraphPrint()
{
    enableGraphPrint_ = true;
    return;
}

void Checker::CloseRankMemCheck()
{
    closeRankMemCheck_ = true;
    return;
}

void Checker::SetDumpFileName(const string &fileName)
{
    DataDumper::Global()->SetFileName(fileName);
    return;
}

void Checker::PrintTask()
{
    for (int rankId = 0; rankId < (int)TaskQueueStub::Global()->GetRankSize(); rankId++) {
        printf("=======================================================\n");
        printf("rank id is %d\n", rankId);

        SingleRankTaskQueues* singleRankTaskQues = TaskQueueStub::Global()->GetTaskQueueOfRank(rankId);
        for (int j = 0; j < (int)singleRankTaskQues->taskQueues.size(); j++) {
            printf("-------------------------------------------------------\n");
            printf("stream/queue id is %d\n", j);
            for(int k = 0; k < (*singleRankTaskQues)[j].size(); k++) {
                string tempstr = (*singleRankTaskQues)[j][k]->Describe();
                printf("[rankId:%d, queueId:%d, index:%d] %s\n", rankId, j, k, tempstr.c_str());
            }
        }
    }
    return;
}

void Checker::PrintGraphRevamp(TaskNodePtr head)
{
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> printedNode;
    for(int i = 0; i < head->children.size(); i++) {
        printedNode.insert(head->children[i]);
        candTaskNodePtr.push_back(head->children[i]);
    }

    while(!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        // print node info
        printf("\n");
        printf("curNode is %s\n", curNode->task->Describe().c_str());
        printf("-----------------------\n");
        for (int i = 0; i < curNode->parents.size(); i++) {
            if (curNode->parents[i]->task != nullptr) {
                printf("parents[%d] is %s\n", i, curNode->parents[i]->task->Describe().c_str());
            }
        }
        printf("-----------------------\n");
        for (int i = 0; i < curNode->children.size(); i++) {
            printf("children[%d] is %s\n", i, curNode->children[i]->task->Describe().c_str());
        }
        printf("-----------------------\n");
        printf("\n");
        for (int i = 0; i < curNode->children.size(); i++) {
            std::set<TaskNodePtr> ::iterator it = printedNode.find(curNode->children[i]);
            if (it == printedNode.end()) {
                candTaskNodePtr.push_back(curNode->children[i]);
                printedNode.insert(curNode->children[i]);
            }
        }
    }
    return;
}

void Checker::PrintAivGraph(bool isCopy = false)
{
    std::set<TaskNode*> parentSet;
    std::queue<TaskNode*> bfsQueue;

    std::map<RankId, std::vector<TaskNode*>> aivStartSet;
    if (isCopy) {
        aivStartSet = AivTaskQueueStub::Global()->GetAllAivTasks().copyRank2AivTask;
    } else {
        aivStartSet = AivTaskQueueStub::Global()->GetAllAivTasks().rank2AivTask;
    }

    for (auto rank2AivTask : aivStartSet) {
        for (auto AivTask : rank2AivTask.second) {
            bfsQueue.push(AivTask);
        }
    }

    while(!bfsQueue.empty()) {
        auto currNode = bfsQueue.front();
        if (parentSet.find(currNode) != parentSet.end()) {
            bfsQueue.pop();
            continue;
        }

        parentSet.insert(currNode);
        printf("\n");

        printf("=========================curNode info=============================\n");
        //如果节点为特殊节点PipeBarrierAll， 此节点会同时存在于三条流上，且每条流上的位置不一定相同，0代表SCALAR， 1代表MET2， 2代表MTE3 
        if (currNode->task->GetType() == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)currNode->task)->IsPipeBarrierAll()) {
            auto pipeStub = (TaskStubPipeBarrier*)currNode->task;
            printf("[%d, %d, %d, [[0, %d], [1, %d], [2, %d]]]%s", currNode->rankIdx, currNode->rankPos, currNode->blockIdx,
                    pipeStub->GetPos(pipe_t::PIPE_S), pipeStub->GetPos(pipe_t::PIPE_MTE2), pipeStub->GetPos(pipe_t::PIPE_MTE3), currNode->task->Describe().c_str());
        } else {
            printf("[%d, %d, %d, %d, %d]%s", currNode->rankIdx, currNode->rankPos, currNode->blockIdx,
                    currNode->pipeIdx, currNode->pipePos, currNode->task->Describe().c_str());
        }
        printf("\n\n");
        printf("-------------------------parents info-----------------------------\n");
        if (currNode->parents.size() == 0) {
            printf("currnode doesn't have parents\n\n");
        } else {
            for (auto parent : currNode->parents) {
                if (parent->task->GetType() == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)parent->task)->IsPipeBarrierAll()) {
                    auto pipeStub = (TaskStubPipeBarrier*)parent->task;
                    printf("[%d, %d, %d, [[0, %d], [1, %d], [2, %d]]]%s", parent->rankIdx, parent->rankPos, parent->blockIdx, pipeStub->GetPos(pipe_t::PIPE_S),
                           pipeStub->GetPos(pipe_t::PIPE_MTE2), pipeStub->GetPos(pipe_t::PIPE_MTE3), parent->task->Describe().c_str());
                } else {
                    printf("[%d, %d, %d, %d, %d]%s", parent->rankIdx, parent->rankPos, parent->blockIdx,
                            parent->pipeIdx, parent->pipePos, parent->task->Describe().c_str());
                }
                printf("\n");
            }
            printf("\n");
        }

        printf("-------------------------children info---------------------------\n");
        if (currNode->children.size() == 0) {
            printf("currnode doesn't have children\n");
        } else {
            for (auto child : currNode->children) {
                if (child->task->GetType() == TaskTypeStub::PIPE_BARRIER && ((TaskStubPipeBarrier*)child->task)->IsPipeBarrierAll()) {
                    auto pipeStub = (TaskStubPipeBarrier*)child->task;
                    printf("[%d, %d, %d, [[0, %d], [1, %d], [2, %d]]]%s", child->rankIdx, child->rankPos, child->blockIdx, pipeStub->GetPos(pipe_t::PIPE_S),
                           pipeStub->GetPos(pipe_t::PIPE_MTE2), pipeStub->GetPos(pipe_t::PIPE_MTE3), child->task->Describe().c_str());
                } else {
                    printf("[%d, %d, %d, %d, %d]%s", child->rankIdx, child->rankPos, child->blockIdx,
                    child->pipeIdx, child->pipePos, child->task->Describe().c_str());
                }
            bfsQueue.push(child);
            printf("\n");
            }
        }

        bfsQueue.pop();
        printf("\n=========================curNode end==============================\n");
        printf("\n");
    }
}

void Checker::PrintAivTask()
{
    AivTaskQueueStub::Global()->PrintAivTask();
}

void Checker::CopyTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode)
{
    // 遍历两遍，先将所有节点拷贝出来，再建立父子关系
    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode; // 用来收录原节点到新节点的映射
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> isVisited;

    originNode2copyNode[originNode] = copyNode;
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
    }

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        TaskStub* newNode = curNode->task;
        TaskNodePtr newNodePtr = new TaskNode(newNode, curNode->rankIdx, curNode->queIdx, curNode->pos);
        toDeleteCopyTaskNodeResource_.push_back(newNodePtr);
        originNode2copyNode[curNode] = newNodePtr;

        for (auto &child : curNode->children) {
            if (isVisited.find(child) == isVisited.end()) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }

    isVisited.clear();
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
        copyNode->children.push_back(originNode2copyNode[originNode->children[i]]);
    }
    while(!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());
        for (auto &parent : curNode->parents) {
            originNode2copyNode[curNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (auto &child : curNode->children) {
            originNode2copyNode[curNode]->children.push_back(originNode2copyNode[child]);
            if (isVisited.count(child) == 0) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }
}

void Checker::CopyAivTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode)
{
    std::map<TaskNodePtr, TaskNodePtr> originNode2copyNode;
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> isVisited;
    std::vector<TaskNodePtr> aivBlockNodePtr;

    originNode2copyNode[originNode] = copyNode;
    for (const auto &child : originNode->children) {
        candTaskNodePtr.push_back(child);
        isVisited.insert(child);
    }

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr.front();
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        TaskStub *newNode = curNode->task;
        if (newNode != nullptr && newNode->GetType() == TaskTypeStub::AIV_TASK) {
            newNode = new AivTaskStub(curNode->rankIdx, ((AivTaskStub*)(curNode->task))->GetRankPos(), ((AivTaskStub*)(curNode->task))->GetMainStreamPos());
            toDeleteCopyTaskResource_.push_back(newNode);

            auto aivStart = new TaskNode(((AivTaskStub*)(curNode->task))->GetAivStart()->task, curNode->rankIdx, ((AivTaskStub*)(curNode->task))->GetRankPos(), -1, -1, -2);
            AivTaskQueueStub::Global()->SetAllCopyAivStart(curNode->rankIdx, aivStart);
            ((AivTaskStub *)newNode)->SetAivStart(aivStart);
            toDeleteCopyTaskNodeResource_.push_back(aivStart);
            originNode2copyNode[((AivTaskStub *)curNode->task)->GetAivStart()] = aivStart;
            aivBlockNodePtr.push_back(((AivTaskStub *)curNode->task)->GetAivStart());

            std::vector<TaskNodePtr> aivTaskNodePtr;
            for (auto &it : ((AivTaskStub *)curNode->task)->GetAivStart()->children) {
                aivTaskNodePtr.push_back(it);
            }
            while (!aivTaskNodePtr.empty()) {
                TaskNodePtr aivNode = aivTaskNodePtr.front();
                aivTaskNodePtr.erase(aivTaskNodePtr.begin());
                TaskStub *aivTask = aivNode->task;
                TaskNodePtr aivNodePtr =
                    new TaskNode(aivTask, aivNode->rankIdx, aivNode->rankPos, aivNode->blockIdx, aivNode->pipeIdx, aivNode->pipePos);
                toDeleteCopyTaskNodeResource_.push_back(aivNodePtr);
                originNode2copyNode[aivNode] = aivNodePtr;
                for (const auto &child : aivNode->children) {
                        if (isVisited.find(child) == isVisited.end()) {
                            isVisited.insert(child);
                            aivTaskNodePtr.push_back(child);
                        }
                }
            }
        }
        TaskNodePtr newNodePtr = new TaskNode(newNode, curNode->rankIdx, curNode->queIdx, curNode->pos);
        toDeleteCopyTaskNodeResource_.push_back(newNodePtr);
        originNode2copyNode[curNode] = newNodePtr;

        for (const auto &child : curNode->children) {
            if (isVisited.find(child) == isVisited.end()) {
            isVisited.insert(child);
            candTaskNodePtr.push_back(child);
            }
        }
    }

    isVisited.clear();

    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
        copyNode->children.push_back(originNode2copyNode[originNode->children[i]]);
    }
    candTaskNodePtr.insert(candTaskNodePtr.end(), aivBlockNodePtr.begin(), aivBlockNodePtr.end());
    isVisited.insert(aivBlockNodePtr.begin(), aivBlockNodePtr.end());

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr.front();
        candTaskNodePtr.erase(candTaskNodePtr.begin());
        for (const auto &parent : curNode->parents) {
            originNode2copyNode[curNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (const auto &child : curNode->children) {
            originNode2copyNode[curNode]->children.push_back(originNode2copyNode[child]);
            if (isVisited.find(child) == isVisited.end()) {
            isVisited.insert(child);
            candTaskNodePtr.push_back(child);
            }
        }
    }
}

#ifdef HCCL_ALG_ANALYZER_DAVID
HcclResult Checker::CopyCcuTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode, uint32_t rankNum)
{
    // 遍历两遍，先将所有节点拷贝出来，再建立父子关系
    Hccl::Ori2NewNodeMap originNode2copyNode; // 用来收录原节点到新节点的映射
    std::vector<TaskNodePtr> candTaskNodePtr;
    std::set<TaskNodePtr> isVisited;

    originNode2copyNode[originNode] = copyNode;
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
    }

    // 记录所有ccu子图的旧->新节点映射关系：方便查询&恢复跨rank的父子关系。
    // vector索引：rankId   map索引：ccu graph describe
    std::vector<Hccl::CcuOri2NewNodeMap> ccuOrigin2CopyNodes;
    std::vector<std::pair<Hccl::TaskStubPtr, Hccl::TaskStubPtr>> ccuGraphs;
    ccuOrigin2CopyNodes.resize(rankNum);

    while (!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());

        TaskStub* newNode = curNode->task;
        if (newNode->GetType() == TaskTypeStub::CCU_GRAPH) {
            TaskStub* newCcu = nullptr;
            // 拷贝ccu子图节点
            CHK_RET(Hccl::CopyCcuSubGraphNode(newNode, &newCcu, ccuGraphs, ccuOrigin2CopyNodes));
            newNode = newCcu;
            toDeleteCopyTaskResource_.push_back(newNode);
        }
        TaskNodePtr newNodePtr = new TaskNode(newNode, curNode->rankIdx, curNode->queIdx, curNode->pos);
        toDeleteCopyTaskNodeResource_.push_back(newNodePtr);
        originNode2copyNode[curNode] = newNodePtr;

        for (auto &child : curNode->children) {
            if (isVisited.find(child) == isVisited.end()) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }
    // 恢复ccu子图内部的连接关系
    CHK_RET(Hccl::CopyCcuSubGraphConnection(ccuGraphs, ccuOrigin2CopyNodes));
    // 恢复外层拓扑图的连接关系
    isVisited.clear();
    for (int i = 0; i < originNode->children.size(); i++) {
        candTaskNodePtr.push_back(originNode->children[i]);
        isVisited.insert(originNode->children[i]);
        copyNode->children.push_back(originNode2copyNode[originNode->children[i]]);
    }
    while(!candTaskNodePtr.empty()) {
        TaskNodePtr curNode = candTaskNodePtr[0];
        candTaskNodePtr.erase(candTaskNodePtr.begin());
        for (auto &parent : curNode->parents) {
            originNode2copyNode[curNode]->parents.push_back(originNode2copyNode[parent]);
        }
        for (auto &child : curNode->children) {
            originNode2copyNode[curNode]->children.push_back(originNode2copyNode[child]);
            if (isVisited.count(child) == 0) {
                isVisited.insert(child);
                candTaskNodePtr.push_back(child);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
#endif

#ifndef HCCL_ALG_ANALYZER_DAVID
HcclResult Checker::Check(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta)
{
    // 产生Task序列
    TaskQuesGenerator gen;
    CHK_RET(gen.Run(checkerOpParam, topoMeta));

    // 打印task队列
    if (enablePrimQuePrint_) {
        PrintTask();
    }

    if (enableGraphicDump_) {
        DataDumper::Global()->Enable();
    }

    u32 rankNum = RankInfoRecorder::Global()->GetRankSize();

    DataDumper::Global()->SetRankSize(rankNum);

    SingleTaskCheck taskChecker;
    CHK_RET(taskChecker.CheckSlaveTaskQueue());

    // 成图以及对图做各种检验
    HcclResult ret = CheckPrimGraphs(checkerOpParam, rankNum);
    if (ret != HcclResult::HCCL_SUCCESS) {
        DataDumper::Global()->SerializeToFile();
        return ret;
    }

    DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_SUCCESS);
    // 序列化出文件
    DataDumper::Global()->SerializeToFile();

    return HcclResult::HCCL_SUCCESS;
}
#endif

#ifdef HCCL_ALG_ANALYZER_DAVID // 只有执行David相关用例时才编译该代码 checker 2.0
HcclResult Checker::CheckA5Aicpu(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta)
{
    Hccl::TaskQuesGeneratorV2 gen;
    Hccl::DavidAlgConfig config;
    CHK_RET(gen.Run(checkerOpParam, topoMeta, config));
    // 打印task队列
    if (enablePrimQuePrint_) {
        PrintTask();
    }

    SingleTaskCheck taskChecker;
    CHK_RET(taskChecker.CheckSlaveTaskQueue());

    u32 rankSize_ = GetRankNumFormTopoMeta(topoMeta);

    // 成图以及对图做各种检验
    CHK_RET(CheckPrimGraphs(checkerOpParam, rankSize_));
    return HcclResult::HCCL_SUCCESS;
}
#endif

HcclResult Checker::CheckPrimGraphs(CheckerOpParam &checkerOpParam, u32 rankNum)
{
    TaskNode dummyStart = TaskNode(nullptr, -1, 0, 0);
    TaskNode dummyStartCopy = TaskNode(nullptr, -1, 0, 0);
    TaskGraphGenerator graphGenerator;
    HcclResult ret = graphGenerator.GenGraph(*(TaskQueueStub::Global()), &dummyStart);

    if (ret != HcclResult::HCCL_SUCCESS) {
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::GEN_GRAPH_FAILED);
        return ret;
    }

    SingleTaskCheck taskChecker;
    CHK_RET(taskChecker.CheckTaskMem(&dummyStart));

    if (dummyStart.hasAivTask) {
        CopyAivTaskGraph(&dummyStart, &dummyStartCopy);
        dummyStartCopy.hasAivTask = true;
    }
#ifdef HCCL_ALG_ANALYZER_DAVID
    else if (dummyStart.hasCcuTask) {
        std::cout<<"zhf-copy task..."<<std::endl;
        CHK_RET(CopyCcuTaskGraph(&dummyStart, &dummyStartCopy, rankNum));
        dummyStartCopy.hasCcuTask = true;
    }
#endif
    else {
        CopyTaskGraph(&dummyStart, &dummyStartCopy);
    }

    CHK_RET(RankMemCheck(dummyStart, dummyStartCopy, checkerOpParam, rankNum));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult Checker::RankMemCheck(TaskNode &dummyStart, TaskNode &dummyStartCopy, CheckerOpParam &checkerOpParam, u32 rankNum)
{
    GraphRevampBilateralSemantics graphRevamp;
#ifdef HCCL_ALG_ANALYZER_DAVID
    GraphRevampParallel parallelRevamp;
    GraphRevampBilateralCcu ccuBilateralRevamp;
#endif
    HcclResult ret = HcclResult::HCCL_SUCCESS;
    if (!closeRankMemCheck_) {
#ifdef HCCL_ALG_ANALYZER_DAVID
        if(dummyStartCopy.hasCcuTask) {
            clock_t ccuStart = clock();
            // 并行化改造：异步节点、Loop指令块
            ret = parallelRevamp.Revamp(&dummyStartCopy);
            if (ret != HcclResult::HCCL_SUCCESS) {
                return ret;
            }
            clock_t afterCcuParallel = clock();

            // 单边->双边语义: CCU子图
            ret = ccuBilateralRevamp.Revamp(&dummyStartCopy);
            if (ret != HcclResult::HCCL_SUCCESS) {
                return ret;
            }
            clock_t afterCcuBilateral = clock();
            double ccuParallel = double(afterCcuParallel - ccuStart) / CLOCKS_PER_SEC;
            double ccuBilateral = double(afterCcuBilateral - afterCcuParallel) / CLOCKS_PER_SEC;
            std::cout<<"CCU revamp cost time: ccuParallel= "<<ccuParallel<<", ccuBilateral= "<<ccuBilateral<<std::endl;
        }
#endif
        clock_t revampStart = clock();
        // 单边转双边
        ret = graphRevamp.Revamp(&dummyStartCopy);
        if (ret != HcclResult::HCCL_SUCCESS) {
            DataDumper::Global()->DumpGraph(&dummyStartCopy, GraphType::ORIGINAL_GRAPH);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::REVAMP_GRAPH_FAILED);
            return ret;
        }
        DataDumper::Global()->DumpGraph(&dummyStartCopy, GraphType::BILATERALSEMANTIC_GRAPH);
        clock_t afterBilateral = clock();
        double bilateral = double(afterBilateral - revampStart) / CLOCKS_PER_SEC;
        std::cout<<"Cost time: bilateral= "<<bilateral<<std::endl;

        CheckRankMem checkRankmem(&dummyStartCopy);
        ret = checkRankmem.Execute();
        if (ret != HcclResult::HCCL_SUCCESS) {
            DataDumper::Global()->DumpGraph(&dummyStart, GraphType::ORIGINAL_GRAPH);
            return ret;
        }
        clock_t afterCheckMem = clock();
        double checkMem = double(afterCheckMem - afterBilateral) / CLOCKS_PER_SEC;
        std::cout<<"Cost time: checkMem= "<<checkMem<<std::endl;
    }
    // 语义检查
    clock_t semCheckStart = clock();
    TaskCheckOpSemantics opSemanticsChcker(&dummyStart, checkerOpParam, rankNum);
    ret = opSemanticsChcker.Execute();
    DataDumper::Global()->DumpGraph(&dummyStart, GraphType::ORIGINAL_GRAPH);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }
    clock_t afterSemCheck = clock();
    double checkSem = double(afterSemCheck - semCheckStart) / CLOCKS_PER_SEC;
    std::cout<<"Cost time: checkSem= "<<checkSem<<std::endl;
    if (enableGraphPrint_) {
        PrintGraphRevamp(&dummyStart);
    }

    return HcclResult::HCCL_SUCCESS;
}

void Checker::setCheckerLogWarn() {
    dlog_setlevel(INVLID_MOUDLE_ID, DLOG_WARN, 1);
}

} // namespace checker