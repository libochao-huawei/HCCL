/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker.h"
#include "hccl.h"
#include "acl/acl_rt.h"
#include <thread>
#include <future>
#include "sim_communicator.h"
#include "sim_world.h"
#include "check_utils.h"
#include "hccl_sim_pub.h"
#include "sim_channel_exchange_handler.h"
#include "singletask_check.h"
#include "sim_task_queue.h"
#include "sim_task.h"
#include "task_def.h"
#include "checker_def.h"
#include "task_graph_generator.h"
#include "check_rank_mem.h"
#include "task_check_op_semantics.h"
#include "task_graph_revamp.h"
#include "sim_run_params.h"

using namespace std;

namespace HcclSim {
uint32_t CalRankSize(const TopoMeta &topoMeta)
{
    uint32_t rankNum = 0;
    for (const auto& superPod : topoMeta) {
        for (const auto& server : superPod) {
            for (const auto &phyId : server) {
                rankNum++;
            }
        }
    }

    return rankNum;
}

Checker::~Checker()
{
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

void Checker::CloseRankMemCheck()
{
    closeRankMemCheck_ = true;
}

bool Checker::Check(const TopoMeta& topoMeta, const CheckerOpParam& opParam)
{
    rankSize_ = CalRankSize(topoMeta);
    SimWorld::Global()->InitSimWorld(topoMeta, opParam.devType);
    SimRunParams::GetInstance().Init(opParam);

    vector<future<HcclResult>> opResults;
    for (uint32_t rankId = 0; rankId < rankSize_; ++rankId) {
        future<HcclResult> curRes = std::async(std::launch::async, &Checker::HcclOpsFlow, this, topoMeta, opParam, rankId);
        opResults.push_back(std::move(curRes));
    }
    // 检查算子接口的结果
    for (uint32_t rankId = 0; rankId < opResults.size(); ++rankId) {
        try {
            HcclResult ret = opResults[rankId].get();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[Checker::Check] Op failed at rank[%u], ret[%d]", rankId, ret);
                return false;
            }
        } catch (HcclException& e) {
            HCCL_ERROR("[Checker::Check] Op failed at rank[%u], throw exception[%s]", rankId, e.what());
            return false;
        } catch (...) {
            HCCL_ERROR("[Checker::Check] Op failed at rank[%u], unkown error occurs", rankId);
            return false;
        }
    }

    // 结果校验
    AllRankTaskQueues allRankTaskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    PrintTask();
    HcclResult res = GenAndCheckGraph(opParam, allRankTaskQueues);

    HcclSim::SimChannelExchangeHandler::GetInstance().Clear();
    SimWorld::Global()->Reset();

    return res == HCCL_SUCCESS;
}

void Checker::PrintTask() {
    AllRankTaskQueues &allRankTaskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    u32 rankIdx = 0;
    for (auto& podIter : allRankTaskQueues) {
        auto& serMap = podIter.second;
        for (auto& serIter : serMap) {
            auto& phyMap = serIter.second;
            for (auto& phyIter : phyMap) {
                printf("=======================================================\n");
                printf("rankId is : %d \n", rankIdx);
                u32 threadIdx = 0;
                const SingleTaskQueue& taskQueue = phyIter.second;
                for (auto& thread : taskQueue) {
                    printf("threadIdx : %d, taskNum : %d \n", threadIdx, thread.size());
                    printf("-------------------------------------------------------\n");
                    for (auto& task : thread) {
                        string tempStr = task->Describe();
                        printf("rankIdx:%d, threadIdx:%d, %s\n", rankIdx, threadIdx, tempStr.c_str());
                    }
                    threadIdx++;
                }
                rankIdx++;
            }
        }
    }
}

HcclResult Checker::HcclOpsFlow(const TopoMeta& topoMeta, const CheckerOpParam& opParam, uint32_t rankId)
{
    // 1.SetDevice
    aclrtSetDevice(rankId);

    // 2.创建流
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // 3.初始化通信域
    HcclComm comm = nullptr;
    CHK_RET(Sim_HcclCommInitClusterInfo(topoMeta, rankId, &comm));

    // 4.算子下发
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(opParam, rankSize_, inputSize, outputSize, rankId);
    SimNpu& simNpu = SimWorld::Global()->GetSimNpuByRankId(rankId);
    void* sendBuf = reinterpret_cast<void*>(simNpu.AllocMemory(BufferType::INPUT, inputSize));
    void* recvBuf = reinterpret_cast<void*>(simNpu.AllocMemory(BufferType::OUTPUT, outputSize));
    uint32_t root = opParam.root;
    uint64_t recvCount = opParam.DataDes.count;
    HcclDataType dataType = g_CheckerDataType2HcclDataType[opParam.DataDes.dataType];
    if (opParam.opType == CheckerOpType::SCATTER) {
        CHK_RET(HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream));
    } else {
        HCCL_ERROR("[Checker::HcclOpsFlow] not support opType[%s]", opParam.opType.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    // 5.销毁通信域
    CHK_RET(HcclCommDestroy(comm));

    return HCCL_SUCCESS;
}

HcclResult Checker::GenAndCheckGraph(CheckerOpParam opParam, AllRankTaskQueues& allRankTaskQueues) {
    // 1. 检查从流
    HCCL_INFO("1. 检查从流");
    SingleTaskCheck taskChecker;
    // CHK_RET 和 SIM_CHK_RET 的差别？
    CHK_RET(taskChecker.CheckSlaveTaskQueue(allRankTaskQueues));

    // 2. 成图
    HCCL_INFO("2. 成图");
    TaskNode dummyStart = TaskNode(nullptr, -1, 0, 0);
    TaskNode dummyStartCopy = TaskNode(nullptr, -1, 0, 0);
    TaskGraphGenerator graphGenerator;
    CHK_RET(graphGenerator.GenGraph(allRankTaskQueues, &dummyStart));

    // 3. Task内存校验
    // 是否可以复用 taskChecker
    HCCL_INFO("3. Task内存校验");
    CHK_RET(taskChecker.CheckTaskMem(&dummyStart));

    // 4. 图复制
    HCCL_INFO("4. 图复制");
    CopyTaskGraph(&dummyStart, &dummyStartCopy);

    if (!closeRankMemCheck_) {
        // 5. 图改造
        HCCL_INFO("5. 图改造");
        GraphRevampBilateralSemantics graphRevamp;
        CHK_RET(graphRevamp.Revamp(&dummyStartCopy, allRankTaskQueues));

        // 6. Rank内存校验
        HCCL_INFO("6. Rank内存校验");
        CheckRankMem checkRankmem(&dummyStartCopy);
        CHK_RET(checkRankmem.Execute());
    }

    // 7. 语义校验
    // 需要把 CheckerOpParam 转成 checkerOpParam
    HCCL_INFO("7. 语义校验");
    u32 rankNum = 0;
    for (const auto& podIter : allRankTaskQueues) {
        const auto& serMap = podIter.second;
        for (const auto& serIter : serMap) {
            const auto& phyMap = serIter.second;
            rankNum += phyMap.size();
        }
    }

    TaskCheckOpSemantics opSemanticsChcker(&dummyStart, opParam, rankNum);
    CHK_RET(opSemanticsChcker.Execute());

    // 成图及校验成功
    return HCCL_SUCCESS;
}

void Checker::CopyTaskGraph(TaskNodePtr originNode, TaskNodePtr copyNode)
{
    // 该函数无修改
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

        TaskNodePtr newNodePtr = new TaskNode(curNode->task, curNode->rankIdx, curNode->queIdx, curNode->pos);
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
}