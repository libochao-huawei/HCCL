/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 算法分析器CCU task定义
 * Create: 2025-04-11
 */
#ifndef ADAPTER_V2_TASK_H
#define ADAPTER_V2_TASK_H

#include "task_stub.h"
#include <vector>
#include "ccu_instr_info.h"
#include "ccu_task_param.h"
#include "ccu_ins.h"
#include "task_def.h"

namespace Hccl {
extern Hccl::Instruction* g_ccuIns;
extern std::map<const Hccl::Instruction*, std::vector<Hccl::CcuRep::CcuInstrInfo>> g_ccuIns2MicroCode;

}

namespace checker {
constexpr uint32_t MAX_LOOPGROUP_NUM = 1024;

struct LoopInfo {
    LoopInfo(TaskNodePtr start, TaskNodePtr end) : loopStart(start), loopEnd(end) {}
    TaskNodePtr loopStart;
    TaskNodePtr loopEnd;
};

// 双边语义改造节点信息
struct BilateralWaitInfo {
    // peerRank对应的wait节点
    std::vector<TaskNodePtr> waitNodes;
};
struct BilateralPostInfo {
    // peerRank对应的异步节点
    std::vector<TaskNodePtr> asyncNodes;
};

struct BilateralNode {
    BilateralNode(TaskNodePtr asyncNode, TaskNodePtr wait, TaskNodePtr post) : asyncNode(asyncNode), backwardWait(wait), forwardPost(post) {}
    TaskNodePtr backwardWait{nullptr};
    TaskNodePtr forwardPost{nullptr};
    TaskNodePtr asyncNode;
};

// 定义外层CCU的Task
class TaskStubCcuGraph : public TaskStub {
public:
    TaskStubCcuGraph(const Hccl::Instruction &ins, RankId rank);
    // 本构造函数只给UT测试使用
    TaskStubCcuGraph(RankId rank) : TaskStub(TaskTypeStub::CCU_GRAPH), rankId(rank) {
        ccuHeadTaskNode = new TaskNode(nullptr, -1, 0, 0);
        toDeleteTaskNode_.push_back(ccuHeadTaskNode);
    }
    TaskStubCcuGraph(const TaskStubCcuGraph *ccuTask);
    ~TaskStubCcuGraph();
    std::string Describe() const override;

    void GetSqe(uint32_t queId, uint16_t sqeArgsId, uint64_t& argVal);
    void GetDieId(uint32_t queId, uint32_t& dieId);
    RankId GetRankId();

    RankId rankId;
    std::string des;
    std::vector<Hccl::CcuRep::CcuInstrInfo> instrInfo;
    std::vector<std::vector<Hccl::CcuTaskParam>> ccuParams;
    std::vector<u32> ccuParamIndexs;     // 记录load加载到哪一组参数
    std::vector<u32> microCodePosInQue;  // 每条que当前处理到的微码位置
    std::vector<u32> startInstrIdInQue;  // 每条que的起始ID
    std::vector<bool> instrQueStatus;    // 一条微码队列成图完成后，其标识位设置为true
    std::vector<TaskNode*> tailNodes;    // 微码队列成图时，对应的尾部node
    TaskNode* ccuHeadTaskNode = nullptr;
    bool isGenGraphed = false; // 可能不需要了

    // 记录queue个数（用于后续分配虚拟流ID）
    uint32_t queueNum_{0};

    // 记录所有loop信息，后续并行化改造使用
    uint32_t loopGroupIdx = 0;
    std::array<uint32_t, MAX_LOOPGROUP_NUM> loopIdx{};
    std::vector<std::vector<LoopInfo>> loopGroupInfo_;
    // 记录所有本端或远端异步节点，后续并行化改造使用
    std::vector<TaskNodePtr> parallelNodes_;
    // 记录ccu子图中配对的local post <--> local wait (ccu子图存在多对一场景，无法用topic id直接判断)
    std::map<TaskNodePtr, TaskNodePtr> localPostWaitPairs_;
    // 记录所有远端异步节点信息，后续双边语义改造使用
    std::vector<BilateralWaitInfo> waitInfoTmp_;
    std::vector<BilateralPostInfo> postInfoTmp_;
    std::vector<std::map<TaskNodePtr, TaskNodePtr>> bilateralPart1_;
    std::vector<std::map<TaskNodePtr, TaskNodePtr>> bilateralPart2_;
    std::vector<std::vector<BilateralNode>> bilateralNodes_;

    // 要释放的资源，在释放这个节点的时候需要进行调用
    std::vector<TaskStub*> toDeleteTask_;
    std::vector<TaskNodePtr> toDeleteTaskNode_;
};

class TaskStubSubGraphEnd : public TaskStub {
public:
    TaskStubSubGraphEnd(TaskNodePtr node);
    std::string Describe() const override;
    TaskNodePtr subGraphNode = nullptr;
};

} // namespace hccl
#endif