/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 算法分析器CCU task定义
 * Create: 2025-04-11
 */
#include "task_ccu.h"
#include <map>
#include "ccu_microcode.h"
#include "log.h"
#include "ccu_task_param.h"

namespace Hccl {
Hccl::Instruction* g_ccuIns = nullptr;
std::map<const Hccl::Instruction*, std::vector<Hccl::CcuRep::CcuInstrInfo>> g_ccuIns2MicroCode;
std::map<u32, std::map<string, std::vector<Hccl::CcuRep::CcuInstrInfo>>> g_ccuSig2MicroCode;

void DumpInstruction(const Hccl::CcuRep::CcuInstrInfo &ccuMicroSeq)
{
    HCCL_ERROR("CcuInstrInfo: startInstrId = %u, instrCount = %u, missionStartInstrId = %u, missionInstrCount = %u",
            ccuMicroSeq.startInstrId, ccuMicroSeq.instrCount, ccuMicroSeq.missionStartInstrId, ccuMicroSeq.missionInstrCount);
    for (uint16_t index = 0; index < ccuMicroSeq.instrVec.size(); index++) {
        HCCL_ERROR("%d: %s", ccuMicroSeq.startInstrId + index, ParseInstr(ccuMicroSeq.instrVec.data() + index).c_str());
    }
}

}

namespace checker {

TaskStubCcuGraph::TaskStubCcuGraph(const Hccl::Instruction &ins, RankId rank)
    : TaskStub(TaskTypeStub::CCU_GRAPH), rankId(rank), des(ins.Describe())
{
    Hccl::CcuInstruction* ccuIns = dynamic_cast<Hccl::CcuInstruction*>(const_cast<Hccl::Instruction*>(&ins));
    ccuIns->Translate(ccuParams);

    if (!Hccl::g_ccuSig2MicroCode[rank].count(ccuIns->GetCtxSignature().GetData())){
        Hccl::g_ccuSig2MicroCode[rank][ccuIns->GetCtxSignature().GetData()] = Hccl::g_ccuIns2MicroCode[ccuIns];
        instrInfo = Hccl::g_ccuSig2MicroCode[rank][ccuIns->GetCtxSignature().GetData()];
    } else {
        instrInfo = Hccl::g_ccuSig2MicroCode[rank][ccuIns->GetCtxSignature().GetData()];
    }

    if (instrInfo.size() == 0) {
        std::cout << "the instrInfo size is zero" << std::endl;
    }

    for (int i = 0; i < instrInfo.size(); i++) {
        microCodePosInQue.push_back(instrInfo[i].missionStartInstrId - instrInfo[i].startInstrId);
        startInstrIdInQue.push_back(instrInfo[i].startInstrId);
        instrQueStatus.push_back(false);
        tailNodes.push_back(nullptr);
        ccuParamIndexs.push_back(0);
    }
    ccuHeadTaskNode = new TaskNode(nullptr, -1, 0, 0);
    toDeleteTaskNode_.push_back(ccuHeadTaskNode);
}

// [注意!!!]拷贝构造函数: 用于后面ccu子图的内存冲突改造(仅保留了内存冲突改造所需要的成员变量, 如果用作它途,可能成员变量须补充)
TaskStubCcuGraph::TaskStubCcuGraph(const TaskStubCcuGraph *ccuTask)
    : TaskStub(TaskTypeStub::CCU_GRAPH), rankId(ccuTask->rankId), des(ccuTask->des), queueNum_(ccuTask->queueNum_)
{
    ccuHeadTaskNode = new TaskNode(nullptr, -1, 0, 0);
    toDeleteTaskNode_.push_back(ccuHeadTaskNode);
}

TaskStubCcuGraph::~TaskStubCcuGraph()
{
    for (auto& task : toDeleteTask_) {
        delete task;
    }
    for (auto& node : toDeleteTaskNode_) {
        delete node;
    }
}

void TaskStubCcuGraph::GetSqe(uint32_t queId, uint16_t sqeArgsId, uint64_t& argVal)
{
    argVal = ccuParams[queId][ccuParamIndexs[queId]].args[sqeArgsId];
    if (sqeArgsId == (Hccl::CCU_SQE_ARGS_LEN - 1)) {
        ccuParamIndexs[queId] += 1;
    }

    return;
}

void TaskStubCcuGraph::GetDieId(uint32_t queId, uint32_t& dieId)
{
    dieId = ccuParams[queId][0].dieId;
    return;
}

RankId TaskStubCcuGraph::GetRankId()
{
    return rankId;
}

std::string TaskStubCcuGraph::Describe() const
{
    for (auto& ccuMicroSeq : instrInfo) {
        Hccl::DumpInstruction(ccuMicroSeq);
    }
    return StringFormat("[CCU GRAPH]: %s", des.c_str());
}

TaskStubSubGraphEnd::TaskStubSubGraphEnd(TaskNodePtr node)
    : TaskStub(TaskTypeStub::SUB_GRAPH_END), subGraphNode(node)
{ }

std::string TaskStubSubGraphEnd::Describe() const
{
    return StringFormat("end node of sub graph: %s", subGraphNode->task->Describe().c_str());
}

} // namespace hccl