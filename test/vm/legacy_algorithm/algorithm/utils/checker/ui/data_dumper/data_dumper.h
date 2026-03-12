#ifndef HCCLV1_DATA_DUMPER_H
#define HCCLV1_DATA_DUMPER_H

#include <map>
#include <set>
#include <string>

#include "analysis_result.pb.h"
#include "task_graph_generator.h"
#include "task_check_op_semantics.h"
#include "check_rank_mem.h"
namespace checker {

enum class GraphType {
    ORIGINAL_GRAPH,
    BILATERALSEMANTIC_GRAPH,
    PARALLELED_GRAPH
};

class DataDumper {
public:
    static DataDumper* Global();
    void SetResultStatus(gui::ResultStatus status);

    // TODO: 能否将成环的信息给捕获出来
    void DumpGraph(TaskNodePtr dummyStart, GraphType graphType);

    void SetRankSize(u32 rankSize)
    {
        analysisResult_.set_ranksize(rankSize);
        return;
    }

    // 待相关特性开发完成后进行补充
    void DumpMemConflictInfo(TaskNodePtr nodeA, TaskNodePtr nodeB, SliceMemoryStatus& statusA, SliceMemoryStatus& statusB);
    void InitSemanticState();
    void DumpSemanticState(RankId rankId, u32 localStep, u32 globalStep, bool change,
                           RankMemorySemantics& memSemantics);
    void MarkInvalidSemantic(RankId rankId, BufferType type, const BufferSemantic& semantic);

    // dump 错误信息
    void AddErrorString(std::string& msg);

    void SerializeToFile();
    void ClearData();

    void Enable();
    void Close();
    void SetFileName(std::string name)
    {
        fileName_ = name;
        return;
    }

    void AddMissingSemantic(RankId rankId, BufferType type, u64 startAddr);

private:
    bool NodeIsReady(TaskNodePtr node, std::set<TaskNodePtr>& visited);
    void GenNodeId(TaskNodePtr dummyStart, GraphType graphType);
    void GenNodeMessage(std::map<u32, TaskNodePtr> &nodeId2nodePtr, std::map<TaskNodePtr, u32> &nodePtr2nodeId,
                        std::map<u32, std::set<u32>> &rank2nodes, gui::WholeGraph* wholeGraph);
    void FillInGuiBufferSemantic(gui::BufferSemantic* guiBufferSemantic, const BufferSemantic& singleBufferSemantic);
    void DumpBufferSemantic(gui::MemBufferSemantic* curState, RankMemorySemantics& memSemantics);

    // 原图
    std::map<u32, TaskNodePtr> nodeId2nodePtr_;
    std::map<TaskNodePtr, u32> nodePtr2nodeId_;
    // rank -> set of nodeId
    std::map<u32, std::set<u32>> rank2nodes_;

    // 修改后的图
    std::map<u32, TaskNodePtr> bilateralNodeId2nodePtr_;
    std::map<TaskNodePtr, u32> bilateralNodePtr2nodeId_;
    // rank -> set of nodeId
    std::map<u32, std::set<u32>> bilateralRank2nodes_;

    // 默认不使能dump，否则会影响性能
    bool enabled_ = false;
    gui::AnalysisResult analysisResult_;
    std::string fileName_ = "analysis_result";
};

}

#endif