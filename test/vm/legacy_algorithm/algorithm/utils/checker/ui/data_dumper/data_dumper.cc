#include "data_dumper.h"

#include <queue>
#include <fstream>
#include "google/protobuf/text_format.h"

#include "task_stub.h"
#include "mem_layout.h"

namespace checker {

DataDumper* DataDumper::Global()
{
    static DataDumper* dataDumper = new DataDumper;
    return dataDumper;
}

void DataDumper::SetResultStatus(gui::ResultStatus status)
{
    analysisResult_.set_resultstatus(status);
    return;
}

bool DataDumper::NodeIsReady(TaskNodePtr node, std::set<TaskNodePtr>& visited)
{
    for (TaskNodePtr& parent : node->parents) {
        if (visited.count(parent) == 0) {
            return false;
        }
    }
    return true;
}

// 为每个node生成nodeIdx
void DataDumper::GenNodeId(TaskNodePtr dummyStart, GraphType graphType)
{
    u32 nodeId = 0;
    std::set<TaskNodePtr> visited;

    if (graphType == GraphType::ORIGINAL_GRAPH) {
        nodeId2nodePtr_[nodeId] = dummyStart;
        nodePtr2nodeId_[dummyStart] = nodeId;
        nodeId++;
    } else if (graphType == GraphType::BILATERALSEMANTIC_GRAPH) {
        bilateralNodeId2nodePtr_[nodeId] = dummyStart;
        bilateralNodePtr2nodeId_[dummyStart] = nodeId;
        nodeId++;
    }

    std::queue<TaskNodePtr> walkQueue;
    visited.insert(dummyStart);
    for (TaskNodePtr& child : dummyStart->children) {
        RankId rankId = child->rankIdx;
        // dummy节点需要放到每一个rank中
        if (graphType == GraphType::ORIGINAL_GRAPH) {
            rank2nodes_[rankId].insert(0);
        } else if (graphType == GraphType::BILATERALSEMANTIC_GRAPH) {
            bilateralRank2nodes_[rankId].insert(0);
        }
        walkQueue.push(child);
    }

    while (!walkQueue.empty()) {
        TaskNodePtr node = walkQueue.front();
        walkQueue.pop();

        // 因为有的节点会被多次添加到queue中，这边需要忽略掉
        if (visited.count(node) != 0) {
            continue;
        }

        if (!NodeIsReady(node, visited)) {
            walkQueue.push(node);
            continue;
        }

        RankId rankId = node->rankIdx;
        if (graphType == GraphType::ORIGINAL_GRAPH) {
            rank2nodes_[rankId].insert(nodeId);
            nodeId2nodePtr_[nodeId] = node;
            nodePtr2nodeId_[node] = nodeId;
        } else if (graphType == GraphType::BILATERALSEMANTIC_GRAPH) {
            bilateralRank2nodes_[rankId].insert(nodeId);
            bilateralNodeId2nodePtr_[nodeId] = node;
            bilateralNodePtr2nodeId_[node] = nodeId;
        }

        nodeId++;
        visited.insert(node);

        for (auto& child : node->children) {
            if (visited.count(child) != 0) {
                continue;
            }
            walkQueue.push(child);
        }
    }

    return;
}

std::map<TaskTypeStub, gui::NodeType> TASK_TYPE_2_NODE_TYPE = {
    {TaskTypeStub::LOCAL_COPY, gui::NodeType::LOCAL_COPY},
    {TaskTypeStub::LOCAL_REDUCE, gui::NodeType::LOCAL_REDUCE},
    {TaskTypeStub::LOCAL_POST_TO, gui::NodeType::LOCAL_POST},
    {TaskTypeStub::LOCAL_WAIT_FROM, gui::NodeType::LOCAL_WAIT},
    {TaskTypeStub::POST, gui::NodeType::POST},
    {TaskTypeStub::WAIT, gui::NodeType::WAIT},
    {TaskTypeStub::READ, gui::NodeType::READ},
    {TaskTypeStub::READ_REDUCE, gui::NodeType::READ_REDUCE},
    {TaskTypeStub::WRITE, gui::NodeType::WRITE},
    {TaskTypeStub::WRITE_REDUCE, gui::NodeType::WRITE_REDUCE},
    {TaskTypeStub::BEING_READ, gui::NodeType::BEING_READ},
    {TaskTypeStub::BEING_READ_REDUCE, gui::NodeType::BEING_READ_REDUCE},
    {TaskTypeStub::BEING_WRITTEN, gui::NodeType::BEING_WRITTEN},
    {TaskTypeStub::BEING_WRITTEN_REDUCE, gui::NodeType::BEING_WRITTEN_REDUCE},
    {TaskTypeStub::LOCAL_POST_TO_SHADOW, gui::NodeType::LOCAL_POST_SHADOW},
    {TaskTypeStub::LOCAL_WAIT_FROM_SHADOW, gui::NodeType::LOCAL_WAIT_FROM_SHADOW},
};

std::map<MemoryStatus, gui::MemOp> MEM_STATUS_2_MEM_OP = {
    {MemoryStatus::READ, gui::MemOp::MEM_READ},
    {MemoryStatus::WRITE, gui::MemOp::MEM_WRITE},
};

void DataDumper::GenNodeMessage(std::map<u32, TaskNodePtr> &nodeId2nodePtr, std::map<TaskNodePtr, u32> &nodePtr2nodeId,
                                std::map<u32, std::set<u32>> &rank2nodes, gui::WholeGraph* wholeGraph)
{
    for (RankId rank = 0; rank < rank2nodes.size(); rank++) {
        gui::RankGraph* rankGraph = wholeGraph->add_rankgraphs();
        for (auto& nodeId : rank2nodes[rank]) {
            gui::Node* node = rankGraph->add_nodes();
            node->set_nodeid(nodeId);
            TaskNodePtr taskNode = nodeId2nodePtr[nodeId];
            if (taskNode->task) {
                node->set_nodetype(TASK_TYPE_2_NODE_TYPE[taskNode->task->GetType()]);
            } else {
                node->set_nodetype(gui::NodeType::RESERVED);
            }

            node->set_queueid(taskNode->queIdx);
            node->set_position(taskNode->pos);

            for (auto& child : taskNode->children) {
                node->add_children(nodePtr2nodeId[child]);
            }

            for (auto& parent : taskNode->parents) {
                node->add_parents(nodePtr2nodeId[parent]);
            }

            node->set_globalstep(taskNode->globalStep);
            gui::LocalStep* localStep = node->mutable_localstep();
            localStep->set_rank(rank);
            localStep->set_localstep(taskNode->localStep);

            // describe 信息占用的空间很大，可能要考虑压缩一下
            if (taskNode->task) {
                node->set_nodedescribe(taskNode->task->Describe());
            } else {
                node->set_nodedescribe(string());
            }
            node->set_rankid(taskNode->rankIdx);
            // 记录不匹配信息
            node->set_unmatch(taskNode->unmatch);

            // 记录生成语义信息是否有问题
            node->set_gensemanticerror(taskNode->genSemanticError);
        }
    }
}

void DataDumper::DumpGraph(TaskNodePtr dummyStart, GraphType graphType)
{
    // 如果未使能，跳过dump处理
    if (!enabled_) {
        return;
    }
    GenNodeId(dummyStart, graphType);

    if (graphType == GraphType::ORIGINAL_GRAPH) {
        gui::WholeGraph* wholeGraph = analysisResult_.mutable_wholegraph();
        GenNodeMessage(nodeId2nodePtr_, nodePtr2nodeId_, rank2nodes_, wholeGraph);
    } else if (graphType == GraphType::BILATERALSEMANTIC_GRAPH) {
        gui::WholeGraph* wholeGraph = analysisResult_.mutable_bilateralgraph();
        GenNodeMessage(bilateralNodeId2nodePtr_, bilateralNodePtr2nodeId_, bilateralRank2nodes_, wholeGraph);
    } else if (graphType == GraphType::PARALLELED_GRAPH) {
        gui::WholeGraph* wholeGraph = analysisResult_.mutable_bilateralgraph();
        GenNodeMessage(bilateralNodeId2nodePtr_, bilateralNodePtr2nodeId_, bilateralRank2nodes_, wholeGraph);
    }

    return;
}

void DataDumper::DumpMemConflictInfo(TaskNodePtr nodeA, TaskNodePtr nodeB, SliceMemoryStatus& statusA, SliceMemoryStatus& statusB)
{
    if (!enabled_) {
        return;
    }
    gui::MemConflict* memConflictInfo = analysisResult_.mutable_memconflict();
    memConflictInfo->set_rankid(nodeA->rankIdx);
    memConflictInfo->set_nodeida(bilateralNodePtr2nodeId_[nodeA]);
    memConflictInfo->set_nodeidb(bilateralNodePtr2nodeId_[nodeB]);

    gui::DataSlice* dataSliceA = memConflictInfo->mutable_dataslicea();
    gui::DataSlice* dataSliceB = memConflictInfo->mutable_datasliceb();

    dataSliceA->set_memop(MEM_STATUS_2_MEM_OP[statusA.status]);
    dataSliceA->set_startaddr(statusA.startAddr);
    dataSliceA->set_endaddr(statusA.startAddr + statusA.size);
    dataSliceA->set_size(statusA.size);

    dataSliceB->set_memop(MEM_STATUS_2_MEM_OP[statusB.status]);
    dataSliceB->set_startaddr(statusB.startAddr);
    dataSliceB->set_endaddr(statusB.startAddr + statusB.size);
    dataSliceB->set_size(statusB.size);
    return;
}

void DataDumper::InitSemanticState()
{
    // 如果未使能，跳过处理
    if (!enabled_) {
        return;
    }
    u32 rankSize = analysisResult_.ranksize();
    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        gui::RankBufferSemanticStates* rankStates = analysisResult_.add_rankstates();
        rankStates->set_rankid(rankId);

        MemBlock inputBlock = MemLayout::Global()->GetMemBlock(BufferType::INPUT, rankId);
        rankStates->set_inputsize(inputBlock.size);

        MemBlock outputBlock = MemLayout::Global()->GetMemBlock(BufferType::OUTPUT, rankId);
        rankStates->set_outputsize(outputBlock.size);

        MemBlock inputCCLBlock = MemLayout::Global()->GetMemBlock(BufferType::INPUT_CCL, rankId);
        rankStates->set_inputcclsize(inputCCLBlock.size);

        MemBlock outputCCLBlock = MemLayout::Global()->GetMemBlock(BufferType::OUTPUT_CCL, rankId);
        rankStates->set_outputcclsize(outputCCLBlock.size);

        MemBlock scratchBlock = MemLayout::Global()->GetMemBlock(BufferType::SCRATCH, rankId);
        rankStates->set_scratchsize(scratchBlock.size);
    }
    return;
}

std::map<CheckerReduceOp, gui::ReduceType> REDUCE_TYPE_TABLE = {
    {CheckerReduceOp::REDUCE_SUM, gui::ReduceType::REDUCE_SUM},
    {CheckerReduceOp::REDUCE_PROD, gui::ReduceType::REDUCE_PROD},
    {CheckerReduceOp::REDUCE_MAX, gui::ReduceType::REDUCE_MAX},
    {CheckerReduceOp::REDUCE_MIN, gui::ReduceType::REDUCE_MIN},
    {CheckerReduceOp::REDUCE_RESERVED, gui::ReduceType::REDUCE_RESERVED},
};

std::map<BufferType, gui::BufferType> BUFFER_TYPE_TABLE = {
    {BufferType::INPUT, gui::BufferType::INPUT},
    {BufferType::OUTPUT, gui::BufferType::OUTPUT},
    {BufferType::INPUT_CCL, gui::BufferType::INPUT_CCL},
    {BufferType::OUTPUT_CCL, gui::BufferType::OUTPUT_CCL},
    {BufferType::SCRATCH, gui::BufferType::SCRATCH},
    {BufferType::RESERVED, gui::BufferType::BUFFER_RESERVED},
};

void DataDumper::FillInGuiBufferSemantic(gui::BufferSemantic* guiBufferSemantic, const BufferSemantic& singleBufferSemantic)
{
    guiBufferSemantic->set_startaddr(singleBufferSemantic.startAddr);
    guiBufferSemantic->set_size(singleBufferSemantic.size);
    guiBufferSemantic->set_isreduce(singleBufferSemantic.isReduce);
    guiBufferSemantic->set_reducetype(REDUCE_TYPE_TABLE[singleBufferSemantic.reduceType]);
    for (auto& sucBuf : singleBufferSemantic.srcBufs) {
        gui::SrcBufDes* guiSrcBuf = guiBufferSemantic->add_srcbufs();
        guiSrcBuf->set_rankid(sucBuf.rankId);
        guiSrcBuf->set_buftype(BUFFER_TYPE_TABLE[sucBuf.bufType]);
        guiSrcBuf->set_srcaddr(sucBuf.srcAddr);
    }
    for (auto& globalStep : singleBufferSemantic.affectedGlobalSteps) {
        guiBufferSemantic->add_affectedglobalsteps(globalStep);
    }

    return;
}

void DataDumper::DumpBufferSemantic(gui::MemBufferSemantic* curState, RankMemorySemantics& memSemantics)
{
    if (memSemantics.count(BufferType::INPUT) != 0) {
        std::set<BufferSemantic>& inputSemantics = memSemantics[BufferType::INPUT];
        for (auto& singleBufferSemantic : inputSemantics) {
            gui::BufferSemantic* guiBufferSemantic = curState->add_inputbuffersemantics();
            FillInGuiBufferSemantic(guiBufferSemantic, singleBufferSemantic);
        }
    }

    if (memSemantics.count(BufferType::OUTPUT) != 0) {
        std::set<BufferSemantic>& outputSemantics = memSemantics[BufferType::OUTPUT];
        for (auto& singleBufferSemantic : outputSemantics) {
            gui::BufferSemantic* guiBufferSemantic = curState->add_outputbuffersemantics();
            FillInGuiBufferSemantic(guiBufferSemantic, singleBufferSemantic);
        }
    }

    if (memSemantics.count(BufferType::INPUT_CCL) != 0) {
        std::set<BufferSemantic>& inputCCLSemantics = memSemantics[BufferType::INPUT_CCL];
        for (auto& singleBufferSemantic : inputCCLSemantics) {
            gui::BufferSemantic* guiBufferSemantic = curState->add_inputcclbuffersemantics();
            FillInGuiBufferSemantic(guiBufferSemantic, singleBufferSemantic);
        }
    }

    if (memSemantics.count(BufferType::OUTPUT_CCL) != 0) {
        std::set<BufferSemantic>& outputCCLSemantics = memSemantics[BufferType::OUTPUT_CCL];
        for (auto& singleBufferSemantic : outputCCLSemantics) {
            gui::BufferSemantic* guiBufferSemantic = curState->add_outputcclbuffersemantics();
            FillInGuiBufferSemantic(guiBufferSemantic, singleBufferSemantic);
        }
    }

    if (memSemantics.count(BufferType::SCRATCH) != 0) {
        std::set<BufferSemantic>& scratchSemantics = memSemantics[BufferType::SCRATCH];
        for (auto& singleBufferSemantic : scratchSemantics) {
            gui::BufferSemantic* guiBufferSemantic = curState->add_scratchbuffersemantics();
            FillInGuiBufferSemantic(guiBufferSemantic, singleBufferSemantic);
        }
    }

    return;
}

void DataDumper::DumpSemanticState(RankId rankId, u32 localStep, u32 globalStep, bool change,
                                   RankMemorySemantics& memSemantics)
{
    // 如果未使能，跳过处理
    if (!enabled_) {
        return;
    }

    gui::RankBufferSemanticStates* rankStates = analysisResult_.mutable_rankstates(rankId);
    rankStates->add_localstep2globalstep(globalStep);
    gui::MemBufferSemantic* curState = rankStates->add_memstates();

    // 第一个语义块一定要填充的
    if (localStep == 1) {
        curState->set_statenochange(false);
        curState->set_localstep(localStep);

        DumpBufferSemantic(curState, memSemantics);
        return;
    }

    if (!change) { // 表示当前节点的语义状态并没有发生改变
        curState->set_statenochange(true);
        u32 stateSize = rankStates->memstates_size();
        gui::MemBufferSemantic* lastState = rankStates->mutable_memstates(stateSize - 2);
        curState->set_localstep(lastState->localstep());
    } else {
        curState->set_statenochange(false);
        curState->set_localstep(localStep);
        DumpBufferSemantic(curState, memSemantics);
    }

    return;
}

void DataDumper::MarkInvalidSemantic(RankId rankId, BufferType type, const BufferSemantic& semantic)
{
    // 如果未使能，跳过处理
    if (!enabled_) {
        return;
    }

    gui::RankBufferSemanticStates* rankStates = analysisResult_.mutable_rankstates(rankId);
    u32 stateSize = rankStates->memstates_size();
    gui::MemBufferSemantic* lastState = rankStates->mutable_memstates(stateSize - 1);
    u32 lastLocalStep = lastState->localstep();
    lastState = rankStates->mutable_memstates(lastLocalStep - 1);

    if (type == BufferType::INPUT) {
        u32 bufSemanticSize = lastState->inputbuffersemantics_size();
        for (u32 index = 0; index < bufSemanticSize; index++) {
            gui::BufferSemantic* singleBufSemantic = lastState->mutable_inputbuffersemantics(index);
            if (singleBufSemantic->startaddr() == semantic.startAddr) {
                singleBufSemantic->set_invalid(true);
            }
        }
    }

    if (type == BufferType::OUTPUT) {
        u32 bufSemanticSize = lastState->outputbuffersemantics_size();
        for (u32 index = 0; index < bufSemanticSize; index++) {
            gui::BufferSemantic* singleBufSemantic = lastState->mutable_outputbuffersemantics(index);
            if (singleBufSemantic->startaddr() == semantic.startAddr) {
                singleBufSemantic->set_invalid(true);
            }
        }
    }
    return;
}

void DataDumper::AddErrorString(std::string& msg)
{
    // 如果未使能，跳过处理
    if (!enabled_) {
        return;
    }

    analysisResult_.add_errormessage(msg);
    return;
}

void DataDumper::SerializeToFile()
{
    // 如果未使能，跳过处理
    if (!enabled_) {
        return;
    }

    // dump文本格式
    std::string fileName = fileName_ + ".txt";
    std::fstream output(fileName, ios::trunc | ios::out);
    string text;
    google::protobuf::TextFormat::PrintToString(analysisResult_, &text);
    output << text;
    output.close();

    // dump二进制格式
    std::string fileNameBinary = fileName_ + "_binary.txt";
    std::fstream outputBin(fileNameBinary, ios::trunc | ios::out | ios::binary);
    analysisResult_.SerializeToOstream(&outputBin);
    outputBin.close();

    return;
}

void DataDumper::ClearData()
{
    nodeId2nodePtr_.clear();
    nodePtr2nodeId_.clear();
    rank2nodes_.clear();
    analysisResult_.Clear();
    return;
}

void DataDumper::Enable()
{
    enabled_ = true;
}

void DataDumper::Close()
{
    enabled_ = false;
}

void DataDumper::AddMissingSemantic(RankId rankId, BufferType type, u64 startAddr)
{
    // 如果未使能，跳过处理
    if (!enabled_) {
        return;
    }

    gui::MissingSemantic* semantic = analysisResult_.add_missingsemantic();
    semantic->set_rankid(rankId);
    semantic->set_buffertype(BUFFER_TYPE_TABLE[type]);
    semantic->set_startaddr(startAddr);
    return;
}

}
