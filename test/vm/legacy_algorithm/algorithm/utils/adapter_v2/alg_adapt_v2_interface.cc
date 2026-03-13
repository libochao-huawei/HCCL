/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 2.0适配器对外提供的编排接口
 * Author: yinding
 * Create: 2025-06-14
 */

#include "alg_adapt_v2_interface.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <vector>

#include "coll_service_stub.h"
#include "hccl_communicator_stub.h"
#include "coll_alg_params.h"
#include "type_conversion.h"
#include "coll_gen_json.h"
#include "mem_layout.h"
#include "rank_info_recorder.h"
#include "link_type_recorder.h"
#include "check_utils.h"
#include "ccu_ins_preprocessor.h"
#include "rdma_handle_manager_stub.h"
#include "rdma_handle_manager.h"
#include "task_ccu.h"
#include "llt_common.h"
#include "transport_utils.h"
#include "ccu_all_rank_param_recorder.h"
#include "env_config.h"

using namespace checker;
namespace Hccl {

// 在析构函数中对适配2.0的所有全局变量进行清理
TaskQuesGeneratorV2::~TaskQuesGeneratorV2()
{
    std::cout<<"zhf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"<<std::endl;
    g_devId2EidInfo.clear();
    g_devId2Ip2DieIdAndFuncId.clear();
    g_devId2funcId.clear();
    g_devId2PortId2DieId.clear();
    g_devId2PortId2funcId.clear();
    g_devId2PortId2EidInfo.clear();

    g_rdmaHandle2DieIdAndFuncId.clear();
    g_rdmaHandlePtr = reinterpret_cast<RdmaHandle>(0x01);
    // RdmaHandleManager是单例，需要销毁其状态，避免影响下一次用例执行
    RdmaHandleManager::GetInstance().DestroyAll();

    extern std::unordered_map<CcuTransport*, vector<char>> g_ccuTransportSendData;
    g_ccuTransportSendData.clear();

    // 这两个全局变量应该可以用其他的方法来处理吧，后面再探索一下
    g_ccuIns = nullptr;
    g_ccuIns2MicroCode.clear();

    // 清除各个rank的transport和channel信息
    g_allRankTransports.clear();
    g_allRankChannelInfo.clear();
    g_transportsPair.clear();

    // 清理CKE、GSA、Xn、seenPost等状态
    AllRankParamRecorder::Global()->Reset();
}

void GenCollOpParams(u32 myRank, CheckerOpParam &checkerOpParam, CollOpParams &opParams)
{
    opParams.opType = g_CheckerOpType2OpType_aicpu[checkerOpParam.opType];
    opParams.reduceOp = g_CheckerReduceOp2ReduceOp_aicpu[checkerOpParam.reduceType];
    if (opParams.opType == OpType::SEND || opParams.opType == OpType::RECV) {
        if (myRank == checkerOpParam.srcRank) {
            opParams.dstRank = checkerOpParam.dstRank;
        } else {
            opParams.dstRank = checkerOpParam.srcRank;
        }
    }
    opParams.root = checkerOpParam.root;

    opParams.sendBuf = MemLayout::Global()->GetMemBlock(checker::BufferType::INPUT, myRank).startAddr;
    if (opParams.opType == OpType::BROADCAST) {
        opParams.recvBuf = opParams.sendBuf;
    } else {
        opParams.recvBuf = MemLayout::Global()->GetMemBlock(checker::BufferType::OUTPUT, myRank).startAddr;
    }

    if (opParams.opType == OpType::ALLTOALL) {
        opParams.all2AllDataDes.sendType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.sendType];
        opParams.all2AllDataDes.recvType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.recvType];
        opParams.all2AllDataDes.sendCount = checkerOpParam.All2AllDataDes.sendCount;
        opParams.all2AllDataDes.recvCount = checkerOpParam.All2AllDataDes.recvCount;
    } else if (opParams.opType == OpType::ALLTOALLV) {
        opParams.all2AllVDataDes.sendType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.sendType];
        opParams.all2AllVDataDes.recvType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.recvType];
        opParams.all2AllVDataDes.sendCounts = static_cast<void *>(checkerOpParam.All2AllDataDes.sendCounts.data());
        opParams.all2AllVDataDes.recvCounts = static_cast<void *>(checkerOpParam.All2AllDataDes.recvCounts.data());
        opParams.all2AllVDataDes.sdispls = static_cast<void *>(checkerOpParam.All2AllDataDes.sdispls.data());
        opParams.all2AllVDataDes.rdispls = static_cast<void *>(checkerOpParam.All2AllDataDes.rdispls.data());
    } else if (opParams.opType == OpType::ALLTOALLVC) {
        opParams.all2AllVCDataDes.sendType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.sendType];
        opParams.all2AllVCDataDes.recvType = g_CheckerDataType2DataType_aicpu[checkerOpParam.All2AllDataDes.recvType];
        opParams.all2AllVCDataDes.sendCountMatrix = static_cast<void *>(checkerOpParam.All2AllDataDes.sendCountMatrix.data());
    } else if (opParams.opType == OpType::REDUCESCATTERV || opParams.opType == OpType::ALLGATHERV) {
        opParams.vDataDes.counts = static_cast<void *>(checkerOpParam.VDataDes.counts.data());
        opParams.vDataDes.displs = static_cast<void *>(checkerOpParam.VDataDes.displs.data());
        opParams.vDataDes.dataType = g_CheckerDataType2DataType_aicpu[checkerOpParam.VDataDes.dataType];
        opParams.dataType = g_CheckerDataType2DataType_aicpu[checkerOpParam.VDataDes.dataType];
    } else {
        opParams.count = checkerOpParam.DataDes.count;
        opParams.dataType = g_CheckerDataType2DataType_aicpu[checkerOpParam.DataDes.dataType];

        opParams.dataDes.dataCount = checkerOpParam.DataDes.count;
        opParams.dataDes.dataType = g_CheckerDataType2DataType_aicpu[checkerOpParam.DataDes.dataType];
    }

    return;
}

void SetBufferSize(RankId rankId, CheckerOpParam &checkerOpParam, u64 scratchSize)
{
    u32 rankSize = RankInfoRecorder::Global()->GetRankSize();
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(checkerOpParam, rankSize, inputSize, outputSize, rankId);

    MemLayout::Global()->SetBufferLen(checker::BufferType::INPUT, inputSize);
    MemLayout::Global()->SetBufferLen(checker::BufferType::OUTPUT, outputSize);
    MemLayout::Global()->SetBufferLen(checker::BufferType::SCRATCH, scratchSize);

    return;
}

enum TopoNetworkType {TOPO_TYPE_1D, TOPO_TYPE_2D, TOPO_TYPE_DETOUR, TOPO_TYPE_NUM, TOPO_TYPE_INVALID};

TopoNetworkType GetTopoNetworkType()
{
    // 判断CCU 2D流程
    const char* dieNum = std::getenv("HCCL_IODIE_NUM");
    if (dieNum != nullptr) {
        std::string value(dieNum);
        if (value == "2" ) {
            return TopoNetworkType::TOPO_TYPE_2D;
        }
    }

    // CCU 1D流程
    return TopoNetworkType::TOPO_TYPE_1D;
}

HcclResult TaskQuesGeneratorV2::Run(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta, DavidAlgConfig &config)
{
    Hccl::EnvConfig::GetInstance().Parse();

    u32 rankSize = GetRankNumFormTopoMeta(topoMeta);

    LinkTypeRecorder::Global()->SetLinkTypeMap(checkerOpParam.devTypes);
    RankInfoRecorder::Global()->InitRankInfo(topoMeta, checkerOpParam.devtype);

    // 给每个rank每个内存类型分配起始地址
    MemLayout::Global()->Init(checkerOpParam);

    TopoNetworkType topoType = GetTopoNetworkType();
    std::string topoPath;
    InitGenTopoJson(topoPath, checkerOpParam, topoType == TopoNetworkType::TOPO_TYPE_1D);
    std::string rankTableString;
    InitGenRankTableJson(topoMeta, checkerOpParam, rankTableString);

    // 给每个rank生成通信域
    std::vector<std::shared_ptr<HcclCommunicator>> communicators;

    checker::HcclUs startut = TIME_NOW();
    for (u32 myRank = 0; myRank < rankSize; myRank++) {
        RankInfoRecorder::Global()->SetRankId(myRank);

        CommParams commParams;
        commParams.myRank = myRank;
        commParams.rankSize = rankSize;
        commParams.rankInParentComm = myRank;
        commParams.devType = g_CheckerDevType2HcclDevType_aicpu[checkerOpParam.devtype];
        std::shared_ptr<HcclCommunicator> curCommunicator = make_shared<HcclCommunicator>(commParams);
        CHK_RET(curCommunicator->Init(rankTableString, topoPath));
        communicators.push_back(curCommunicator);
    }
    HCCL_INFO("communicator init take time [%lld]us.", DURATION_US(TIME_NOW() - startut));

    string algName = checkerOpParam.algName;
    for (u32 myRank = 0; myRank < rankSize; myRank++) {
        RankInfoRecorder::Global()->SetRankId(myRank);

        // 刷新每块内存的地址大小
        u64 tmpMemSize = EnvConfig::GetInstance().GetAlgoConfig().GetBuffSize();
        SetBufferSize(myRank, checkerOpParam, tmpMemSize);

        CollOpParams opParams;
        GenCollOpParams(myRank, checkerOpParam, opParams);
        CHK_RET(communicators[myRank]->LoadOpbasedCollOp(opParams, algName));
    }

    // 匹配transport
    for (u32 localRank = 0; localRank < rankSize; localRank++) {
        for (auto& iter : g_allRankTransports[localRank]) {
            u32 remoteRank = iter.first;
            for (auto& localTransport : g_allRankTransports[localRank][remoteRank]) {
                if (g_transportsPair.count(localTransport) != 0) {
                    continue;
                }

                IpAddress localAddr = localTransport->GetLocAddr();
                IpAddress rmtAddr = localTransport->GetRmtAddr();

                bool found = false;
                for (auto& remoteTransport : g_allRankTransports[remoteRank][localRank]) {
                    if (localAddr == remoteTransport->GetRmtAddr() && rmtAddr == remoteTransport->GetLocAddr()) {
                        found = true;
                        g_transportsPair[localTransport] = remoteTransport;
                        g_transportsPair[remoteTransport] = localTransport;
                    }
                }
                if (!found) {
                    HCCL_ERROR("fail to found corresponding transport, localTransport is from rank:%u,Ip:%s to rank:%u,Ip:%s",
                        localRank, remoteRank, localAddr.Describe().c_str(), rmtAddr.Describe().c_str());
                }
            }
        }
    }

    // 给每个transport生成待交换的数据
    for (u32 myRank = 0; myRank < rankSize; myRank++) {
        RankInfoRecorder::Global()->SetRankId(myRank);
        CcuInsPreprocessor* ccuInsProcess = communicators[myRank]->GetCcuInsPreprocessor();
        ccuInsProcess->GetCcuComm()->GetCcuTransportMgr()->GenSendData();
    }

    // 每个transport接收数据，并进行注册
    for (u32 myRank = 0; myRank < rankSize; myRank++) {
        RankInfoRecorder::Global()->SetRankId(myRank);
        CcuInsPreprocessor* ccuInsProcess = communicators[myRank]->GetCcuInsPreprocessor();
        ccuInsProcess->GetCcuComm()->GetCcuTransportMgr()->RecvData();
        ccuInsProcess->RegisterCtx(false);
    }

    // 生成channel相关的信息
    for (u32 localRank = 0; localRank < rankSize; localRank++) {
        for (auto& iter : g_allRankTransports[localRank]) {
            u32 remoteRank = iter.first;
            for (auto& localTransport : g_allRankTransports[localRank][remoteRank]) {
                CcuTransport* remoteTransport = g_transportsPair[localTransport];
                u32 remoteDieId = remoteTransport->GetDieId();
                RemoteDieInfo dieInfo {remoteRank, remoteDieId};
                u32 localDieId = localTransport->GetDieId();
                u32 localChannelId = localTransport->GetChannelId();
                g_allRankChannelInfo[localRank][localDieId][localChannelId] = dieInfo;
            }
        }
    }

    // 补充环回链路信息
    for (u32 localRank = 0; localRank < rankSize; localRank++) {
        if (topoType == TopoNetworkType::TOPO_TYPE_2D) {
            g_allRankChannelInfo[localRank][0][0] = RemoteDieInfo{localRank, 1};
            g_allRankChannelInfo[localRank][1][0] = RemoteDieInfo{localRank, 0};
        } else {
            g_allRankChannelInfo[localRank][0][0] = RemoteDieInfo{localRank, 0};
        }
    }

    // 打印各个Rank的channel信息
    for (u32 srcRank = 0; srcRank < rankSize; srcRank++) {
        HCCL_INFO("###########################");
        for (auto& rankChannels : g_allRankChannelInfo[srcRank]) {
            u32 dieId = rankChannels.first;
            for (auto& iter : rankChannels.second)
            HCCL_INFO("src rank is %d, local dieId is %u, channel is %u, dst rank is %u, dst dieId is %u",
                srcRank, dieId, iter.first, iter.second.dstRank, iter.second.remoteDieId);
        }
    }

    for (u32 myRank = 0; myRank < rankSize; myRank++) {
        RankInfoRecorder::Global()->SetRankId(myRank);
        communicators[myRank]->TransformTask();
    }

    return HcclResult::HCCL_SUCCESS;
}

TaskNode* UpdateNodeForCcuGraph(TaskNode *node,  std::set<TaskNode *>& simulatedNodes)
{
    TaskNode* retNode = node;
    if (node->task != nullptr && node->task->GetType() == TaskTypeStub::CCU_GRAPH) {
        // 首次进入子图
        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(node->task);
        retNode = curCcuTask->ccuHeadTaskNode;
        simulatedNodes.insert(retNode);
    } else if (node->task != nullptr && node->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
        // 走到子图的最后一个子节点了，就回到整图
        TaskStubSubGraphEnd *subGraphEnd = dynamic_cast<TaskStubSubGraphEnd *>(node->task);
        retNode = subGraphEnd->subGraphNode;
    }
    return retNode;
}

TaskNodePtr GetCcuTaskHead(TaskNodePtr node)
{
    TaskNode* retNode = node;
    if (node->task != nullptr && node->task->GetType() == TaskTypeStub::CCU_GRAPH) {
        // 首次进入子图
        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(node->task);
        retNode = curCcuTask->ccuHeadTaskNode;
    } else if (node->task != nullptr && node->task->GetType() == TaskTypeStub::SUB_GRAPH_END) {
        // 走到子图的最后一个子节点了，就回到整图
        TaskStubSubGraphEnd *subGraphEnd = dynamic_cast<TaskStubSubGraphEnd *>(node->task);
        retNode = subGraphEnd->subGraphNode;
    }
    return retNode;
}

HcclResult GetNewNode(const Ori2NewNodeMap &originNode2copyNode, TaskNodePtr oldNode, TaskNodePtr &newNode, bool retErr = true)
{
    if (oldNode == nullptr && !retErr) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (oldNode == nullptr && retErr) {
        HCCL_ERROR("[GetNewNode] target node is nullptr.");
        return HCCL_E_INTERNAL;
    }

    auto iter = originNode2copyNode.find(oldNode);
    if (iter == originNode2copyNode.end() && retErr) {
        HCCL_ERROR("[GetNewNode] cannot find new node to pair with the targe node.");
        return HCCL_E_INTERNAL;
    }
    newNode = iter->second;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CopyCcuSubGraphNode(TaskStub *originCcu, TaskStub **newCcu,
    std::vector<std::pair<TaskStubPtr, TaskStubPtr>> &ccuGraphs, std::vector<CcuOri2NewNodeMap> &AllOri2NewNodeMap)
{
    if (originCcu->GetType() != TaskTypeStub::CCU_GRAPH) {
        HCCL_ERROR("origin node is not ccu graph");
        return HcclResult::HCCL_E_INTERNAL;
    }

    TaskStubCcuGraph *oriCcuTask = dynamic_cast<TaskStubCcuGraph *>(originCcu);
    *newCcu = new TaskStubCcuGraph(oriCcuTask);
    TaskStubCcuGraph *newCcuTask = dynamic_cast<TaskStubCcuGraph *>(*newCcu);
    ccuGraphs.emplace_back(std::make_pair(originCcu, *newCcu));

    auto rankId = oriCcuTask->rankId;

    Ori2NewNodeMap originNode2copyNode; // 用来收录原节点到新节点的映射
    originNode2copyNode[oriCcuTask->ccuHeadTaskNode] = newCcuTask->ccuHeadTaskNode;
    // 拷贝节点
    for (auto &oriNode : oriCcuTask->toDeleteTaskNode_) {
        if (oriNode == oriCcuTask->ccuHeadTaskNode) {
            continue;
        }
        auto newNode = new TaskNode(oriNode->task, oriNode->rankIdx, oriNode->queIdx, oriNode->pos);
        originNode2copyNode[oriNode] = newNode;
        newCcuTask->toDeleteTaskNode_.push_back(newNode);
    }
    
    AllOri2NewNodeMap[rankId][originCcu] = originNode2copyNode;

    // 恢复内存冲突改造所需的成员变量
    // 拷贝loop节点 —— loop并行化改造
    for (const auto &loopGroupInfo : oriCcuTask->loopGroupInfo_) {
        std::vector<LoopInfo> loopGroup;
        for (const auto &loop : loopGroupInfo) {
            loopGroup.push_back(LoopInfo(originNode2copyNode.at(loop.loopStart), originNode2copyNode.at(loop.loopEnd)));
        }
        newCcuTask->loopGroupInfo_.push_back(loopGroup);
    }
    for (auto it = oriCcuTask->localPostWaitPairs_.begin(); it != oriCcuTask->localPostWaitPairs_.end(); ++it) {
        auto newLocPost = originNode2copyNode.at(it->first);
        newCcuTask->localPostWaitPairs_[newLocPost] = originNode2copyNode.at(it->second);
    }
    // 拷贝双边语义节点 —— 单边转双边改造
    newCcuTask->bilateralNodes_.resize(newCcuTask->queueNum_);
    for (const auto &part1 : oriCcuTask->bilateralPart1_) {
        std::map<TaskNodePtr, TaskNodePtr> mapTmp;
        for (auto iter = part1.begin(); iter != part1.end(); ++iter) {
            TaskNodePtr newPost = nullptr;
            TaskNodePtr newWait = nullptr;
            CHK_RET(GetNewNode(originNode2copyNode, iter->first, newPost, true));
            CHK_RET(GetNewNode(originNode2copyNode, iter->second, newWait, false));
            mapTmp[newPost] = newWait;
        }
        newCcuTask->bilateralPart1_.push_back(mapTmp);
    }
    for (const auto &part2 : oriCcuTask->bilateralPart2_) {
        std::map<TaskNodePtr, TaskNodePtr> mapTmp;
        for (auto iter = part2.begin(); iter != part2.end(); ++iter) {
            TaskNodePtr newPost = nullptr;
            TaskNodePtr newWait = nullptr;
            CHK_RET(GetNewNode(originNode2copyNode, iter->first, newPost, true));
            CHK_RET(GetNewNode(originNode2copyNode, iter->second, newWait, false));
            mapTmp[newPost] = newWait;
        }
        newCcuTask->bilateralPart2_.push_back(mapTmp);
    }

    // 拷贝异步节点 —— 并行化改造
    for (const auto &parNode : oriCcuTask->parallelNodes_) {
        newCcuTask->parallelNodes_.push_back(originNode2copyNode.at(parNode));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult QueryCcuGraphNode(const Ori2NewNodeMap &ccuNodeMap,
    const std::vector<CcuOri2NewNodeMap> &AllOri2NewNodeMap, TaskNodePtr oriNode, TaskNodePtr &newNode)
{
    auto locRes = ccuNodeMap.find(oriNode);
    if (locRes == ccuNodeMap.end()) {
        if (oriNode->task->GetType() != TaskTypeStub::WAIT && oriNode->task->GetType() != TaskTypeStub::POST) {
            HCCL_ERROR("[CopyCcuSubGraph] can not find node in local ccu graph.");
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        uint32_t rmtRankId = 0;
        TaskStubPtr rmtCcuTaskPtr = nullptr;
        if (oriNode->task->GetType() == TaskTypeStub::POST) {
            TaskStubPost *candPost = dynamic_cast<TaskStubPost *>(oriNode->task);
            if (candPost->ccuTaskPtr_ == 0) {
                HCCL_ERROR("[QueryCcuGraphNode] cannot find ccu subgraph head node by post node");
                return HCCL_E_INTERNAL;
            }
            TaskStubCcuGraph *rmtCcuTask = reinterpret_cast<TaskStubCcuGraph *>(candPost->ccuTaskPtr_);
            rmtRankId = oriNode->rankIdx;
            rmtCcuTaskPtr = reinterpret_cast<TaskStubPtr>(candPost->ccuTaskPtr_);
        } else if (oriNode->task->GetType() == TaskTypeStub::WAIT) {
            TaskStubWait *candWait = dynamic_cast<TaskStubWait *>(oriNode->task);
            if (candWait->ccuTaskPtr_ == 0) {
                HCCL_ERROR("[QueryCcuGraphNode] cannot find ccu subgraph head node by wait node");
                return HCCL_E_INTERNAL;
            }
            TaskStubCcuGraph *rmtCcuTask = reinterpret_cast<TaskStubCcuGraph *>(candWait->ccuTaskPtr_);
            rmtRankId = oriNode->rankIdx;
            rmtCcuTaskPtr = reinterpret_cast<TaskStubPtr>(candWait->ccuTaskPtr_);
        }
        auto rmtCcuNodeMapIter = AllOri2NewNodeMap[rmtRankId].find(rmtCcuTaskPtr);
        if (rmtCcuNodeMapIter == AllOri2NewNodeMap[rmtRankId].end()) {
            HCCL_ERROR("[QueryCcuGraphNode] can not find node map of remote ccu graph.");
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        auto rmtCcuNodeMap = rmtCcuNodeMapIter->second;
        auto rmtRes = rmtCcuNodeMap.find(oriNode);
        if (rmtRes == rmtCcuNodeMap.end()) {
            HCCL_ERROR("[CopyCcuSubGraph] can not find node in remote ccu graph.");
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        newNode = rmtRes->second;
        return HcclResult::HCCL_SUCCESS;
    }
    newNode = locRes->second;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CopyCcuSubGraphConnection(std::vector<std::pair<TaskStubPtr, TaskStubPtr>> &ccuGraphs,
    const std::vector<CcuOri2NewNodeMap> &AllOri2NewNodeMap)
{
    for (auto &ccuPair : ccuGraphs) {
        TaskStubCcuGraph *oriCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuPair.first);
        TaskStubCcuGraph *newCcuTask = dynamic_cast<TaskStubCcuGraph *>(ccuPair.second);

        auto rankId = oriCcuTask->rankId;
        auto ccuNodeMapIter = AllOri2NewNodeMap[rankId].find(ccuPair.first);
        if (ccuNodeMapIter == AllOri2NewNodeMap[rankId].end()) {
            HCCL_ERROR("[CopyCcuSubGraphConnection] can not find node map of ccu graph [%s].", oriCcuTask->des.c_str());
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        auto ccuNodeMap = ccuNodeMapIter->second;

        // 按原节点，拷贝副本连接关系
        for (auto &oriNode : oriCcuTask->toDeleteTaskNode_) {
            for (auto &parent : oriNode->parents) {
                TaskNodePtr newParent = nullptr;
                CHK_RET(QueryCcuGraphNode(ccuNodeMap, AllOri2NewNodeMap, parent, newParent));
                ccuNodeMap[oriNode]->parents.push_back(newParent);
            }
            for (auto &child : oriNode->children) {
                TaskNodePtr newChild = nullptr;
                CHK_RET(QueryCcuGraphNode(ccuNodeMap, AllOri2NewNodeMap, child, newChild));
                ccuNodeMap[oriNode]->children.push_back(newChild);
            }
        }
    }
    
    return HcclResult::HCCL_SUCCESS;
}
}