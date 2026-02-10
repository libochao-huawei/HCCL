/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump_utils.h"
#include <fstream>
#include <queue>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>
#include "storage_manager.h"
#include "task_ccu.h"

std::string GetFileName(DumpDataType dumpDataType) {
    if (mkdir("output", 0755) == -1) {
        if (errno != EEXIST) {
            // 如果不是因为目录已存在而导致的失败，打印错误
            std::cerr << "[Dumper] 无法创建目录 'output': " << strerror(errno) << std::endl;
        }
    }
    std::string prefix = "output/";
    switch (dumpDataType) {
        case DumpDataType::TASK_META:
            prefix += "task_meta_";
            break;
        case DumpDataType::TASK_STUB:
            prefix += "task_stub_";
            break;
        default:
            prefix += "unknown_";
    }

    std::ostringstream oss;
    oss << prefix << HcclSim::StorageManager::GetInstance().GetDataId() << ".json";

    return oss.str();
}

std::string HcclCMDTypeToString(const HcclCMDType &cmd)
{
    switch (cmd) {
        case HCCL_CMD_INVALID:          return "HCCL_CMD_INVALID";
        case HCCL_CMD_BROADCAST:        return "HCCL_CMD_BROADCAST";
        case HCCL_CMD_ALLREDUCE:        return "HCCL_CMD_ALLREDUCE";
        case HCCL_CMD_REDUCE:           return "HCCL_CMD_REDUCE";
        case HCCL_CMD_SEND:             return "HCCL_CMD_SEND";
        case HCCL_CMD_RECEIVE:          return "HCCL_CMD_RECEIVE";
        case HCCL_CMD_ALLGATHER:        return "HCCL_CMD_ALLGATHER";
        case HCCL_CMD_REDUCE_SCATTER:   return "HCCL_CMD_REDUCE_SCATTER";
        case HCCL_CMD_ALLTOALLV:        return "HCCL_CMD_ALLTOALLV";
        case HCCL_CMD_ALLTOALLVC:       return "HCCL_CMD_ALLTOALLVC";
        case HCCL_CMD_ALLTOALL:         return "HCCL_CMD_ALLTOALL";
        case HCCL_CMD_GATHER:           return "HCCL_CMD_GATHER";
        case HCCL_CMD_SCATTER:          return "HCCL_CMD_SCATTER";
        case HCCL_CMD_BATCH_SEND_RECV:  return "HCCL_CMD_BATCH_SEND_RECV";
        case HCCL_CMD_BATCH_PUT:        return "HCCL_CMD_BATCH_PUT";
        case HCCL_CMD_BATCH_GET:        return "HCCL_CMD_BATCH_GET";
        case HCCL_CMD_ALLGATHER_V:      return "HCCL_CMD_ALLGATHER_V";
        case HCCL_CMD_REDUCE_SCATTER_V: return "HCCL_CMD_REDUCE_SCATTER_V";
        case HCCL_CMD_BATCH_WRITE:      return "HCCL_CMD_BATCH_WRITE";
        case HCCL_CMD_HALF_ALLTOALLV:   return "HCCL_CMD_HALF_ALLTOALLV";
        case HCCL_CMD_ALL:              return "HCCL_CMD_ALL";
        case HCCL_CMD_FINALIZE:         return "HCCL_CMD_FINALIZE";
        case HCCL_CMD_INTER_GROUP_SYNC: return "HCCL_CMD_INTER_GROUP_SYNC";
        case HCCL_CMD_INIT:             return "HCCL_CMD_INIT";
        case HCCL_CMD_BARRIER:          return "HCCL_CMD_BARRIER";
        case HCCL_CMD_MAX:              return "HCCL_CMD_MAX";
        default:
            HCCL_VM_ERROR("[Dumper] Invalid HCCL CMD {}", static_cast<u32>(cmd));
            return "UNKNOWN_HCCL_CMD";
    }
}

std::string HcclDataTypeToString(const HcclDataType &dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:     return "INT8";     
        case HCCL_DATA_TYPE_INT16:    return "INT16";    
        case HCCL_DATA_TYPE_INT32:    return "INT32";    
        case HCCL_DATA_TYPE_FP16:     return "FP16";     
        case HCCL_DATA_TYPE_FP32:     return "FP32";     
        case HCCL_DATA_TYPE_INT64:    return "INT64";    
        case HCCL_DATA_TYPE_UINT64:   return "UINT64";   
        case HCCL_DATA_TYPE_UINT8:    return "UINT8";    
        case HCCL_DATA_TYPE_UINT16:   return "UINT16";   
        case HCCL_DATA_TYPE_UINT32:   return "UINT32";   
        case HCCL_DATA_TYPE_FP64:     return "FP64";     
        case HCCL_DATA_TYPE_BFP16:    return "BFP16";    
        case HCCL_DATA_TYPE_INT128:   return "INT128";   
        case HCCL_DATA_TYPE_HIF8:     return "HIF8";     
        case HCCL_DATA_TYPE_FP8E4M3:  return "FP8E4M3";  
        case HCCL_DATA_TYPE_FP8E5M2:  return "FP8E5M2";  
        case HCCL_DATA_TYPE_FP8E8M0:  return "FP8E8M0";  
        case HCCL_DATA_TYPE_RESERVED: return "RESERVED"; 
        default:
            HCCL_VM_ERROR("[Dumper] Invalid dataType {}", static_cast<u32>(dataType));
            return "UNKNOWN";   // 处理未知枚举值
    }
}

json GenBasicInfo(DumpDataType dumpDataType, uint32_t taskCount)
{
    std::string dumpDataTypeString;
    switch (dumpDataType) {
        case DumpDataType::TASK_META:
            dumpDataTypeString = "TASK_META";
            break;
        case DumpDataType::TASK_STUB:
            dumpDataTypeString = "TASK_STUB";
            break;
        default:
            dumpDataTypeString = "UNKNOWN";
    }

    json j;
    j["dumpDataType"] = dumpDataTypeString;
    j["taskCount"] = taskCount;
    HcclSim::CheckerParam checkerParam = HcclSim::StorageManager::GetInstance().GetCheckerParam();
    j["rankSize"] = checkerParam.rankSize;
    j["CMDType"] = HcclCMDTypeToString(checkerParam.cmdType);
    j["dataType"] = HcclDataTypeToString(checkerParam.dataType);
    j["dataCount"] = checkerParam.dataCount;
    return j;
}

json TaskMetaToJson(const HcclTaskMetaData& s)
{
    json j;
    ResId streamIdInfo(s.streamId);

    j = json{
        {"taskType", s.taskType},
        {"commId", s.commId},
        {"rankId", s.rankId},
        {"streamId", s.streamId},
        {"streamIdStr", streamIdInfo.ToString()},
    };

    // 根据任务类型，序列化相应的union成员
    switch (s.taskType) {
        case HccLTaskMetaType::NOTIFY_WAIT:
        case HccLTaskMetaType::NOTIFY_RECORD:
        {
            ResId notifyIdInfo(s.taskData.notify.notifyId);
            j["taskData"] = json{
                {"srcRankId", s.taskData.notify.srcRankId},
                {"notifyId", s.taskData.notify.notifyId},
                {"notifyIdStr", notifyIdInfo.ToString()},
                {"dstRankId", s.taskData.notify.dstRankId},
                {"notifyCount", s.taskData.notify.notifyCount},
                {"protocol", s.taskData.notify.protocol}
            };
            break;
        }
        case HccLTaskMetaType::REDUCE:
        {
            j["taskData"] = json{
                {"srcRankId", s.taskData.reduce.srcRankId},
                {"srcOffset", ToHexStr(s.taskData.reduce.srcOffset)},
                {"dstRankId", s.taskData.reduce.dstRankId},
                {"dstOffset", ToHexStr(s.taskData.reduce.dstOffset)},
                {"dataType", s.taskData.reduce.dataType},
                {"dataCount", s.taskData.reduce.dataCount},
                {"reduceOp", s.taskData.reduce.reduceOp},
                {"protocol", s.taskData.reduce.protocol}
            };
            break;
        }
        case HccLTaskMetaType::MEM_CPY:
        {
            j["taskData"] = json{
                {"srcRankId", s.taskData.transMem.srcRankId},
                {"srcOffset", ToHexStr(s.taskData.transMem.srcOffset)},
                {"dstRankId", s.taskData.transMem.dstRankId},
                {"dstOffset", ToHexStr(s.taskData.transMem.dstOffset)},
                {"len", s.taskData.transMem.len},
                {"protocol", s.taskData.transMem.protocol}
            };
            break;
        }
        case HccLTaskMetaType::CCU_GRAPH:
        {
            std::vector<uint64_t> args(s.taskData.ccu.args, s.taskData.ccu.args + RT_CCU_SQE_ARGS_LEN);
            j["taskData"] = json{
                {"dieId", s.taskData.ccu.dieId},
                {"missionId", s.taskData.ccu.missionId},
                {"timeout", s.taskData.ccu.timeout},
                {"instStartId", s.taskData.ccu.instStartId},
                {"instCnt", s.taskData.ccu.instCnt},
                {"key", s.taskData.ccu.key},
                {"argSize", s.taskData.ccu.argSize},
                {"args", args}
            };
            break;
        }
        default:
        {
            HCCL_VM_ERROR("[Dumper] Invalid task type {}", static_cast<u32>(s.taskType));
            j["taskData"] = json{};
            break;
        }
    }
    return j;
}

HcclResult DumpTaskMetaToFile(const std::vector<HcclTaskMetaData>& taskCollection)
{
    std::string fileName = GetFileName(DumpDataType::TASK_META);
    std::ofstream file(fileName);
    if (!file.is_open()) {
        HCCL_VM_ERROR("[Dumper] Open TaskMeta Dump file failed.");
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;
    }
    json j = json::array();
    for (uint32_t i = 0; i < taskCollection.size(); i++) {
        json item = TaskMetaToJson(taskCollection[i]);

        // 给生成的 json 对象插入 index 属性
        item["index"] = i; 

        j.push_back(item);
    }
    json dumpJson = GenBasicInfo(DumpDataType::TASK_META, taskCollection.size());
    dumpJson["data"] = j;
    file << dumpJson;
    file.close();
    return HcclResult::HCCL_SUCCESS;
}

/**
 * @brief 第一阶段：迭代式全局预编号
 * 遍历主图及所有嵌套子图，为所有 TaskNode 分配唯一的 localStep。
 */
void AssignGlobalIndices(HcclSim::TaskNode* head, uint32_t& globalCounter)
{
    if (!head) return;
    std::queue<HcclSim::TaskNode*> q;
    std::set<HcclSim::TaskNode*> visited;

    q.push(head);
    visited.insert(head);

    while (!q.empty()) {
        HcclSim::TaskNode* curr = q.front();
        q.pop();

        // 分配全局唯一 ID
        curr->localStep = globalCounter++;

        // 递归检查 CCU 子图内部节点
        if (curr->task && curr->task->GetType() == HcclSim::TaskTypeStub::CCU_GRAPH) {
            auto ccuTask = dynamic_cast<HcclSim::TaskStubCcuGraph*>(curr->task);
            if (ccuTask && ccuTask->ccuHeadTaskNode) {
                for (auto& subChild : ccuTask->ccuHeadTaskNode->children) {
                    if (visited.find(subChild) == visited.end()) {
                        visited.insert(subChild);
                        q.push(subChild);
                    }
                }
            }
        }

        // 检查普通拓扑子节点
        for (auto& child : curr->children) {
            if (visited.find(child) == visited.end()) {
                visited.insert(child);
                q.push(child);
            }
        }
    }
}

json TaskNodePtrToJson(const HcclSim::TaskNodePtr taskNode, u32 &graphCounter)
{
    json j;
    j = {
        {"id", taskNode->localStep},
        {"rankId", taskNode->rankIdx},
        {"queueId", taskNode->queIdx},
        {"pos", taskNode->pos}
    };
    std::set<HcclSim::TaskNodePtr> parentSet(taskNode->parents.begin(), taskNode->parents.end());
    j["parents"] = json::array();
    for (auto& p : parentSet) {
        j["parents"].push_back(p->localStep);
    }
    std::set<HcclSim::TaskNodePtr> childSet(taskNode->children.begin(), taskNode->children.end());
    j["children"] = json::array();
    for (auto& c : childSet) {
        j["children"].push_back(c->localStep);
    }
    if (taskNode->task) {
        j["task"] = TaskStubToJson(taskNode->task, graphCounter);
    } else {
        j["task"] = {};
    }
    return j;
}

json DataSliceToJson(const DataSlice &dataSlice)
{
    return json{
        {"type", dataSlice.GetType()},
        {"offset", ToHexStr(dataSlice.GetOffset())},
        {"size", dataSlice.GetSize()}
    };
}

json LinkProtoStubToJson(const HcclSim::LinkProtoStub &linkProto)
{
    json j;
    switch (linkProto) {
        case HcclSim::LinkProtoStub::SDMA:  j = "SDMA"; break;
        case HcclSim::LinkProtoStub::RDMA:  j = "RDMA"; break;
        case HcclSim::LinkProtoStub::CCU:   j = "CCL";  break;
        default:
            HCCL_VM_ERROR("[Dumper] Invalid linkProto {}", static_cast<u32>(linkProto));
            j = "UNKNOWN";
    }
    return j;
}

std::string HcclReduceOpToString(const HcclReduceOp &reduceOp)
{
    std::string s;
    switch (reduceOp) {
        case HCCL_REDUCE_SUM:       s = "SUM";      break;
        case HCCL_REDUCE_PROD:      s = "PROD";     break;
        case HCCL_REDUCE_MAX:       s = "MAX";      break;
        case HCCL_REDUCE_MIN:       s = "MIN";      break;
        case HCCL_REDUCE_RESERVED:  s = "RESERVED"; break;
        default:
            HCCL_VM_ERROR("[Dumper] Invalid reduceOp {}", static_cast<u32>(reduceOp));
            s = "UNKNOWN"; // 处理未知枚举值
    }
    return s;
}

/**
 * @brief 子图渲染逻辑
 */
json CcuSingleQueToJson(HcclSim::TaskNodePtr head, uint32_t rankId, uint32_t queueIdx, uint32_t& globalCounter)
{
    json j = json::array();
    std::deque<HcclSim::TaskNode*> candNode;
    std::set<HcclSim::TaskNode*> isVisitedNode;
    std::set<HcclSim::TaskNode*> alreadyPrintedNodes;

    // 预编号阶段已确保 head->localStep 有效
    j.push_back(TaskNodePtrToJson(head, globalCounter));
    alreadyPrintedNodes.insert(head);

    for (auto& child : head->children) {
        if (isVisitedNode.count(child) == 0 && child->rankIdx == rankId && child->queIdx == queueIdx) {
            candNode.push_back(child);
            isVisitedNode.insert(child);
        }
    }

    while(!candNode.empty()) {
        HcclSim::TaskNodePtr curNode = candNode.front();
        candNode.pop_front();

        for (auto& child : curNode->children) {
            if (isVisitedNode.count(child) == 0 && child->rankIdx == rankId && child->queIdx == queueIdx) {
                candNode.push_back(child);
                isVisitedNode.insert(child);
            }
        }

        bool parentsReady = true;
        for (auto& parent : curNode->parents) {
            if (parent->rankIdx == rankId && parent->queIdx == queueIdx) {
                if (alreadyPrintedNodes.count(parent) == 0) {
                    parentsReady = false;
                    break;
                }
            }
        }

        if (parentsReady) {
            j.push_back(TaskNodePtrToJson(curNode, globalCounter));
            alreadyPrintedNodes.insert(curNode);
        } else {
            candNode.push_back(curNode);
        }
    }
    return j;
}

/**
 * @brief 所有 Task 类型序列化入口
 */
json TaskStubToJson(HcclSim::TaskStub *task, uint32_t& globalCounter)
{
    HcclSim::TaskTypeStub taskType = task->GetType();
    switch (taskType) {
        case HcclSim::TaskTypeStub::LOCAL_COPY: {
            auto localCopyTask = dynamic_cast<HcclSim::TaskStubLocalCopy *>(task);
            return json{
                {"taskType", "LOCAL_COPY"},
                {"srcSlice", DataSliceToJson(localCopyTask->GetSrcSlice())},
                {"dstSlice", DataSliceToJson(localCopyTask->GetDstSlice())},
                {"isGenFromSync", localCopyTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::LOCAL_REDUCE: {
            auto localReduceTask = dynamic_cast<HcclSim::TaskStubLocalReduce *>(task);
            return json{
                {"taskType", "LOCAL_REDUCE"},
                {"srcSlice", DataSliceToJson(localReduceTask->GetSrcSlice())},
                {"dstSlice", DataSliceToJson(localReduceTask->GetDstSlice())},
                {"dataType", HcclDataTypeToString(localReduceTask->GetDataType())},
                {"reduceOp", HcclReduceOpToString(localReduceTask->GetReduceOp())},
                {"isGenFromSync", localReduceTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::LOCAL_BATCH_REDUCE: {
            auto localBatchReduceTask = dynamic_cast<HcclSim::TaskStubLocalBatchReduce *>(task);
            json jArray = json::array();
            for (auto &slice : localBatchReduceTask->GetSrcSlices()) {
                jArray.push_back(DataSliceToJson(slice));
            }
            return json{
                {"taskType", "LOCAL_BATCH_REDUCE"},
                {"srcSlices", jArray},
                {"dstSlice", DataSliceToJson(localBatchReduceTask->GetDstSlice())},
                {"dataType", HcclDataTypeToString(localBatchReduceTask->GetDataType())},
                {"reduceOp", HcclReduceOpToString(localBatchReduceTask->GetReduceOp())}
            };
        }
        case HcclSim::TaskTypeStub::LOCAL_POST_TO: {
            auto localPostToTask = dynamic_cast<HcclSim::TaskStubLocalPostTo *>(task);
            return json{
                {"taskType", "LOCAL_POST_TO"},
                {"notifyId", localPostToTask->GetNotifyId()},
                {"notifyIdBack", localPostToTask->GetNotifyIdBack()},
                {"postQid", localPostToTask->GetPostQid()},
                {"waitQid", localPostToTask->GetWaitQid()}
            };
        }
        case HcclSim::TaskTypeStub::LOCAL_WAIT_FROM: {
            auto localWaitFromTask = dynamic_cast<HcclSim::TaskStubLocalWaitFrom *>(task);
            return json{
                {"taskType", "LOCAL_WAIT_FROM"},
                {"notifyId", localWaitFromTask->GetNotifyId()},
                {"postQid", localWaitFromTask->GetPostQid()},
                {"waitQid", localWaitFromTask->GetWaitQid()}
            };
        }
        case HcclSim::TaskTypeStub::POST: {
            auto postTask = dynamic_cast<HcclSim::TaskStubPost *>(task);
            return json{
                {"taskType", "POST"},
                {"remoteRank", postTask->GetRemoteRank()},
                {"link", LinkProtoStubToJson(postTask->GetLinkType())},
                {"notifyId", postTask->GetNotifyId()}
            };
        }
        case HcclSim::TaskTypeStub::WAIT: {
            auto waitTask = dynamic_cast<HcclSim::TaskStubWait *>(task);
            ResId resId(waitTask->GetNotifyId());
            return json{
                {"taskType", "WAIT"},
                {"remoteRank", waitTask->GetRemoteRank()},
                {"link", LinkProtoStubToJson(waitTask->GetLinkType())},
                {"notifyId", waitTask->GetNotifyId()}
            };
        }
        case HcclSim::TaskTypeStub::READ: {
            auto readTask = dynamic_cast<HcclSim::TaskStubRead *>(task);
            return json{
                {"taskType", "READ"},
                {"remoteRank", readTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(readTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(readTask->GetRemoteSlice())},
                {"link", LinkProtoStubToJson(readTask->GetLinkType())},
                {"isGenFromSync", readTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::READ_REDUCE: {
            auto readReduceTask = dynamic_cast<HcclSim::TaskStubReadReduce *>(task);
            return json{
                {"taskType", "READ_REDUCE"},
                {"remoteRank", readReduceTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(readReduceTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(readReduceTask->GetRemoteSlice())},
                {"dataType", HcclDataTypeToString(readReduceTask->GetDataType())},
                {"reduceOp", HcclReduceOpToString(readReduceTask->GetReduceOp())},
                {"link", LinkProtoStubToJson(readReduceTask->GetLinkType())},
                {"isGenFromSync", readReduceTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::WRITE: {
            auto writeTask = dynamic_cast<HcclSim::TaskStubWrite *>(task);
            return json{
                {"taskType", "WRITE"},
                {"remoteRank", writeTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(writeTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(writeTask->GetRemoteSlice())},
                {"link", LinkProtoStubToJson(writeTask->GetLinkType())},
                {"isGenFromSync", writeTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::WRITE_REDUCE: {
            auto writeReduceTask = dynamic_cast<HcclSim::TaskStubWriteReduce *>(task);
            return json{
                {"taskType", "WRITE_REDUCE"},
                {"remoteRank", writeReduceTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(writeReduceTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(writeReduceTask->GetRemoteSlice())},
                {"dataType", HcclDataTypeToString(writeReduceTask->GetDataType())},
                {"reduceOp", HcclReduceOpToString(writeReduceTask->GetReduceOp())},
                {"link", LinkProtoStubToJson(writeReduceTask->GetLinkType())},
                {"isGenFromSync", writeReduceTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::LOCAL_POST_TO_SHADOW: {
            auto localPostTask = dynamic_cast<HcclSim::TaskStubLocalPostToShadow *>(task);
            return json{
                {"taskType", "LOCAL_POST_TO_SHADOW"},
                {"neighborRank", localPostTask->GetNeighborRank()},
                {"curQueId", localPostTask->GetCurQueId()},
                {"peerQueId", localPostTask->GetPeerQueId()}
            };
        }
        case HcclSim::TaskTypeStub::LOCAL_WAIT_FROM_SHADOW: {
            auto localWaitTask = dynamic_cast<HcclSim::TaskStubLocalWaitFromShadow *>(task);
            return json{
                {"taskType", "LOCAL_WAIT_FROM_SHADOW"},
                {"neighborRank", localWaitTask->GetNeighborRank()},
                {"curQueId", localWaitTask->GetCurQueId()},
                {"peerQueId", localWaitTask->GetPeerQueId()}
            };
        }
        case HcclSim::TaskTypeStub::BEING_READ: {
            auto beingReadTask = dynamic_cast<HcclSim::TaskStubBeingRead *>(task);
            return json{
                {"taskType", "BEING_READ"},
                {"remoteRank", beingReadTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(beingReadTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(beingReadTask->GetRemoteSlice())},
                {"link", LinkProtoStubToJson(beingReadTask->GetLinkType())},
                {"isGenFromSync", beingReadTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::BEING_READ_REDUCE: {
            auto beingReadReduceTask = dynamic_cast<HcclSim::TaskStubBeingReadReduce *>(task);
            return json{
                {"taskType", "BEING_READ_REDUCE"},
                {"remoteRank", beingReadReduceTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(beingReadReduceTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(beingReadReduceTask->GetRemoteSlice())},
                {"dataType", HcclDataTypeToString(beingReadReduceTask->GetDataType())},
                {"reduceOp", HcclReduceOpToString(beingReadReduceTask->GetReduceOp())},
                {"link", LinkProtoStubToJson(beingReadReduceTask->GetLinkType())},
                {"isGenFromSync", beingReadReduceTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::BEING_WRITTEN: {
            auto beingWrittenTask = dynamic_cast<HcclSim::TaskStubBeingWritten *>(task);
            return json{
                {"taskType", "BEING_WRITTEN"},
                {"remoteRank", beingWrittenTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(beingWrittenTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(beingWrittenTask->GetRemoteSlice())},
                {"link", LinkProtoStubToJson(beingWrittenTask->GetLinkType())},
                {"isGenFromSync", beingWrittenTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::BEING_WRITTEN_REDUCE: {
            auto beingWrittenReduceTask = dynamic_cast<HcclSim::TaskStubBeingWrittenReduce *>(task);
            return json{
                {"taskType", "BEING_WRITTEN_REDUCE"},
                {"remoteRank", beingWrittenReduceTask->GetRemoteRank()},
                {"localSlice", DataSliceToJson(beingWrittenReduceTask->GetLocalSlice())},
                {"remoteSlice", DataSliceToJson(beingWrittenReduceTask->GetRemoteSlice())},
                {"dataType", HcclDataTypeToString(beingWrittenReduceTask->GetDataType())},
                {"reduceOp", HcclReduceOpToString(beingWrittenReduceTask->GetReduceOp())},
                {"link", LinkProtoStubToJson(beingWrittenReduceTask->GetLinkType())},
                {"isGenFromSync", beingWrittenReduceTask->IsGenFromSync()}
            };
        }
        case HcclSim::TaskTypeStub::LOOP_START: {
            auto loopStartTask = dynamic_cast<HcclSim::TaskStubLoopStart *>(task);
            return json{
                {"taskType", "LOOP_START"},
                {"loopIdx", loopStartTask->GetLoopIdx()},
                {"loopGroupIdx", loopStartTask->GetLoopGroupIdx()}
            };
        }
        case HcclSim::TaskTypeStub::LOOP_END: {
            auto loopEndTask = dynamic_cast<HcclSim::TaskStubLoopEnd *>(task);
            return json{
                {"taskType", "LOOP_END"},
                {"loopIdx", loopEndTask->loopIdx},
                {"loopGroupIdx", loopEndTask->loopGroupIdx}
            };
        }
        case HcclSim::TaskTypeStub::CCU_GRAPH: {
            auto ccuGrapthTask = dynamic_cast<HcclSim::TaskStubCcuGraph *>(task);
            json jsonCcuGraph = json::array();
            for (auto& child : ccuGrapthTask->ccuHeadTaskNode->children) {
                u32 queueIdx = child->queIdx;
                jsonCcuGraph.push_back(CcuSingleQueToJson(child, ccuGrapthTask->rankId, queueIdx, globalCounter));
            }
            return json{
                {"taskType", "CCU_GRAPH"},
                {"subGraph", jsonCcuGraph}
            };
        }
        default: {
            HCCL_VM_ERROR("[Dumper] Invalid TaskTypeStub {}", static_cast<u32>(taskType));
            return json{};
        }
    }
    return json{};
}

HcclResult DumpTaskStubToFile(HcclSim::TaskNode* headNode)
{
    std::string fileName = GetFileName(DumpDataType::TASK_STUB);
    std::ofstream file(fileName);
    if (!file.is_open()) {
        HCCL_VM_ERROR("[ERROR] Dump TaskStub: Open file failed.");
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;
    }
    // --- 全局预编号 ---
    uint32_t globalCounter = 0;
    AssignGlobalIndices(headNode, globalCounter);

    json j = json::array();
    std::queue<HcclSim::TaskNode*> queue;
    std::set<HcclSim::TaskNode*> isVisitedNode;
    queue.push(headNode);
    isVisitedNode.insert(headNode);

    while (!queue.empty()) {
        HcclSim::TaskNodePtr node = queue.front();
        queue.pop();

        for (auto &child : node->children) {
            if (isVisitedNode.count(child) == 0) {
                queue.push(child);
                isVisitedNode.insert(child);
            }
        }
        json item = TaskNodePtrToJson(node, globalCounter);
        j.push_back(item);
    }
    json dumpJson = GenBasicInfo(DumpDataType::TASK_STUB, globalCounter);
    dumpJson["data"] = j;
    file << dumpJson;
    file.close();
    return HcclResult::HCCL_SUCCESS;
}