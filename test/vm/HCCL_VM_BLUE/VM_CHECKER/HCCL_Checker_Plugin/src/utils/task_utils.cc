#include "log.h"
#include "task_utils.h"
#include "data_slice.h"
#include "checker_def.h"
#include "task_meta_defs.h"
#include "hccl_types.h"
#include "storage_manager.h"
#include "ccu_instr_info.h"
#include "task_ccu.h"
#include "hccl_vm_log.h"

#include <iostream>

namespace HcclSim {

LinkProtoStub GetLinkProto(uint8_t commProtocol) {
    if (commProtocol == CommProtocol::COMM_PROTOCOL_ROCE) {
        return LinkProtoStub::RDMA;
    }
    return LinkProtoStub::SDMA;
}

uint64_t CalcDataSize(HcclDataType dataType, uint64_t dataCount)
{
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        // invalid data type
        HCCL_ERROR("[CalcDataSize] invalid dataType %d", dataType);
        return 0;
    }
    uint64_t dataTypeSize = CHECK_SIZE_TABLE[dataType];
    return dataTypeSize * dataCount;
}

// rankId, dieId, missionId
std::map<uint32_t, std::map<uint8_t, std::map<uint8_t, std::shared_ptr<HcclSim::TaskStubCcuGraph>>>> g_missionTask;

std::shared_ptr<TaskStub> ConvertTask(const HcclSim::StorageManager& storage, HcclTaskMetaData hcclTask)
{
    switch (hcclTask.taskType) {
        case HccLTaskMetaType::NOTIFY_WAIT: {
            RankId remoteRank = hcclTask.taskData.notify.srcRankId;
            if (remoteRank == hcclTask.rankId) {
                // Local
                return std::make_shared<HcclSim::TaskStubLocalWaitFrom>(hcclTask.taskData.notify.notifyId);
            } else {
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.notify.protocol);
                HcclSim::LinkInfo link(linkType);
                return std::make_shared<HcclSim::TaskStubWait>(remoteRank, link, hcclTask.taskData.notify.notifyId);
            }
            break;
        }
        case HccLTaskMetaType::NOTIFY_RECORD: {
            RankId remoteRank = hcclTask.taskData.notify.dstRankId;
            if (remoteRank == hcclTask.rankId) {
                // Local
                return std::make_shared<HcclSim::TaskStubLocalPostTo>(hcclTask.taskData.notify.notifyId);
            } else {
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.notify.protocol);
                HcclSim::LinkInfo link(linkType);
                return std::make_shared<HcclSim::TaskStubPost>(remoteRank, link, hcclTask.taskData.notify.notifyId);
            }
            break;
        }
        case HccLTaskMetaType::REDUCE: {
            RankId srcRank = hcclTask.taskData.reduce.srcRankId;
            RankId dstRank = hcclTask.taskData.reduce.dstRankId;
            uint64_t rankSrcOffset = hcclTask.taskData.reduce.srcOffset;
            uint64_t rankDstOffset = hcclTask.taskData.reduce.dstOffset;
            HcclDataType dataType = static_cast<HcclDataType>(hcclTask.taskData.reduce.dataType);
            uint64_t dataCount = hcclTask.taskData.reduce.dataCount;
            uint64_t size = CalcDataSize(dataType, dataCount);
            HcclReduceOp reduceOp = static_cast<HcclReduceOp>(hcclTask.taskData.reduce.reduceOp);
            
            DataSlice srcDataSlice = StorageManager::GetInstance().GetDataSlice(srcRank, rankSrcOffset, size);
            DataSlice dstDataSlice = StorageManager::GetInstance().GetDataSlice(dstRank, rankDstOffset, size);
            // PluginGetDataSlice(srcRank, rankSrcOffset, size, &srcDataSlice);
            // PluginGetDataSlice(dstRank, rankDstOffset, size, &dstDataSlice);

            if (srcRank == dstRank) {
                // Local
                return std::make_shared<HcclSim::TaskStubLocalReduce>(srcDataSlice, dstDataSlice, dataType, reduceOp);
            } else if (srcRank == hcclTask.rankId) {
                // Write
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                return std::make_shared<HcclSim::TaskStubWriteReduce>(dstRank, link, srcDataSlice, dstDataSlice, dataType, reduceOp);
            } else if (dstRank == hcclTask.rankId) {
                // Read
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                return std::make_shared<HcclSim::TaskStubReadReduce>(srcRank, link, dstDataSlice, srcDataSlice, dataType, reduceOp);
            }

            break;
        }
        case HccLTaskMetaType::MEM_CPY: {
            RankId srcRank = hcclTask.taskData.transMem.srcRankId;
            RankId dstRank = hcclTask.taskData.transMem.dstRankId;
            uint64_t rankSrcOffset = hcclTask.taskData.transMem.srcOffset;
            uint64_t rankDstOffset = hcclTask.taskData.transMem.dstOffset;
            uint64_t size = hcclTask.taskData.transMem.len;

            DataSlice srcDataSlice = StorageManager::GetInstance().GetDataSlice(srcRank, rankSrcOffset, size);
            DataSlice dstDataSlice = StorageManager::GetInstance().GetDataSlice(dstRank, rankDstOffset, size);
            
            // PluginGetDataSlice(srcRank, rankSrcOffset, size, &srcDataSlice);
            // PluginGetDataSlice(dstRank, rankDstOffset, size, &dstDataSlice);

            if (srcRank == dstRank) {
                // Local
                return std::make_shared<HcclSim::TaskStubLocalCopy>(srcDataSlice, dstDataSlice);
            } else if (srcRank == hcclTask.rankId) {
                // Write
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                return std::make_shared<HcclSim::TaskStubWrite>(dstRank, link, srcDataSlice, dstDataSlice);
            } else if (dstRank == hcclTask.rankId) {
                // Read
                LinkProtoStub linkType = GetLinkProto(hcclTask.taskData.transMem.protocol);
                HcclSim::LinkInfo link(linkType);
                return std::make_shared<HcclSim::TaskStubRead>(srcRank, link, dstDataSlice, srcDataSlice);
            }
            break;
        }
        case HccLTaskMetaType::CCU_GRAPH: {
            auto dieId = hcclTask.taskData.ccu.dieId;
            auto missionId = hcclTask.taskData.ccu.missionId;
            // 组装SQE数据
            std::vector<HcclSim::CcuTaskParam> missionParam;
            HcclSim::CcuTaskParam ccuParam;
            ccuParam.dieId = dieId;
            ccuParam.missionId = missionId;
            ccuParam.timeout   = hcclTask.taskData.ccu.timeout;
            ccuParam.instStartId = hcclTask.taskData.ccu.instStartId;
            ccuParam.instCnt = hcclTask.taskData.ccu.instCnt;
            ccuParam.key = hcclTask.taskData.ccu.key;
            ccuParam.argSize = hcclTask.taskData.ccu.argSize;
            memcpy(ccuParam.args, hcclTask.taskData.ccu.args, sizeof(uint64_t) * ccuParam.argSize);
            missionParam.push_back(ccuParam);
            HCCL_VM_INFO("zhf-rank {}, dieId= {}", hcclTask.rankId, static_cast<uint32_t>(dieId));
            HCCL_VM_INFO("[ConvertTask-1] Get sqe info: missionId= {:d}, startId= {:d}, cnt= {:d}, argSize= {:d}",
                missionId, ccuParam.instStartId, ccuParam.instCnt, ccuParam.argSize);

            if (g_missionTask.find(hcclTask.rankId) != g_missionTask.end()) {
                auto rankMap = g_missionTask[hcclTask.rankId];
                if (rankMap.find(dieId) != rankMap.end()) {
                    auto dieMap = rankMap[dieId];
                    if (dieMap.find(missionId) != dieMap.end()) {
                        dieMap[missionId]->AddCcuParams(ccuParam);
                        return nullptr;
                    }
                }
            }

            // 组装微码指令数据
            auto hvmInstrData = storage.GetHvmInstrData();
            for (auto &ccuInstr : hvmInstrData.instr_data) {
                if (ccuInstr.desc.rank_id != hcclTask.rankId || ccuInstr.desc.die_id != dieId) {
                    continue;
                }
                hcomm::CcuRep::CcuInstrInfo ccuInstrInfo;
                ccuInstrInfo.instrVec = ccuInstr.data;
                HCCL_VM_INFO("[ConvertTask] Create new ccu graph base node, rank_id= {}, die_id= {}",
                    ccuInstr.desc.rank_id, static_cast<uint32_t>(ccuInstr.desc.die_id));
                auto taskPtr = std::make_shared<HcclSim::TaskStubCcuGraph>(ccuInstrInfo, missionParam, ccuInstr.desc.rank_id);
                g_missionTask[hcclTask.rankId][dieId][missionId] = taskPtr;
                return taskPtr;
            }
            break;
        }
        default: {
            HCCL_ERROR("[ConvertTask] taskType unknown %d", hcclTask.taskType);
            break;
        }
    }
    return nullptr;
}

void ConvertTaskQueue(AllRankTaskQueues& allRankTaskQueues)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    auto taskMetaVec = storage.GetHvmTaskMetaData().task_meta;
    uint32_t size = taskMetaVec.size();
    uint32_t outputInterval = std::max(1u, size / 10); // 十分之一数据量
    outputInterval = std::min(outputInterval, 100000u); // 不超过10万条
    for (uint32_t i = 0; i < size; i++) {
        auto task = ConvertTask(storage, taskMetaVec[i]);
        if (task == nullptr) {
            continue;
        }
        if (size > 10000 && i % outputInterval == 0) {
            HCCL_VM_INFO("[ConvertTaskQueue] Processing at index: {:d} / {:d} ({:f}% complete)",
                i, size, i * 100.0 / size);
        }
        uint32_t rankId = taskMetaVec[i].rankId;
        uint32_t streamId = (taskMetaVec[i].streamId >> 32);
        
        if (allRankTaskQueues.find(rankId) == allRankTaskQueues.end()) {
            allRankTaskQueues[rankId] = std::vector<std::vector<std::shared_ptr<HcclSim::TaskStub>>> {};
        }
        
        if (allRankTaskQueues[rankId].size() <= streamId) {
            allRankTaskQueues[rankId].resize(streamId + 1);
        }
        
        allRankTaskQueues[rankId][streamId].push_back(task);
    }
}

}  // namespace HcclSim