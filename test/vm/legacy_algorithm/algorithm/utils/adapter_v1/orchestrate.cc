/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: HccL 1.0 编排实现细节
 * Create: 2025-05-6
 */
#include "orchestrate.h"
#include "transport.h"
#include "topo_meta.h"
#include "llt_common.h"
#include "rank_info_recorder.h"
#include "mem_layout.h"
#include "check_utils.h"
#include "device_info_recorder.h"

#include "topoinfo_struct.h"
#include "transformer.h"

using namespace std;

namespace hccl {
unordered_map<RankId, vector<shared_ptr<TransportCompared>>> AllTransport_;
unordered_map<Transport*, shared_ptr<TransportCompared>> links2TransportCompare_;
map<TransportType, unordered_map<RankId, unordered_map<RankId, std::shared_ptr<Transport>>>> CreatedLinksDict_;

HcclResult CheckTransportLink()
{
    u32 count = 0;
    for (auto iter = AllTransport_.begin(); iter != AllTransport_.end(); iter++) {
        RankId rankId = iter->first;
        for (auto iSend = 0; iSend < AllTransport_[rankId].size(); iSend++) {
            if (AllTransport_[rankId][iSend]->isCompared == true ||
                AllTransport_[rankId][iSend]->isValid == false) {
                continue;
            }
            RankId remoteRank = AllTransport_[rankId][iSend]->remoteRank;
            bool isMatched = false;
            auto iRecv = 0;
            for (; iRecv < AllTransport_[remoteRank].size(); iRecv++) {
                if (AllTransport_[remoteRank][iRecv]->isValid == false ||
                    AllTransport_[remoteRank][iRecv]->isCompared == true) {
                    continue;
                }
                if (AllTransport_[remoteRank][iRecv]->remoteRank != rankId) {
                    continue;
                } else {
                    isMatched = true;
                    break;
                }
            }
            if (!isMatched) {
                HCCL_ERROR("TransportID matched is failed");
                return HCCL_E_PARA;
            } else {
                AllTransport_[rankId][iSend]->isCompared = true;
                AllTransport_[remoteRank][iRecv]->isCompared = true;
                AllTransport_[rankId][iSend]->transportId_ = count;
                AllTransport_[remoteRank][iRecv]->transportId_ = count;

                AllTransport_[rankId][iSend]->remoteinputMemType = AllTransport_[remoteRank][iRecv]->inputMemType;
                AllTransport_[rankId][iSend]->remoteoutputMemType = AllTransport_[remoteRank][iRecv]->outputMemType;
                AllTransport_[remoteRank][iRecv]->remoteinputMemType = AllTransport_[rankId][iSend]->inputMemType;
                AllTransport_[remoteRank][iRecv]->remoteoutputMemType = AllTransport_[rankId][iSend]->outputMemType;
                count++;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InitCommParams(HcclCommParams &params, RankTable_t& rankTable, RankId myRank)
{
    params.rank = myRank;
    params.userRank = myRank;
    params.totalRanks = rankTable.rankNum;

    params.serverId = RankInfoRecorder::Global()->rankId2serverId[myRank];
    params.logicDevId = RankInfoRecorder::Global()->rankId2phyId[myRank];
    params.deviceType = g_CheckerDevType2HcclDevType[RankInfoRecorder::Global()->GetDevType()];

    return HCCL_SUCCESS;
}

void InitOpParam(OpParam &opParam, CheckerOpParam &checkerOpParam, RankId myRank,
    u32 rankSize, bool initStream, bool isIOSameAddr)
{
    opParam.reduceType = g_CheckerReduceOp2HcclReduceOp[checkerOpParam.reduceType];
    opParam.opType = g_CheckerOpType2HcclCMDType[checkerOpParam.opType];
    opParam.supportZeroCopy = checkerOpParam.supportZeroCopy;
    opParam.isZeroCopy = checkerOpParam.isZeroCopy;

    u32 mySuperPodId = RankInfoRecorder::Global()->rankId2superpodId[myRank];
    u32 myServerId = RankInfoRecorder::Global()->rankId2serverId[myRank];
    u32 myPhyRankId = RankInfoRecorder::Global()->rankId2phyId[myRank];

    if (g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_SEND || g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_RECEIVE) {
        if (myRank == checkerOpParam.srcRank) {
            opParam.dstRank = checkerOpParam.dstRank;
            opParam.opType = HcclCMDType::HCCL_CMD_SEND;
        } else if (myRank == checkerOpParam.dstRank) {
            opParam.srcRank = checkerOpParam.srcRank;
            opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
        } else {
            HCCL_ERROR("send_recv only support two ranks");
        }
    }

    SingleRankMemLayout mem = MemLayout::Global()->allSuperPodLayout[mySuperPodId][myServerId][myPhyRankId];
    opParam.inputPtr = mem[BufferType::INPUT].startAddr;
    if (isIOSameAddr) {
        opParam.outputPtr = mem[BufferType::INPUT].startAddr;
    } else {
        opParam.outputPtr = mem[BufferType::OUTPUT].startAddr;
    }
    if (IsAllToAllSeries(g_HcclCMDType2CheckerOpType[opParam.opType])) {
        opParam.All2AllDataDes.sendType = g_CheckerDataType2HcclDataType[checkerOpParam.All2AllDataDes.sendType];
        opParam.All2AllDataDes.recvType = g_CheckerDataType2HcclDataType[checkerOpParam.All2AllDataDes.recvType];
        opParam.All2AllDataDes.sendCount = checkerOpParam.All2AllDataDes.sendCount;
        opParam.All2AllDataDes.sendCounts = static_cast<void *>(checkerOpParam.All2AllDataDes.sendCounts.data());
        opParam.All2AllDataDes.recvCounts = static_cast<void *>(checkerOpParam.All2AllDataDes.recvCounts.data());
        opParam.All2AllDataDes.sdispls = static_cast<void *>(checkerOpParam.All2AllDataDes.sdispls.data());
        opParam.All2AllDataDes.rdispls = static_cast<void *>(checkerOpParam.All2AllDataDes.rdispls.data());
        opParam.All2AllDataDes.sendCountMatrix = static_cast<void *>(checkerOpParam.All2AllDataDes.sendCountMatrix.data());
    } else if (g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        char_t* batchSendRecvInputAddr = mem[BufferType::INPUT].startAddr;
        char_t* batchSendRecvOutputAddr = mem[BufferType::OUTPUT].startAddr;
        u64 memSizePerTask = CHECK_SIZE_TABLE[checkerOpParam.DataDes.dataType] * checkerOpParam.DataDes.count;
        checkerOpParam.allRanksSendRecvInfoVec[myRank].clear();
        if (memSizePerTask * rankSize > CHECKER_MEM_BLOCK_SIZE) {
            HCCL_ERROR("send total dataSize is larger than input or output user size buffer.");
            return;
        }
        u64 dataTotalSize = 0;
        for (u32 i = 0; i < rankSize; i++) {
            //send task
            char_t* tempInputMem = batchSendRecvInputAddr + i * memSizePerTask;
            //recv task
            char_t* tempOutputMem = batchSendRecvOutputAddr + i * memSizePerTask;
            dataTotalSize += memSizePerTask;
            checkerOpParam.allRanksSendRecvInfoVec[myRank].emplace_back(CheckerSendRecvItem{CheckerSendRecvType::CHECK_SEND, static_cast<void*>(tempInputMem),
                checkerOpParam.DataDes.count, checkerOpParam.DataDes.dataType, i});
            checkerOpParam.allRanksSendRecvInfoVec[myRank].emplace_back(CheckerSendRecvItem{CheckerSendRecvType::CHECK_RECV, static_cast<void*>(tempOutputMem),
                checkerOpParam.DataDes.count, checkerOpParam.DataDes.dataType, i});
            HCCL_INFO("[BatchSendRecv] InitOpParam : Localrank[%u], remoteRank[%u], send userbuffer[%p], receive userbuffer[%p], count[%llu]",
                myRank, i, tempInputMem, tempOutputMem, checkerOpParam.DataDes.count);
        }
        opParam.BatchSendRecvDataDes.itemNum = checkerOpParam.allRanksSendRecvInfoVec[myRank].size();
        // TODO: 需要添加转换函数
        opParam.BatchSendRecvDataDes.sendRecvItemsPtr = (HcclSendRecvItem*)checkerOpParam.allRanksSendRecvInfoVec[myRank].data();
    } else if (g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V ||
        g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        opParam.VDataDes.counts = static_cast<void *>(checkerOpParam.VDataDes.counts.data());
        opParam.VDataDes.displs = static_cast<void *>(checkerOpParam.VDataDes.displs.data());
        opParam.VDataDes.dataType = g_CheckerDataType2HcclDataType[checkerOpParam.VDataDes.dataType];
    } else if (g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
        opParam.BatchWriteDataDes.itemNum = checkerOpParam.BatchWriteDataDes.itemNum;
        opParam.BatchWriteDataDes.queueNum = checkerOpParam.BatchWriteDataDes.queueNum;
        opParam.BatchWriteDataDes.queueIdx = checkerOpParam.BatchWriteDataDes.queueIdx;
    } else {
        opParam.DataDes.count = checkerOpParam.DataDes.count;
        opParam.DataDes.dataType = g_CheckerDataType2HcclDataType[checkerOpParam.DataDes.dataType];
    }

    opParam.root = checkerOpParam.root;

    if (initStream) {
        Stream stream;
        stream.isMainStream_ = true;
        stream.stream_ = (void*)StreamAddrRecorder::Global()->streamAddr++;
        stream.streamId_ = 0;
        opParam.stream = stream;
    }

    opParam.tag = checkerOpParam.tag;
    opParam.aicpuUnfoldMode = checkerOpParam.aicpuUnfoldMode;

    return;
}

HcclResult GenRankTable(hccl::RankTable_t &rankTable, TopoMeta topoMate)
{
    u32 currentRankId = 0;        // rankNum 及 rankid 计数器
    u32 boxIpStart = 168430090;   // 超节点 起始IP(主机序)
    u32 devIpStart = 3232238090;  // 设备 起始 IP (主机序)
    u32 superPodId_ = 0;          // 超级节点 superPodId
    u32 serverIdx_ = 0;           // 服务器下标编号
    // Box 遍历
    std::vector<SuperPodMeta>::iterator topomate_item_begin = topoMate.begin();
    std::vector<SuperPodMeta>::iterator topomate_item_end = topoMate.end();
    for (; topomate_item_begin != topomate_item_end; topomate_item_begin++) {
        u32 superDeviceId_in_pod = 0;  // 超节点内 superDeviceId 计数器

        // Server 遍历
        std::vector<ServerMeta>::iterator ServerMate_item_begin = (*topomate_item_begin).begin();
        std::vector<ServerMeta>::iterator ServerMate_item_end = (*topomate_item_begin).end();
        for (; ServerMate_item_begin != ServerMate_item_end; ServerMate_item_begin++) {
            HcclIpAddress newServerIp(htonl(boxIpStart++));  // 当前服务器的的 IP 指派
            // device 遍历
            std::vector<PhyDeviceId>::iterator device_item_begin = (*ServerMate_item_begin).begin();
            std::vector<PhyDeviceId>::iterator device_item_end = (*ServerMate_item_begin).end();
            for (; device_item_begin != device_item_end; device_item_begin++) {
                // *device_item_begin 是每个 devicePhdId
                HcclIpAddress newDeviceIp(htonl(devIpStart++));             // 当前 device 的 IP 指派
                RankInfo_t temp_rankInfo;                                   // 临时 rankInfo
                temp_rankInfo.rankId = currentRankId++;                     // rankId 自增 1
                temp_rankInfo.hostIp = newServerIp;                         // serverIp 对象
                temp_rankInfo.serverId = newServerIp.GetReadableIP();       // serverIP 的点分十进制
                temp_rankInfo.superPodId = std::to_string(superPodId_);     // 超节点ID
                temp_rankInfo.superPodIdx = superPodId_;                    // 超节点ID
                temp_rankInfo.superDeviceId = superDeviceId_in_pod++;       // 超节点内  deviceID
                temp_rankInfo.deviceInfo.devicePhyId = *device_item_begin;  // 服务器内  device 标识
                temp_rankInfo.deviceInfo.deviceIp.push_back(newDeviceIp);   // 服务器内  deviceIP
                temp_rankInfo.serverIdx = serverIdx_;                       // 服务器下标编号
                rankTable.rankList.push_back(temp_rankInfo);                // 将临时 rankInfo 插入到rankTable 中
            }
            devIpStart += 256;  // 更新下一个服务器(刀片)下的设备起始 IP
            serverIdx_++;
            ServerInfo_t temp_serverInfo;
            temp_serverInfo.serverId = newServerIp.GetReadableIP();  // serverId
            rankTable.serverList.push_back(temp_serverInfo);         // 将临时temp_serverInfo 插入 serverList
        }
        superPodId_++;
        boxIpStart += 256;  // 更新下一个 SupePod 超级节点的起始 IP
    }
    rankTable.serverNum = rankTable.serverList.size();  // 刀片数 server 数
    rankTable.deviceNum = rankTable.rankList.size();    // device 总数
    rankTable.rankNum =
        (currentRankId == rankTable.rankList.size()) ? rankTable.rankList.size() : currentRankId;  // rank 总数
    rankTable.superPodNum = topoMate.size();                     // 超级节点数 superPodNum
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;  // Device 网卡挂载位置

    return HCCL_SUCCESS;
}

HcclResult OrchestraTask(CheckerOpParam &checkerOpParam, RankTable_t &rankTable, u32 rankNum, bool isRunning,
    vector<shared_ptr<hccl::HcclCommunicator>> &communicators, bool isIOSameAddr)
{
    for (RankId myRank = 0; myRank < rankNum; myRank++) {
        RankInfoRecorder::Global()->SetRankId(myRank);
        CheckerDevType curDevType = DeviceInfoRecorder::Global()->rankId2devType[myRank];
        if (curDevType == CheckerDevType::DEV_TYPE_NOSOC) {
            curDevType = checkerOpParam.devtype;
        }
        RankInfoRecorder::Global()->SetDevType(curDevType);

        if (checkerOpParam.opMode == CheckerOpMode::OPBASE) {
            CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
        } else if (checkerOpParam.opMode == CheckerOpMode::OFFLOAD) {
            CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
        } else {
            HCCL_ERROR("unsupported opMode %d", checkerOpParam.opMode);
            return HCCL_E_NOT_SUPPORT;
        }

        HcclCommParams params;
        InitCommParams(params, rankTable, myRank);
        if (!isRunning) {
            shared_ptr<hccl::HcclCommunicator> communicator = make_shared<hccl::HcclCommunicator>();
            CHK_RET(communicator->Init(params, rankTable));
            communicator->SetAlgOpContext(checkerOpParam.algOpContext);
            communicators.push_back(communicator);
        }

        OpParam opParam;
        InitOpParam(opParam, checkerOpParam, myRank, rankNum, true, isIOSameAddr);
        CHK_RET(communicators[myRank]->ExecOp(opParam.opType, opParam, isRunning, checkerOpParam.algName,
            checkerOpParam.aiCoreLimit));
        if (checkerOpParam.opType == CheckerOpType::BATCH_SEND_RECV) {
            CHK_RET(communicators[myRank]->ExecOp(opParam.opType, opParam, isRunning, checkerOpParam.algName,
                checkerOpParam.aiCoreLimit));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hccl