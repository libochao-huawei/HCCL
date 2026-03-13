#include "base.h"
#include "checker.h"
#include "communicator_stub.h"

#include "hccl_common.h"
#include "rank_info_recorder.h"
#include "transport_pub.h"
#include "stream_pub.h"
#include "device_capacity.h"
#include "config.h"
#include "mem_layout.h"
#include "utils_stub.h"
#include "check_utils.h"
#include "link_type_recorder.h"
#include "transport.h"
#include "transformer.h"
#include "orchestrate.h"
#include "hccl_aiv.h"

using namespace std;

namespace hccl {
std::string g_algName;
static std::mutex g_hcomInitMutex;

std::map<LinkTypeInServer, LinkType> LinkTypeInServer2LinkType = {
    {LinkTypeInServer::HCCS_TYPE, LinkType::LINK_HCCS},
    {LinkTypeInServer::PXI_TYPE, LinkType::LINK_PCIE},
    {LinkTypeInServer::SIO_TYPE, LinkType::LINK_SIO},
    {LinkTypeInServer::HCCS_SW_TYPE, LinkType::LINK_HCCS_SW},
    {LinkTypeInServer::RESERVED_LINK_TYPE, LinkType::LINK_RESERVED}};

const std::unordered_set<std::string> g_aiv_rdma_executors = {
    "AllReduceMidCountAivRdmaExecutor",
    "AllReduceSmallCountAivRdmaExecutor"
};

void CalcInputOutputSize(const OpParam &opParam, u32 ranksize, u64 &inputSize, u64 &outputSize, RankId myRank)
{
    u32 unitSize = 0;
    if (!IsAllToAllSeries(g_HcclCMDType2CheckerOpType[opParam.opType]) && opParam.opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV &&
        opParam.opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V &&
        opParam.opType != HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        unitSize = SIZE_TABLE[opParam.DataDes.dataType];
    }

    u64 count = opParam.DataDes.count;
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_SEND) {
        inputSize = count * unitSize;
        outputSize = 0;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        inputSize = 0;
        outputSize = count * unitSize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_REDUCE) {
        if (myRank == opParam.root) {
            outputSize = count * unitSize;
        } else {
            // 当前代码中非root节点还是会用到OUTPUT内存块
            outputSize = count * unitSize;
        }
        inputSize = count * unitSize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        inputSize = count * unitSize;
        outputSize = count * unitSize * ranksize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        inputSize = count * unitSize * ranksize;
        outputSize = count * unitSize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        u64 curSendOffset = 0;
        u64 curRecvOffset = 0;
        void *sendCountMatrix = opParam.All2AllDataDes.sendCountMatrix;
        // 对于AllToAllV/AllToAllVC来说，当前checker还不支持不均匀的数据收发，每个rank收发的数据量是一样的，
        // 所以这边以rank0来计算即可
        RankId curRank = 0;
        // sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < ranksize; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + curRank * ranksize + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[opParam.All2AllDataDes.sendType];
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + curRank + ranksize * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[opParam.All2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        void* sendCounts = opParam.All2AllDataDes.sendCounts;
        void* recvCounts = opParam.All2AllDataDes.recvCounts;

        u64 curSendOffset = 0;
        u64 curRecvOffset = 0;
        for (u32 i = 0; i < ranksize; i++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCounts) + i);
            u64 curSendLength = curSendCounts * SIZE_TABLE[opParam.All2AllDataDes.sendType];
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(recvCounts) + i);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[opParam.All2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_SCATTER) {
        inputSize = count * unitSize * ranksize;
        outputSize = count * unitSize;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        if (opParam.BatchSendRecvDataDes.sendRecvItemsPtr == nullptr) {
            HCCL_ERROR("BatchSendRecv task ItemsPtr is nullptr.");
            return;
        }
        u32 unitSizePerTask = SIZE_TABLE[opParam.BatchSendRecvDataDes.sendRecvItemsPtr->dataType];
        u64 countPerTask = opParam.BatchSendRecvDataDes.sendRecvItemsPtr->count;
        inputSize = ranksize * countPerTask * unitSizePerTask;
        outputSize = ranksize * countPerTask * unitSizePerTask;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        void* counts = opParam.VDataDes.counts;
        inputSize = 0;
        for (u32 i = 0; i < ranksize; i++) {
            u64 curCounts = *(static_cast<const u64 *>(counts) + i);
            u64 curLength = curCounts * SIZE_TABLE[opParam.VDataDes.dataType];
            inputSize += curLength;
        }
        outputSize = static_cast<const u64 *>(counts)[myRank] * SIZE_TABLE[opParam.VDataDes.dataType];
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        void* counts = opParam.VDataDes.counts;
        outputSize = 0;
        for (u32 i = 0; i < ranksize; i++) {
            u64 curCounts = *(static_cast<const u64 *>(counts) + i);
            u64 curLength = curCounts * SIZE_TABLE[opParam.VDataDes.dataType];
            outputSize += curLength;
        }
        inputSize = static_cast<const u64 *>(counts)[myRank] * SIZE_TABLE[opParam.VDataDes.dataType];
    }
    return;
}

HcclResult HcclCommunicator::CreateNotifies(u32 notifyNum, vector<shared_ptr<LocalNotify>> &NotifysM2S,
    vector<shared_ptr<LocalNotify>> &NotifysS2M)
{
    u32 signalNum = notifyNum >> 1;

    for (u32 i = 0; i < signalNum; i++) {
        NotifysM2S.emplace_back(new LocalNotify);
        NotifysM2S[i]->SetNotifyId(2 * i);

        NotifysS2M.emplace_back(new LocalNotify);
        NotifysS2M[i]->SetNotifyId(2 * i + 1);
    }
    return HCCL_SUCCESS;
}

LinkType HcclCommunicator::GetLinkType(TransportType transportType, u32 localRank, u32 remoteRank)
{
    u32 localphyId = RankInfoRecorder::Global()->rankId2phyId[localRank];
    u32 remotephyId = RankInfoRecorder::Global()->rankId2phyId[remoteRank];
    u32 localServerId = RankInfoRecorder::Global()->rankId2serverId[localRank];
    u32 remoteServerId = RankInfoRecorder::Global()->rankId2serverId[remoteRank];
    u32 localSuperPodId = RankInfoRecorder::Global()->rankId2superpodId[localRank];
    u32 remoteSuperPodId = RankInfoRecorder::Global()->rankId2superpodId[remoteRank];

    LinkType linkType = LinkType::LINK_RESERVED;

    if (transportType == TransportType::TRANS_TYPE_IBV_EXP) {
        linkType = LinkType::LINK_ROCE;
    } else {
        linkType = LinkType::LINK_HCCS;
    }

    return linkType;
}

HcclResult HcclCommunicator::CreateTransport(OpCommTransport &algResRequest, RankId rankId, OpCommTransport &algRespond,
    const bool &isZeroCopy, const HcclCMDType &opType)
{
    MachinePara machine;
    TransportPara transportPara;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    for (u32 levelIdx = 0; levelIdx < algResRequest.size(); levelIdx++) {
        for (auto &singleSubCommTransport : algResRequest[levelIdx]) {
            machine.supportDataReceivedAck = singleSubCommTransport.supportDataReceivedAck;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                TransportType transportType;
                if (transportRequest.isUsedRdma) {
                    transportType = TransportType::TRANS_TYPE_IBV_EXP;
                } else {
                    transportType = TransportType::TRANS_TYPE_P2P;
                }

                if (devType == DevType::DEV_TYPE_910_93 && levelIdx == COMM_LEVEL0 && isZeroCopy) {
                    // 如果是零拷贝场景下level0通信域交换零拷贝的共享内存
                    if (transportRequest.inputMemType != TransportMemType::RESERVED) {
                        transportRequest.inputMemType = TransportMemType::PARAM_INPUT;
                    }
                    if (transportRequest.outputMemType != TransportMemType::RESERVED) {
                        transportRequest.outputMemType = opType == HcclCMDType::HCCL_CMD_BROADCAST ? TransportMemType::PARAM_INPUT : TransportMemType::PARAM_OUTPUT;
                    }
                }

                machine.localUserrank = transportRequest.localUserRank;
                machine.remoteUserrank = transportRequest.remoteUserRank;
                machine.notifyNum = transportRequest.notifyNum;
                LinkType linkType = GetLinkType(transportType, transportRequest.localUserRank, transportRequest.remoteUserRank);

                auto iterTransportType = CreatedLinksDict_.find(transportType);
                if (iterTransportType != CreatedLinksDict_.end()
                        && iterTransportType->second.find(transportRequest.localUserRank) != iterTransportType->second.end()
                        && iterTransportType->second[transportRequest.localUserRank].find(transportRequest.remoteUserRank) != iterTransportType->second[transportRequest.localUserRank].end()) {
                    if (transportRequest.isValid) {
                        std::shared_ptr<Transport> link = iterTransportType->second[transportRequest.localUserRank][transportRequest.remoteUserRank];
                        singleSubCommTransport.links.push_back(link);
                    } else {
                        singleSubCommTransport.links.push_back(nullptr);
                    }
                    continue;
                }

                std::shared_ptr<Transport> link(new Transport(transportType, transportPara, machine, linkType));
                CreatedLinksDict_[transportType][transportRequest.localUserRank][transportRequest.remoteUserRank] = link;
                if (transportRequest.isValid) {
                    singleSubCommTransport.links.push_back(link);

                    shared_ptr<TransportCompared> transportCompared(new TransportCompared());
                    transportCompared->isValid = transportRequest.isValid;
                    transportCompared->localRank = transportRequest.localUserRank;
                    transportCompared->remoteRank = transportRequest.remoteUserRank;
                    transportCompared->isCompared = false;
                    transportCompared->inputMemType = transportRequest.inputMemType;
                    transportCompared->outputMemType = transportRequest.outputMemType;

                    links2TransportCompare_.insert(make_pair(link.get(), transportCompared));
                    AllTransport_[rankId].push_back(transportCompared);
                } else {
                    singleSubCommTransport.links.push_back(nullptr);
                }
            }
        }
    }
    algRespond = algResRequest;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateStream(u32 streamNum, vector<Stream>& streams)
{
    // 因为0号streamId给主流用了，这边streamId从1开始编号
    s32 streamCount = 1;
    for (u32 i = 0; i < streamNum; i++) {
        Stream stream;
        stream.stream_ = (void*)StreamAddrRecorder::Global()->streamAddr++;
        stream.streamId_ = streamCount++;
        streams.emplace_back(stream);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RefreshMemLayoutAndGetMemResponse(const OpParam &opParam, AlgResourceRequest &resRequest,
    AlgResourceResponse &algResResponse, RankId rankId)
{
    u64 inputCCLSize = cclBufferManager_.GetInCCLbufferSize();
    u64 outputCCLSize = cclBufferManager_.GetOutCCLbufferSize();

    u32 ranksize = RankInfoRecorder::Global()->rankSize_;

    u32 superPodId = RankInfoRecorder::Global()->rankId2superpodId[rankId];
    u32 serverId = RankInfoRecorder::Global()->rankId2serverId[rankId];
    u32 phyrankId = RankInfoRecorder::Global()->rankId2phyId[rankId];

    void* inputCCLMemPtr = MemLayout::Global()->allSuperPodLayout[superPodId][serverId][phyrankId][BufferType::INPUT_CCL].startAddr;
    void* outputCCLMemPtr = MemLayout::Global()->allSuperPodLayout[superPodId][serverId][phyrankId][BufferType::OUTPUT_CCL].startAddr;
    void* scracthMemPtr = MemLayout::Global()->allSuperPodLayout[superPodId][serverId][phyrankId][BufferType::SCRATCH].startAddr;
    void* inputAIVMemPtr = MemLayout::Global()->allSuperPodLayout[superPodId][serverId][phyrankId][BufferType::INPUT_AIV].startAddr;
    void* outputAIVMemPtr = MemLayout::Global()->allSuperPodLayout[superPodId][serverId][phyrankId][BufferType::OUTPUT_AIV].startAddr;
    void* aivCommInfoMemPtr = MemLayout::Global()->allSuperPodLayout[superPodId][serverId][phyrankId][BufferType::AIV_COMMINFO].startAddr;

    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(opParam, ranksize, inputSize, outputSize, rankId);

    algResResponse.paramInputMem = DeviceMem::create(opParam.inputPtr, inputSize);
    algResResponse.paramOutputMem = DeviceMem::create(opParam.outputPtr, outputSize);
    algResResponse.cclInputMem = DeviceMem::create(inputCCLMemPtr, inputCCLSize);
    algResResponse.cclOutputMem = DeviceMem::create(outputCCLMemPtr, outputCCLSize);
    algResResponse.scratchMem = DeviceMem::create(scracthMemPtr, resRequest.scratchMemSize);
    if (AIV_COMM_BUFFER_BITMASK & resRequest.aivBufferRequest) {
        algResResponse.aivInputMem = DeviceMem::create(inputAIVMemPtr, AIV_DATA_SIZE);
        algResResponse.aivOutputMem = DeviceMem::create(outputAIVMemPtr, AIV_FLAG_SIZE);
    }
    if (AIV_COMM_INFO_BUFFER_BITMASK & resRequest.aivBufferRequest) {
        algResResponse.aivCommInfoMem = DeviceMem::create(aivCommInfoMemPtr, AIV_COMM_INFO_SIZE);
        MemLayout::Global()->MemAlloc((u64)aivCommInfoMemPtr, AIV_COMM_INFO_SIZE);
    }

    CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::INPUT,(char_t*)opParam.inputPtr, inputSize));
    if (opParam.inputPtr != opParam.outputPtr) {
        CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::OUTPUT, (char_t*)opParam.outputPtr, outputSize));
    }
    CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::INPUT_CCL, (char_t*)inputCCLMemPtr, inputCCLSize));
    CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::OUTPUT_CCL, (char_t*)outputCCLMemPtr, outputCCLSize));
    CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::SCRATCH, (char_t*)scracthMemPtr, resRequest.scratchMemSize));
    if (AIV_COMM_BUFFER_BITMASK & resRequest.aivBufferRequest) {
        CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::INPUT_AIV, (char_t*)inputAIVMemPtr, AIV_DATA_SIZE));
        CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::OUTPUT_AIV, (char_t*)outputAIVMemPtr, AIV_FLAG_SIZE));
    }
    if (AIV_COMM_INFO_BUFFER_BITMASK & resRequest.aivBufferRequest) {
        CHK_RET(MemLayout::Global()->SetBufferAddrAndLen(BufferType::AIV_COMMINFO, (char_t*)aivCommInfoMemPtr, AIV_COMM_INFO_SIZE));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgResource(const std::string &newTag, const OpParam &opParam,
    AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
{
    // 校验streamNum、noitfyNum、scratchMemSize是否为异常值
    u32 u32ShiftNum = 31;
    if ((resRequest.streamNum >> u32ShiftNum) & 1) {
        HCCL_ERROR("Invalid stream num %u", resRequest.streamNum);
        return HCCL_E_INTERNAL;
    }

    if ((resRequest.notifyNum >> u32ShiftNum) & 1) {
        HCCL_ERROR("Invalid notify num %u", resRequest.notifyNum);
        return HCCL_E_INTERNAL;
    }

    u32 u64ShiftNum = 63;
    if ((resRequest.scratchMemSize >> u64ShiftNum) & 1) {
        HCCL_ERROR("Invalid scratch mem size %llu", resRequest.scratchMemSize);
        return HCCL_E_INTERNAL;
    }

    RankId rankId = RankInfoRecorder::Global()->GetRankId();
    CHK_RET(CreateStream(resRequest.streamNum, algResResponse.slaveStreams));
    CHK_RET(CreateNotifies(resRequest.notifyNum, algResResponse.notifiesMain, algResResponse.notifiesAux));
    CHK_RET(CreateTransport(resRequest.opTransport, rankId, algResResponse.opTransportResponse, opParam.isZeroCopy, opParam.opType));
    CHK_RET(RefreshMemLayoutAndGetMemResponse(opParam, resRequest, algResResponse, rankId));

    return HCCL_SUCCESS;
}

HcclCommunicator::HcclCommunicator()
    : dispatcher_(nullptr), vDispatcher_(nullptr), userRank_(INVALID_VALUE_RANKID), realUserRank_(INVALID_VALUE_RANKID),
      userRankSize_(INVALID_VALUE_RANKSIZE), deviceLogicId_(-1), hcomGroupNicInit_(false),
      deviceType_(DevType::DEV_TYPE_COUNT), commHandle_(nullptr), commWorkMode_(WorkMode::HCCL_MODE_NORMAL),
      meshAggregationRankSize_(0), ranktableCrc_(0), profilingInitiated_(false), cclBufferManager_(CCLBufferManager()),
      devicePhyId_(INVALID_UINT)
{ }

HcclCommunicator::~HcclCommunicator()
{
    HCCL_DEBUG("Enter ~HcclCommunicator");
    if (implAlg_ != nullptr) {
        delete implAlg_;
        implAlg_ = nullptr;
    }

    resMap_.clear();


    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    if (vDispatcher_ != nullptr) {
        HcclDispatcherDestroy(vDispatcher_);
        vDispatcher_ = nullptr;
    }

    HCCL_DEBUG("~HcclCommunicator success");
}

HcclResult HcclCommunicator::InitCommParams(HcclCommParams &params)
{
#ifndef CCL_KERNEL_AICPU
    commHandle_ = params.commHandle;
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    profilingOption_ = params.profilingOption;
    profilingInitiated_ = params.profilingInitiated;
    deviceType_ = params.deviceType;
    commWorkMode_ = params.commWorkMode;
    hcomGroupNicInit_ = params.hcomGroupNicInit;
    identifier_ = params.identifier;
    collectiveId_ = params.id.internal;
    ranktableCrc_ = params.ranktableCrc;
    commConnections_ = params.commConnections;

    HCCL_DEBUG(
        " userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
        userRank_,
        realUserRank_,
        userRankSize_,
        deviceLogicId_,
        deviceType_,
        commWorkMode_);
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRankInfo(const RankTable_t &rankTable)
{
    deviceLogicId_ = attrCollector_.GetDeviceLogicId();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDispatcher()
{
    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devicePhyId_, &dispatcher_));
    CHK_SMART_PTR_NULL(dispatcher_);

    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_VIRTURAL, devicePhyId_, &vDispatcher_));
    CHK_SMART_PTR_NULL(vDispatcher_);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitPara()
{
#ifndef CCL_KERNEL_AICPU
    // 检查当前user_rank对应的devid和rt查到的一致
    CHK_RET(attrCollector_.CheckLocalRankInfo());
    CHK_RET(attrCollector_.CalAndSetMeshAggRankSize());
    meshAggregationRankSize_ = attrCollector_.GetMeshAggregationRankSize();

    CHK_RET(InitDispatcher());

    HcclTopoAttr topoAttr{};
    attrCollector_.GetTopoAttr(topoAttr);

    HcclAlgoAttr algoAttr{};
    attrCollector_.GetAlgoAttr(algoAttr);

    implAlg_ = new (std::nothrow) HcclAlg(cclBufferManager_, dispatcher_, vDispatcher_);
    CHK_SMART_PTR_NULL(implAlg_);

    CHK_RET(implAlg_->Init(algoAttr, topoAttr));

#endif
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Init(HcclCommParams &params, const RankTable_t &rankTable)
{
#ifndef CCL_KERNEL_AICPU
    CHK_RET(InitCommParams(params));
    CHK_RET(attrCollector_.Init(params, rankTable));
    CHK_RET(InitRankInfo(rankTable));

/*--------------加锁区--------------*/
    std::unique_lock<std::mutex> lock(g_hcomInitMutex);

    attrCollector_.GenCollectiveId(params, rankTable);
    collectiveId_ = attrCollector_.GetCollectiveId();

    // 初始化参数(需要放置在ranktable解析之后)
    HcclResult ret = InitPara();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][Init]errNo[0x%016llx] collectiveid[%s] parameter initialization failed",
        HCCL_ERROR_CODE(ret), params.id.internal), ret);
    lock.unlock();
/*--------------加锁区--------------*/
    cclBufferManager_.CreateCommCCLbuffer();

#endif
    return HCCL_SUCCESS;
}

void HcclCommunicator::GenAllGatherResultForAllToAllV(OpParam &opParam, void* result)
{
    std::vector<u64> vctSendLength(userRankSize_, 0);
    std::vector<u64> vctSendOffset(userRankSize_, 0);
    std::vector<u64> vctRecvLength(userRankSize_, 0);
    std::vector<u64> vctRecvOffset(userRankSize_, 0);

    HcclDataType sendType = opParam.All2AllDataDes.sendType;
    HcclDataType recvType = opParam.All2AllDataDes.recvType;
    void* sendCounts = opParam.All2AllDataDes.sendCounts;
    void* sdispls = opParam.All2AllDataDes.sdispls;
    void* recvCounts = opParam.All2AllDataDes.recvCounts;
    void* rdispls = opParam.All2AllDataDes.rdispls;

    for (u32 i = 0; i < userRankSize_; i++) {
        vctSendLength[i] = *(static_cast<const u64 *>(sendCounts) + i) * SIZE_TABLE[sendType];
        vctSendOffset[i] = *(static_cast<const u64 *>(sdispls) + i) * SIZE_TABLE[sendType];
        vctRecvLength[i] = *(static_cast<const u64 *>(recvCounts) + i) * SIZE_TABLE[recvType];
        vctRecvOffset[i] = *(static_cast<const u64 *>(rdispls) + i) * SIZE_TABLE[recvType];
    }

    u64 stepSize = (u64)sizeof(u64) * userRankSize_;

    for (u32 i = 0; i < userRankSize_; i++) {
        memcpy_s(static_cast<u8 *>(result) + stepSize * 4 * i,
            stepSize, vctSendLength.data(), stepSize);
        memcpy_s(static_cast<u8 *>(result) + stepSize * 4 * i + stepSize,
            stepSize, vctSendOffset.data(), stepSize);
        memcpy_s(static_cast<u8 *>(result) + stepSize * 4 * i + stepSize * 2,
            stepSize, vctRecvLength.data(), stepSize);
        memcpy_s(static_cast<u8 *>(result) + stepSize * 4 * i + stepSize * 3,
            stepSize, vctRecvOffset.data(), stepSize);
    }

    return;
}

HcclResult HcclCommunicator::SetAlgOpContext(AlgOpContext algOpContext)
{
    algOpContext_ = algOpContext;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAivTag(std::string algName, bool isCapture, s32 &aivTag)
{
    s32 aivTagNum = 1;
    if (g_aiv_rdma_executors.find(algName) != g_aiv_rdma_executors.end()) {
        aivTagNum = 2;
    }
    
    bool useOpbaseFlag = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isCapture);
    if (useOpbaseFlag) {
        aivTag = aivOpbaseTag_;
        aivOpbaseTag_ = GetNextAivTag(aivOpbaseTag_, aivTagNum);
    } else {
        aivTag = aivOffloadTag_;
        aivOffloadTag_ = GetNextAivTag(aivOffloadTag_, aivTagNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOp(HcclCMDType opType, OpParam &opParam, bool isRunning, string givenAlgName, u32 aiCoreLimit)
{
#ifndef CCL_KERNEL_AICPU
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);

    if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
        u64 allGatherRetSize = (u64)sizeof(u64) * userRankSize_ * 4 * userRankSize_;
        HostMem hostCollectBuffer = HostMem::alloc(allGatherRetSize);
        CHK_PTR_NULL(hostCollectBuffer.ptr());
        GenAllGatherResultForAllToAllV(opParam, hostCollectBuffer.ptr());
        // checker中需要模拟allgather的结果，并set给alltoAllOperator算子
        alltoAllOperator->SetPreProcessResult(std::move(hostCollectBuffer));
    }

    // 算法选择
    std::string algName;
    std::string newTag;

    if (givenAlgName.empty()) {
        ResourceLimit limit;
        limit.ifLimit = aiCoreLimit > 0;
        limit.aivCoreLimit = aiCoreLimit;
        AlgDesc algDesc;
        CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, limit, algName, algDesc, newTag));
        if (algDesc.isZeroCopy) {
            opParam.isZeroCopy = true;
        }
    } else {
        algName = givenAlgName;
        newTag = opParam.tag + algName;
    }
    g_algName = algName;
    algOperator->SetAlgOpContext(algOpContext_);

    CHK_RET(algOperator->SetAivClearEnable(true));

    bool needIncreLink = false;
    // 资源创建
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, opParam, resRequest, resMap_[newTag]));
    } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        AlgResourceRequest resRequest;
        std::set<u32> ranksLinked;
        CHK_RET(algOperator->CalcIncreLinkRequest(algName, opParam, ranksLinked, resRequest, needIncreLink));
    }

    if (isRunning) {
        if (resMap_[newTag].aivCommInfoMem.ptr() != nullptr) {
            algOperator->PrepareCommInfoToDevice(algName, resMap_[newTag]);
        }
        if (aiCoreLimit > 0 && aiCoreLimit < 48) {
            CHK_RET(algOperator->SetBlockDim(aiCoreLimit));
        }

        GetAivTag(algName, opParam.isCapture, opParam.aivTag);
        CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
    }
    AivSuperKernelArgs aivSuperKernelArgs;
    //std::string name = "ReduceScatterMeshAivExecutor";
    CHK_RET(algOperator->GetAivExecParam(algName, opParam, resMap_[newTag], aivSuperKernelArgs));

    AdjInfo nslbAdjInfo = {};
    CHK_RET(algOperator->GetAdjInfo(algName, opParam, resMap_[newTag], nslbAdjInfo));
#endif
    return HCCL_SUCCESS;
}

}