#include "ccu_transport_manager.h"
#include "ccu_jetty_mgr.h"
#include "coll_service_stub.h"
#include "communicator_impl_stub.h"
#include "rank_info_recorder.h"
#include "mem_layout.h"
#include "transport_utils.h"

namespace Hccl {

CcuTransportMgr::CcuTransportMgr(const CommunicatorImpl &comm, const int32_t devLogicId)
    : comm(&comm), devLogicId_(devLogicId)
{
}


HcclResult CcuTransportMgr::PrepareCreate(const LinkData &link, CcuTransport *&transport)
{
    // 如果ccuLink2TransportMap中存在link对应的transport，则直接返回
    auto linkIter = ccuLink2TransportMap.find(link);
    if (linkIter != ccuLink2TransportMap.end()) {
        transport = linkIter->second.get();
        return HcclResult::HCCL_SUCCESS;
    }

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    MemBlock block = MemLayout::Global()->GetMemBlock(checker::BufferType::SCRATCH, curRank);

    CcuTransport::CclBufferInfo locCclBufInfo;
    locCclBufInfo.addr       = reinterpret_cast<uintptr_t>(block.startAddr);
    locCclBufInfo.size       = block.size;
    locCclBufInfo.tokenId    = curRank; // 先用rankId来充当tokenId
    locCclBufInfo.tokenValue = curRank; // 先用rankId来充当tokenValue

    // socket + connection作为参数传入transport的构造函数
    Socket *socket = nullptr;

    CcuJettyMgr *ccuJettyMgr = dynamic_cast<CollServiceStub *>(comm->GetCollService())
        ->GetCcuInsPreprocessor()->GetCcuComm()->GetCcuJettyMgr();
    const auto channelJettys = ccuJettyMgr->GetChannelJettys(link);
    const CcuChannelInfo &channelInfo = channelJettys.first;
    const std::vector<CcuJetty *> &ccuJettys = channelJettys.second;
    const auto &locAddr = link.GetLocalAddr();
    const auto &rmtAddr = link.GetRemoteAddr();
    CcuTransport::CcuConnectionType type = link.GetLinkProtocol() == LinkProtocol::UB_CTP ?
        CcuTransport::CcuConnectionType::UBC_CTP : CcuTransport::CcuConnectionType::UBC_TP;
    CcuTransport::CcuConnectionInfo connectionInfo{type, locAddr, rmtAddr, channelInfo, ccuJettys};

     // 当前不支持创建非UBC协议的链路
    std::unique_ptr<CcuTransport> transportPtr = nullptr;
    auto ret = CcuCreateTransport(socket, connectionInfo, locCclBufInfo, transportPtr);
    if (ret == HcclResult::HCCL_E_UNAVAIL) {
        HCCL_WARNING("[CcuTransportMgr][%s] failed, some ccu resources are unavaialble, "
            "locAddr[%s] rmtAddr[%s].", __func__, locAddr.Describe().c_str(), rmtAddr.Describe().c_str());
        return ret;
    }
    CHK_RET(ret);

    tempTransport.emplace_back(link); // 插入TempTransport中表明Transport并未真正创建成功，需要等待握手确认
    ccuLink2TransportMap[link] = std::move(transportPtr);
    const auto &rawTransportPtr = ccuLink2TransportMap[link].get();
    g_allRankTransports[link.GetLocalRankId()][link.GetRemoteRankId()].push_back(rawTransportPtr);
    ccuRank2TransportsMap[link.GetRemoteRankId()].insert(rawTransportPtr);

    transport = rawTransportPtr;
    return HcclResult::HCCL_SUCCESS;
}

unordered_map<LinkData, unique_ptr<CcuTransport>>& CcuTransportMgr::GetCcuLink2Transport()
{
    return ccuLink2TransportMap;
}

set<CcuTransport*> CcuTransportMgr::Get(RankId rank)
{
    auto rankIter = ccuRank2TransportsMap.find(rank);
    if (rankIter != ccuRank2TransportsMap.end()) {
        return rankIter->second;
    }
    HCCL_WARNING("[CcuTransportMgr::%s] CcuTransport does not existed, "
                 "errNo[0x%016llx], remoteRank[%d]", __func__,
                 HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), rank);
    return set<CcuTransport*>();
}

CcuTransportMgr::~CcuTransportMgr()
{ }

void CcuTransportMgr::Confirm()
{
    // transport建链
    TransportsConnect();
    // 清空TempTransport
    tempTransport.clear();
}

vector<std::pair<CcuTransport*, LinkData>> CcuTransportMgr::GetUnConfirmedTrans()
{
    if (tempTransport.size() == 0) {
        HCCL_WARNING("[CcuTransportMgr::%s] UnConfirmedTrans does not exist, please check.", __func__);
        return vector<std::pair<CcuTransport*, LinkData>>();
    }

    vector<std::pair<CcuTransport*, LinkData>> unConfirmedTrans;
    for (const auto &linkId : tempTransport) {
        auto iterLink = ccuLink2TransportMap.find(linkId);
        unConfirmedTrans.emplace_back(std::make_pair(iterLink->second.get(), linkId));
    }
    return unConfirmedTrans;
}

std::unordered_map<CcuTransport*, vector<char>> g_ccuTransportSendData;

void CcuTransportMgr::GenSendData()
{
    HcclResult res = HcclResult::HCCL_SUCCESS;
    vector<std::pair<CcuTransport*, LinkData>> transLinkPairs = GetUnConfirmedTrans();
    auto op = comm->GetCurrentCollOperator();
    for (auto &pair : transLinkPairs) {
        auto transport = pair.first;
        transport->SetHandshakeMsg(op->GetUniqueId());

        LinkData linkData = pair.second;
        transport->SendConnAndTransInfo(linkData.GetLocalRankId(), linkData.GetRemoteRankId(), g_ccuTransportSendData);
        ccuChannel2RemoteRankMap[transport->GetChannelId()] = pair.second.GetRemoteRankId();
    }
}

void CcuTransportMgr::RecvData()
{
    HcclResult res = HcclResult::HCCL_SUCCESS;
    vector<std::pair<CcuTransport*, LinkData>> transLinkPairs = GetUnConfirmedTrans();
    for (auto &pair : transLinkPairs) {
        auto transport = pair.first;
        LinkData linkData = pair.second;
        transport->RecvDataProcess(linkData.GetRemoteRankId(), linkData.GetLocalRankId(), g_ccuTransportSendData);
    }
}

void CcuTransportMgr::TransportsConnect()
{ }

void CcuTransportMgr::Fallback()
{ }

void CcuTransportMgr::RecoverConfirm()
{
   // transport建链
   // RecoverTransportsConnect();
   // 清空TempTransport
   tempTransport.clear();
}

}