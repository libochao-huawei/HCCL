
#include "ccu_transport.h"
#include "ccu_device_manager.h"
#include "transport_utils.h"

namespace Hccl {

HcclResult CcuCreateTransport(Socket *socket, const CcuTransport::CcuConnectionInfo &ccuConnectionInfo,
    const CcuTransport::CclBufferInfo &cclBufferInfo, std::unique_ptr<CcuTransport> &ccuTransport)
{
    auto ccuConnection = std::make_unique<CcuCtpConnection>(ccuConnectionInfo.locAddr,
            ccuConnectionInfo.rmtAddr, ccuConnectionInfo.channelInfo,
            ccuConnectionInfo.ccuJettys);

    auto ret = ccuConnection->Init();
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    ccuTransport = std::make_unique<CcuTransport>(socket, std::move(ccuConnection), cclBufferInfo);
    ret = ccuTransport->Init();
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

CcuTransport::CcuTransport(Socket *socket, std::unique_ptr<CcuConnection> &&connection,
    const CclBufferInfo &locCclBufInfo)
    : socket(socket), ccuConnection(std::move(connection)), locCclBufInfo(locCclBufInfo)
{
}

HcclResult CcuTransport::Init()
{
    dieId      = ccuConnection->GetDieId();
    devLogicId = ccuConnection->GetDevLogicId();
    auto ret   = AppendCkes(INIT_CKE_NUM);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendCkes failed.", ret);
        return ret;
    }
    ret = AppendXns(INIT_XN_NUM);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendXns failed.", ret);
        return ret;
    }
    transStatus = TransStatus::INIT;
    return HCCL_SUCCESS;
}

uint32_t CcuTransport::GetDieId() const
{
    return dieId;
}

HcclResult CcuTransport::AppendXns(uint32_t xnsNum)
{
    vector<ResInfo> resInfo;
    auto            ret = CcuDeviceManager::AllocXn(devLogicId, dieId, xnsNum, resInfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendXns failed.", ret);
        return ret;
    }

    for (uint32_t i = 0; i < resInfo.size(); i++) {
        uint32_t xnNum     = resInfo[i].num;
        uint32_t xnsSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < xnNum; j++) {
            locRes.xns.push_back(xnsSartId + j);
        }
    }
    xnsRes.push_back(resInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuTransport::AppendCkes(uint32_t ckesNum)
{
    vector<ResInfo> resInfo;
    auto            ret = CcuDeviceManager::AllocCke(devLogicId, dieId, ckesNum, resInfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("errNo[0x%016llx]:AppendCkes failed.", ret);
        return ret;
    }

    for (uint32_t i = 0; i < resInfo.size(); i++) {
        uint32_t ckeNum     = resInfo[i].num;
        uint32_t ckesSartId = resInfo[i].startId;
        for (uint32_t j = 0; j < ckeNum; j++) {
            locRes.ckes.push_back(ckesSartId + j);
        }
    }
    ckesRes.push_back(resInfo);
    return HCCL_SUCCESS;
}

void CcuTransport::HandshakeMsgPack(BinaryStream &binaryStream)
{
    binaryStream << attr.handshakeMsg;
    HCCL_DEBUG("[CcuTransport][%s] start pack handshakeMsg, handshakeMsg=%s", __func__,
                Bytes2hex(attr.handshakeMsg.data(), attr.handshakeMsg.size()).c_str());
}

void CcuTransport::TransResPack(BinaryStream &binaryStream)
{
    uint32_t locCkesSize = locRes.ckes.size();
    binaryStream << locCkesSize;
    for (uint32_t i = 0; i < locRes.ckes.size(); i++) {
        binaryStream << locRes.ckes[i];
    }

    uint32_t locCntCkesSize = locRes.cntCkes.size();
    binaryStream << locCntCkesSize;
    for (uint32_t i = 0; i < locRes.cntCkes.size(); i++) {
        binaryStream << locRes.cntCkes[i];
    }

    uint32_t locXnsSize = locRes.xns.size();
    binaryStream << locXnsSize;
    for (uint32_t i = 0; i < locRes.xns.size(); i++) {
        binaryStream << locRes.xns[i];
    }
    HCCL_DEBUG("Send ckesSize[%u], cntCkesSize[%u], xnsSize[%u]", locCkesSize, locCntCkesSize, locXnsSize);
}

void CcuTransport::CclBufferInfoPack(BinaryStream &binaryStream) const
{
    locCclBufInfo.Pack(binaryStream);
}

void CcuTransport::SendConnAndTransInfo(RankId srcRank, RankId dstRank,
    std::unordered_map<CcuTransport*, vector<char>>& allSendData)
{
    // 推动ccu connection状态机
    BinaryStream binaryStream;
    HandshakeMsgPack(binaryStream);
    TransResPack(binaryStream);
    CclBufferInfoPack(binaryStream);
    binaryStream.Dump(sendData);
    allSendData[this] = sendData;
    exchangeDataSize = sendData.size();
}

void CcuTransport::HandshakeMsgUnpack(BinaryStream &binaryStream)
{
    binaryStream >> rmtHandshakeMsg;
    HCCL_DEBUG("[CcuTransport][%s] start unpack handshakeMsg, rmtHandshakeMsg=%s", __func__,
                Bytes2hex(rmtHandshakeMsg.data(), rmtHandshakeMsg.size()).c_str());
}

void CcuTransport::TransResUnpackProc(BinaryStream &binaryStream)
{
    uint32_t resSzie;
    binaryStream >> resSzie;
    rmtRes.ckes.clear();
    for (uint32_t i = 0; i < resSzie; i++) {
        uint32_t cke;
        binaryStream >> cke;
        rmtRes.ckes.push_back(cke);
    }
    HCCL_DEBUG("Recv ckesSize[%u]", resSzie);

    binaryStream >> resSzie;
    rmtRes.cntCkes.clear();
    for (uint32_t i = 0; i < resSzie; i++) {
        uint32_t cntCke;
        binaryStream >> cntCke;
        rmtRes.cntCkes.push_back(cntCke);
    }
    HCCL_DEBUG("Recv cntCkesSize[%u]", resSzie);

    binaryStream >> resSzie;
    rmtRes.xns.clear();
    for (uint32_t i = 0; i < resSzie; i++) {
        uint32_t xn;
        binaryStream >> xn;
        rmtRes.xns.push_back(xn);
    }
    HCCL_DEBUG("Recv xnsSize[%u]", resSzie);
}

void CcuTransport::CclBufferInfoUnpack(BinaryStream &binaryStream)
{
    rmtCclBufInfo.Unpack(binaryStream);
}

void CcuTransport::RecvDataProcess(RankId srcRank, RankId dstRank,
    std::unordered_map<CcuTransport*, vector<char>>& allSendData)
{
    CcuTransport* peerTransport = g_transportsPair[this];
    recvData = allSendData[peerTransport];
    BinaryStream binaryStream(recvData);
    HandshakeMsgUnpack(binaryStream);
    TransResUnpackProc(binaryStream);
    CclBufferInfoUnpack(binaryStream);
}

void CcuTransport::SetCntCke(const vector<uint32_t> &cntCke)
{
    locRes.cntCkes = cntCke;
}

uint32_t CcuTransport::GetRmtCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= rmtRes.ckes.size()) {
        THROW<InternalException>(
            StringFormat("[GetRmtCkeByIndex]:index[%u] is bigger than ckes size[%u]", index, rmtRes.ckes.size()));
    }
    return rmtRes.ckes[index];
}

uint32_t CcuTransport::GetRmtCntCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= rmtRes.cntCkes.size()) {
        THROW<InternalException>(StringFormat("[GetRmtCntCkeByIndex]:index[%u] is bigger than cntCkes size[%u]", index,
                                              rmtRes.cntCkes.size()));
    }
    return rmtRes.cntCkes[index];
}

uint32_t CcuTransport::GetRmtXnByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= rmtRes.xns.size()) {
        THROW<InternalException>(
            StringFormat("[GetRmtXnByIndex]:index[%u] is bigger than xns size[%u]", index, rmtRes.xns.size()));
    }
    return rmtRes.xns[index];
}

uint32_t CcuTransport::GetLocXnByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= locRes.xns.size()) {
        THROW<InternalException>(
            StringFormat("[GetLocXnByIndex]:index[%u] is bigger than xns size[%u]", index, locRes.xns.size()));
    }
    return locRes.xns[index];
}

uint32_t CcuTransport::GetLocCntCkeByIndex(uint32_t index) const
{
    std::shared_lock<std::shared_timed_mutex> lock(transMutex);
    if (index >= locRes.cntCkes.size()) {
        THROW<InternalException>(StringFormat("[GetLocCntCkeByIndex]:index[%u] is bigger than cntCkes size[%u]", index,
                                              locRes.cntCkes.size()));
    }
    return locRes.cntCkes[index];
}

uint32_t CcuTransport::GetChannelId() const
{
    return ccuConnection->GetChannelId();
}

IpAddress CcuTransport::GetLocAddr() const
{
    return ccuConnection->GetLocAddr();
}

IpAddress CcuTransport::GetRmtAddr() const
{
    return ccuConnection->GetRmtAddr();
}

HcclResult CcuTransport::GetRmtBuffer(CclBufferInfo &bufferInfo, const uint32_t &bufNum) const
{
    (void)bufNum;
    bufferInfo = rmtCclBufInfo;
    return HCCL_SUCCESS;
}

void CcuTransport::ReleaseTransRes()
{
    for (uint32_t i = 0; i < ckesRes.size(); i++) {
        auto ret = CcuDeviceManager::ReleaseCke(devLogicId, dieId, ckesRes[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<InternalException>("errNo[0x%016llx]:Release ckesRes failed.", ret);
        }
    }

    for (uint32_t i = 0; i < xnsRes.size(); i++) {
        auto ret = CcuDeviceManager::ReleaseXn(devLogicId, dieId, xnsRes[i]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<InternalException>("errNo[0x%016llx]:Release xnsRes failed.", ret);
        }
    }
}

CcuTransport::~CcuTransport()
{
    DECTOR_TRY_CATCH("CcuTransport", ReleaseTransRes());
}

}

