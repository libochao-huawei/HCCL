/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu transport header file
 * Create: 2025-02-17
 */

#ifndef HCCL_CCU_TRANSPORT_H
#define HCCL_CCU_TRANSPORT_H

#include <mutex>
#include <memory>
#include <vector>
#include <shared_mutex>

#include "socket.h"
#include "op_mode.h"
#include "binary_stream.h"
#include "ccu_connection.h"

namespace Hccl {

class CcuTransport {
public:
    static constexpr uint32_t INIT_CKE_NUM = 16;
    static constexpr uint32_t INIT_XN_NUM  = 16;
    MAKE_ENUM(TransStatus, INIT, SEND_ALL_INFO, RECV_ALL_INFO, SEND_TRANS_RES, RECV_TRANS_RES, SEND_FIN, RECV_FIN,
              RECVING_FIN, RECVING_TRANS_RES, READY, CONNECT_FAILED, SOCKET_TIMEOUT)

    struct CclBufferInfo {
        uint64_t addr{0};
        uint32_t size{0};
        uint32_t tokenId{0};
        uint32_t tokenValue{0};

        explicit CclBufferInfo() = default;
        CclBufferInfo(const uint64_t addr, const uint32_t size,
            const uint32_t tokenId, const uint32_t tokenValue)
            : addr(addr), size(size), tokenId(tokenId), tokenValue(tokenValue) {}

        void Pack(BinaryStream &binaryStream) const {
            binaryStream << addr << size << tokenId << tokenValue;
            HCCL_INFO("Pack Ccl Buffer Info: addr[%llu] size[%u]", addr, size);
        }

        void Unpack(BinaryStream &binaryStream) {
            binaryStream >> addr >> size >> tokenId >> tokenValue;
            HCCL_INFO("Unpack Ccl Buffer Info: addr[%llu] size[%u]", addr, size);
        }
    };

    MAKE_ENUM(CcuConnectionType, UBC_TP, UBC_CTP);
    struct CcuConnectionInfo {
        CcuConnectionType type{CcuConnectionType::UBC_TP};
        IpAddress locAddr{};
        IpAddress rmtAddr{};
        CcuChannelInfo channelInfo{};
        std::vector<CcuJetty *> ccuJettys;

        explicit CcuConnectionInfo() = default;
        CcuConnectionInfo(const CcuConnectionType type,
            const IpAddress &locAddr, const IpAddress &rmtAddr,
            const CcuChannelInfo &channelInfo,
            const std::vector<CcuJetty *> &ccuJettys)
            : type(type), locAddr(locAddr), rmtAddr(rmtAddr),
            channelInfo(channelInfo), ccuJettys(ccuJettys) {}
    };

    CcuTransport(Socket *socket, std::unique_ptr<CcuConnection> &&connection, const CclBufferInfo &locCclBufInfo);
    CcuTransport(const CcuTransport &that)             = delete;
    CcuTransport &operator=(const CcuTransport &other) = delete;
    ~CcuTransport();
    HcclResult AppendRes(uint32_t ckesNum, uint32_t xnsNum);
    HcclResult Init();

    struct Attribution {
        OpMode       opMode;
        u32          devicePhyId;
        std::vector<char> handshakeMsg{0};
        string       Describe() const
        {
            return StringFormat("CcuTransportAttribution[opMode=%s, devicePhyId=%u, handshakeMsg=%s]",
                                opMode.Describe().c_str(), devicePhyId,
                                Bytes2hex(handshakeMsg.data(), handshakeMsg.size()).c_str());
        }
    };

    std::vector<char> &GetRmtHandshakeMsg() // 返回握手消息
    {
        return rmtHandshakeMsg;
    }

    std::vector<char> &GetLocalHandshakeMsg() // 返回握手消息
    {
        return attr.handshakeMsg;
    }

    void SetHandshakeMsg(const std::vector<char> &handshakeMsg)
    {
        attr.handshakeMsg = handshakeMsg;
    }

    CcuConnection *GetCcuConnection() const
    {
        return ccuConnection.get();
    }

    // 下面接口为平台层接口，不能在框架层使用
    uint32_t    GetDieId() const;
    uint32_t    GetChannelId() const;
    void        SetCntCke(const std::vector<uint32_t> &cntCke);
    uint32_t    GetLocCkeByIndex(uint32_t index) const;
    uint32_t    GetLocCntCkeByIndex(uint32_t index) const;
    uint32_t    GetLocXnByIndex(uint32_t index) const;
    uint32_t    GetRmtCkeByIndex(uint32_t index) const;
    uint32_t    GetRmtCntCkeByIndex(uint32_t index) const;
    uint32_t    GetRmtXnByIndex(uint32_t index) const;
    HcclResult  GetLocBuffer(CclBufferInfo &bufferInfo, const uint32_t &bufNum) const;
    HcclResult  GetRmtBuffer(CclBufferInfo &bufferInfo, const uint32_t &bufNum) const;
    TransStatus GetStatus();
    std::string Describe() const;
    void RecvDataProcess(RankId srcRank, RankId dstRank,
        std::unordered_map<CcuTransport*, vector<char>>& allSendData);
    void SendConnAndTransInfo(RankId srcRank, RankId dstRank,
        std::unordered_map<CcuTransport*, vector<char>>& allSendData);
    IpAddress GetLocAddr() const;
    IpAddress GetRmtAddr() const;

private:
    // 保存transport中需要使用的cke，xn等ccu资源
    struct TransRes {
        std::vector<uint32_t> ckes;
        std::vector<uint32_t> cntCkes;
        std::vector<uint32_t> xns;
    };
    TransStatus                              StateMachine();
    HcclResult                               AppendCkes(uint32_t ckesNum);
    HcclResult                               AppendXns(uint32_t xnsNum);
    void                                     SendFinish();
    void                                     RecvFinish();
    void                                     CheckFinish();
    void                                     RecvTransInfoProcess();
    void                                     ReleaseTransRes();
    void                                     RecvConnAndTransInfo();
    void                                     SendTransInfo();
    void                                     RecvTransInfo();
    void                                     HandshakeMsgPack(BinaryStream &binaryStream);
    void                                     ConnInfoPack(BinaryStream &binaryStream) const;
    void                                     TransResPack(BinaryStream &binaryStream);
    void                                     CclBufferInfoPack(BinaryStream &binaryStream) const;
    void                                     HandshakeMsgUnpack(BinaryStream &binaryStream);
    void                                     ConnInfoUnpackProc(BinaryStream &binaryStream) const;
    void                                     TransResUnpackProc(BinaryStream &binaryStream);
    void                                     CclBufferInfoUnpack(BinaryStream &binaryStream);
    uint32_t                                 dieId{0};
    int32_t                                  devLogicId{0};
    Attribution                              attr;
    std::vector<char>                        rmtHandshakeMsg{0}; // 远端握手消息
    Socket                                  *socket;
    std::unique_ptr<CcuConnection>           ccuConnection;
    TransRes                                 locRes;
    TransRes                                 rmtRes;
    CcuTransport::TransStatus                transStatus;
    std::vector<std::vector<ResInfo>>        ckesRes;
    std::vector<std::vector<ResInfo>>        xnsRes;
    CclBufferInfo                            locCclBufInfo;
    CclBufferInfo                            rmtCclBufInfo;
    uint32_t                                 exchangeDataSize{0};
    mutable std::shared_timed_mutex          transMutex;
    std::vector<char>                        recvData{};
    std::vector<char>                        recvTrans{};
    std::vector<char>                        sendData{};
    std::vector<char>                        sendTrans{};
    std::vector<char>                        recvFinishMsg{};
    std::vector<char>                        sendFinishMsg{};
};

HcclResult CcuCreateTransport(Socket *socket, const CcuTransport::CcuConnectionInfo &ccuConnectionInfo,
    const CcuTransport::CclBufferInfo &cclBufferInfo, std::unique_ptr<CcuTransport> &ccuTransport);

} // namespace Hccl
#endif // HCCL_CCU_TRANSPORT_H