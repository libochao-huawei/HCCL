/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu connection header file
 * Create: 2025-02-17
 */

#ifndef HCCL_CCU_CONNECTION_H
#define HCCL_CCU_CONNECTION_H

#include "ccu_jetty.h"
#include "tp_manager.h"
#include "orion_adapter_hccp.h"
#include "ccu_device_manager.h"

namespace Hccl {

MAKE_ENUM(CcuConnStatus,
    INIT,           // 初始化
    EXCHANGEABLE,   // 可与对端交换
    CONNECTED,      // 建链完成
    CONN_INVALID);  // 链接错误

class CcuConnection {
public:
    CcuConnection(const IpAddress &locAddr, const IpAddress &rmtAddr,
        const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys);
    CcuConnection(const CcuConnection &that)             = delete;
    CcuConnection &operator=(const CcuConnection &other) = delete;
    ~CcuConnection();

    std::vector<CcuJetty *> GetCcuJettys() const
    {
        return ccuJettys_;
    }

    // 用于建链过程CcuTransport调用
    HcclResult    Init();
    CcuConnStatus GetStatus();
    void          Serialize(std::vector<char> &dtoData);
    void          Deserialize(const std::vector<char> &dtoData);
    void          ImportJetty();

    uint32_t GetChannelId() const;
    uint32_t GetDieId() const;
    int32_t  GetDevLogicId() const;
    IpAddress GetLocAddr();
    IpAddress GetRmtAddr();

protected:
    TpProtocol tpProtocol{TpProtocol::INVALID};

private:
    MAKE_ENUM(InnerStatus,
        INIT, JETTY_CREATING, TP_INFO_GETTING,
        EXCHANGEABLE, JETTY_IMPORTING,
        CONNECTED,
        CONN_INVALID);

    struct ImportJettyCtx {
        // 保存对端的key，传递数据结构时不需要整个copy
        u8 remoteQpKey[HRT_UB_QP_KEY_MAX_LEN]{0};
        HrtRaUbJettyImportedInParam  inParam{};
        HrtRaUbJettyImportedOutParam outParam{};
    };

    CcuConnStatus status{CcuConnStatus::CONN_INVALID};
    InnerStatus   innerStatus{InnerStatus::CONN_INVALID};
    bool          isJettyCreated{false};
    bool          isJettyImported{false};

    IpAddress               locAddr_{};
    IpAddress               rmtAddr_{};
    CcuChannelInfo          channelInfo_{};
    std::vector<CcuJetty *> ccuJettys_;

    int32_t       devLogicId{0};
    uint32_t      dieId{0};
    uint32_t      funcId{0};
    RdmaHandle    rdmaHandle{nullptr};
    uint32_t      jettyNum{0};

    // 通过ccu comp 获取 ccu buffer信息
    uint64_t      ccuBufAddr{0};
    uint32_t      ccuBufTokenId{0};
    uint32_t      ccuBufTokenValue{0};

    vector<ImportJettyCtx> importJettyCtxs;  // 记录import jetty相关信息
    JettyImportCfg         jettyImportCfg{}; // import配置信息，因复用TpHandle只需一份

    // 交换后获取对端ccu buffer信息
    uint64_t      rmtCcuBufAddr{0};
    uint32_t      rmtCcuBufTokenId{0};
    uint32_t      rmtCcuBufTokenValue{0};

    // 感知tp获取tp handle，import jetty后urma提供tpn
    TpInfo   tpInfo{};

    // 异步import上下文信息
    vector<RequestHandle>  reqHandles;
    vector<vector<char_t>> reqDataBuffers;
    vector<void*>          remoteJettyHandlePtrs;

    HcclResult    StatusMachine();
    void          UpdateInitStatus();
    void          UpdateExchangeStatus();

    HcclResult    GetLocalCcuRmaBufferInfo();
    bool          CreateJetty();
    bool          GetTpInfo();
    void          GenerateLocalPsn();
    void          ResetRequestCtxs();
    HcclResult    StartImportJettyRequest(uint32_t jettyIndex, RequestHandle &reqHandle);
    bool          CheckRequestResults();
    void          ConfigChannel();
    HcclResult    ReleaseConnRes();
    void          ThrowAbnormalStatus(const std::string &funcName);
    std::string   Describe();
};

class CcuTpConnection : public CcuConnection {
public:
    CcuTpConnection(const IpAddress &locAddr, const IpAddress &rmtAddr,
        const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys);
};

class CcuCtpConnection : public CcuConnection {
public:
    CcuCtpConnection(const IpAddress &locAddr, const IpAddress &rmtAddr,
        const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys);
};

} // namespace Hccl
#endif // HCCL_CCU_CONNECTION_H