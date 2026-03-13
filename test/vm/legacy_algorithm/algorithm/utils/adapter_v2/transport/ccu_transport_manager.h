/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu transport manager header file
 * Create: 2025-02-18
 */

#ifndef HCCL_CCU_TRANS_MANAGER_H
#define HCCL_CCU_TRANS_MANAGER_H

#include <unordered_map>
#include <vector>
#include "types.h"
#include "virtual_topo.h"
#include "ccu_transport.h"
#include "socket_manager.h"

namespace Hccl {

class CcuTransportMgr {
public:
    CcuTransportMgr(const CommunicatorImpl &comm, const int32_t devLogicId);
    virtual ~CcuTransportMgr();
    CcuTransport       *Get(const LinkData &link);
    set<CcuTransport *> Get(RankId rank);
    HcclResult          PrepareCreate(const LinkData &link, CcuTransport *&transport);

    void Confirm(); // 用于正常建联流程，失败需要回退
    void Fallback();
    void Destroy();

    // 以下接口用于N秒快恢特性
    void             Clean(){}
    vector<LinkData> ResumeAll();

    unordered_map<LinkData, unique_ptr<CcuTransport>>& GetCcuLink2Transport();
    void GenSendData();
    void RecvData();
    void RecoverConfirm(); // 用于快照恢复特性，失败报错不回退

private:
    const CommunicatorImpl *comm{nullptr};
    const int32_t devLogicId_{0};
    bool isDestroyed{false};

    unordered_map<LinkData, unique_ptr<CcuTransport>> ccuLink2TransportMap;
    unordered_map<RankId, set<CcuTransport *>>        ccuRank2TransportsMap;
    unordered_map<uint32_t, RankId>                   ccuChannel2RemoteRankMap; // 暂时保留
    vector<LinkData>                                  tempTransport;

    vector<std::pair<CcuTransport*, LinkData>> GetUnConfirmedTrans();
    HcclResult CreateTransportByLink(const LinkData &link, CcuTransport *&transport);
    void       TransportsConnect();
    void       WaitTransportsReady(vector<std::pair<CcuTransport*, LinkData>> &transports) const;

    void RecoverTransportsConnect();
    void WaitTransportsRecoverReady(vector<std::pair<CcuTransport*, LinkData>> &transports) const;
};

} // namespace Hccl

#endif // HCCL_CCU_TRANS_MANAGER_H