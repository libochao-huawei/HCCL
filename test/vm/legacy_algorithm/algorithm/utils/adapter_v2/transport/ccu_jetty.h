/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu jetty header file
 * Create: 2025-08-11
 */

#ifndef HCCL_CCU_JETTY_H
#define HCCL_CCU_JETTY_H

#include "ip_address.h"
#include "ccu_dev_mgr.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

class CcuJetty final {
public:
    CcuJetty(const IpAddress &ipAddr, const CcuJettyInfo &jettyInfo);
    ~CcuJetty();
    CcuJetty(const CcuJetty &that) = delete;
    CcuJetty &operator=(const CcuJetty &that) = delete;
    CcuJetty(CcuJetty &&that) = delete;
    CcuJetty &operator=(CcuJetty &&that) = delete;

    HcclResult CreateJetty();

    HrtRaUbCreateJettyParam GetCreateJettyParam() const;
    HrtRaUbJettyCreatedOutParam GetJettyedOutParam() const;

private:
    int32_t devLogicId_{0};
    IpAddress ipAddr_{};
    CcuJettyInfo jettyInfo_{};

    RdmaHandle rdmaHandle_{nullptr};
    bool isCreated_{false};
    bool isError_{false};

    RequestHandle reqHandle_{0};
    std::vector<char> reqDataBuffer_;
    void *jettyHandlePtr_{nullptr};

    HrtRaUbCreateJettyParam inParam_{};
    HrtRaUbJettyCreatedOutParam outParam_{};

    HcclResult HandleAsyncRequest();
};

//
HcclResult CcuCreateJetty(const IpAddress &ipAddr, const CcuJettyInfo &jettyInfo,
    std::unique_ptr<CcuJetty> &ccuJetty);

} // namespace Hccl
#endif // HCCL_CCU_JETTY_H