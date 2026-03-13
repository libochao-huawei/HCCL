/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu jetty
 * Create: 2025-08-11
 */

#include "ccu_jetty.h"

#include "hccp_ctx.h"
#include "rdma_handle_manager.h"
#include "local_ub_rma_buffer.h"

namespace Hccl {

HcclResult CcuCreateJetty(const IpAddress &ipAddr, const CcuJettyInfo &jettyInfo,
    std::unique_ptr<CcuJetty> &ccuJetty)
{
    TRY_CATCH_RETURN(
        ccuJetty = std::make_unique<CcuJetty>(ipAddr, jettyInfo);
    );
    return HcclResult::HCCL_SUCCESS;
}

CcuJetty::CcuJetty(const IpAddress &ipAddr, const CcuJettyInfo &jettyInfo)
    : ipAddr_(ipAddr), jettyInfo_(jettyInfo)
{
    devLogicId_ = HrtGetDevice();
    uint32_t devPhyId = HrtGetDevicePhyIdByIndex(devLogicId_);
    auto &rdmaHandleMgr = RdmaHandleManager::GetInstance();
    rdmaHandle_ = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);
    const auto jfcHandle = 1;
    const auto &tokenInfo = rdmaHandleMgr.GetTokenIdInfo(rdmaHandle_);
    const auto tokenIdHandle = tokenInfo.first;
    const auto tokenValue = GetUbToken();
    const auto jettyMode = HrtJettyMode::CCU_CCUM_CACHE; // 当前仅支持该模式

    inParam_ = HrtRaUbCreateJettyParam{jfcHandle, jfcHandle, tokenValue,
        tokenIdHandle, jettyMode, jettyInfo.taJettyId, jettyInfo.sqBufVa,
        jettyInfo.sqBufSize, jettyInfo.wqeBBStartId, jettyInfo.sqDepth};
}

CcuJetty::~CcuJetty()
{
}

HcclResult CcuJetty::CreateJetty()
{
    return HcclResult::HCCL_SUCCESS; // 打桩保证创建一定成功
}

HrtRaUbCreateJettyParam CcuJetty::GetCreateJettyParam() const
{
    return inParam_;
}

HrtRaUbJettyCreatedOutParam CcuJetty::GetJettyedOutParam() const
{
    return outParam_;
}

} // namespace Hccl