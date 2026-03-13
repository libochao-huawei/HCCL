

#include "ccu_connection.h"
#include "adapter_rts.h"
#include "rdma_handle_manager.h"

namespace Hccl {

CcuConnection::CcuConnection(const IpAddress &locAddr, const IpAddress &rmtAddr,
    const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys)
    : locAddr_(locAddr), rmtAddr_(rmtAddr), channelInfo_(channelInfo), ccuJettys_(ccuJettys)
{
}

CcuTpConnection::CcuTpConnection(const IpAddress &locAddr, const IpAddress &rmtAddr,
    const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys)
{
    tpProtocol = TpProtocol::TP;
}

CcuCtpConnection::CcuCtpConnection(const IpAddress &locAddr, const IpAddress &rmtAddr,
    const CcuChannelInfo &channelInfo, const std::vector<CcuJetty *> &ccuJettys)
    : CcuConnection(locAddr, rmtAddr, channelInfo, ccuJettys)
{
    tpProtocol = TpProtocol::CTP;
}

HcclResult CcuConnection::Init()
{
    devLogicId = HrtGetDevice();
    uint32_t devPhyId = HrtGetDevicePhyIdByIndex(devLogicId);

    auto &rdmaHandleMgr = RdmaHandleManager::GetInstance();
    rdmaHandle = rdmaHandleMgr.GetByIp(devPhyId, locAddr_);
    auto dieIdAndFuncId = HraGetDieAndFuncId(rdmaHandle);
    dieId = channelInfo_.dieId;
    status = CcuConnStatus::INIT;
    innerStatus = InnerStatus::INIT;
    return HcclResult::HCCL_SUCCESS;
}

IpAddress CcuConnection::GetLocAddr()
{
    return locAddr_;
}

IpAddress CcuConnection::GetRmtAddr()
{
    return rmtAddr_;
}

HcclResult CcuConnection::ReleaseConnRes()
{
    return HcclResult::HCCL_SUCCESS;
}


CcuConnection::~CcuConnection()
{
}

uint32_t CcuConnection::GetChannelId() const
{
    return channelInfo_.channelId;
}

int32_t CcuConnection::GetDevLogicId() const
{
    return devLogicId;
}

uint32_t CcuConnection::GetDieId() const
{
    return channelInfo_.dieId;
}

}