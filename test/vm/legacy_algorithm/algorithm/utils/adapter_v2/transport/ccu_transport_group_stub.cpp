
#include "ccu_transport_group.h"
#include "adapter_rts.h"

namespace Hccl {

u32 CcuTransportGroup::GetCntCkeId(u32 index) const
{
    if (index >= cntCkesGroup.size()) {
        THROW<InvalidParamsException>(StringFormat("[CcuTransportGroup::%s] Index[%u] is bigger than cntCkesGroup size[%u], please check.",
            __func__, index, cntCkesGroup.size()));
    }
    return cntCkesGroup[index];
}


bool CcuTransportGroup::CheckTransports(const vector<CcuTransport*> &transports)
{
    if (transports.size() == 0) {
        HCCL_ERROR("[CcuTransportGroup::%s] Transports size is 0, please check.", __func__);
        return false;
    }

    // 校验transports中所有的DieId是否相等，如果不相等,则构建失败，如果相等，则将transports传入transportsGrp中
    for (unsigned int i = 0; i < transports.size(); i++) {
        if (transports[0]->GetDieId() != transports[i]->GetDieId()) {
            HCCL_ERROR("[CcuTransportGroup::%s] Transports dieId is not equal, please check.", __func__);
            return false;
        }
    }
    return true;
}

bool CcuTransportGroup::CheckTransportCntCke()
{
    devLogicId = HrtGetDevice();
    HcclResult allocResHandleReturnValue = CcuDeviceManager::AllocCke(devLogicId, cntCkesGroupDieId, cntCkeNumTransportGroupUse, ckeInfoTransportGroupUse);
    if (allocResHandleReturnValue != HCCL_SUCCESS) {
        HCCL_ERROR("[CcuTransportGroup::%s] Failed to allocate cntCke resource, please check.", __func__);
        return false;
    }

    for (u32 i = 0; i < ckeInfoTransportGroupUse.size(); i++) {
        u32 ckeNum = ckeInfoTransportGroupUse[i].num;
        u32 ckesStartId = ckeInfoTransportGroupUse[i].startId;
        for (unsigned int j = 0; j < ckeNum; j++) {
            cntCkesGroup.push_back(ckesStartId + j);
        }
    }

    for (auto &transport : transportsGrp) {
        transport->SetCntCke(cntCkesGroup);
    }
    return true;
}

CcuTransportGroup::CcuTransportGroup(const vector<CcuTransport*> &transports, u32 cntCkeNum):isDestroyed(false)
{
    if (!CheckTransports(transports)) {
        grpStatus = TransportGrpStatus::FAIL;
        HCCL_ERROR("[CcuTransportGroup::%s] Func CheckTransports failed, please check.", __func__);
        return;
    }

    transportsGrp = transports;
    cntCkesGroupDieId = transports[0]->GetDieId();
    cntCkeNumTransportGroupUse = cntCkeNum;

    if (!CheckTransportCntCke()) {
        grpStatus = TransportGrpStatus::FAIL;
        HCCL_ERROR("[CcuTransportGroup::%s] Func CheckTransportCntCke failed, please check.", __func__);
        return;
    }

    grpStatus = TransportGrpStatus::INIT;
}

const vector<CcuTransport*> &CcuTransportGroup::GetTransports() const
{
    return transportsGrp;
}

TransportGrpStatus CcuTransportGroup::GetGrpStatus() const
{
    return grpStatus;
}

CcuTransportGroup::~CcuTransportGroup()
{
    if (!isDestroyed) {
        Destroy();
    }

    // 调用ReleaseResHandle接口，用来释放cntResHandleTransportGroupUse
    auto ret = CcuDeviceManager::ReleaseCke(devLogicId, cntCkesGroupDieId, ckeInfoTransportGroupUse);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuTransportGroup::%s] Release ckesRes failed, ret[%d], ckeInfo size is [%u]", __func__, ret, ckeInfoTransportGroupUse.size());
        for (auto& ckeInfo : ckeInfoTransportGroupUse) {
            HCCL_ERROR("[CcuTransportGroup::%s] Release ckesRes failed, ckeInfo.startId[%u], ckeInfo.num[%u]", __func__, ckeInfo.startId, ckeInfo.num);
        }
    } else {
        HCCL_DEBUG("[CcuTransportGroup::%s] CcuTransportGroup Destructor success.", __func__);
    }
}

void CcuTransportGroup::Destroy()
{
    isDestroyed = true;
    transportsGrp.clear();
}

}