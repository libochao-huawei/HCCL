#include "tp_manager.h"

#include "hccp_ctx.h"
#include "hccl_common_v2.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"


namespace Hccl {

void TpManager::Init()
{ }

TpManager& TpManager::GetInstance(const int32_t deviceLogicId)
{
    static TpManager tpManager[MAX_MODULE_DEVICE_NUM];

    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>("[TpManager][%s] Failed to get instance. "
            "devLogicId[%d] should be less than %u.", __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
    }

    return tpManager[deviceLogicId];
}

HcclResult TpManager::GetTpInfo(const RaUbGetTpInfoParam &param, TpInfo &tpInfo)
{
    TpInfo info;
    tpInfo = info;
    return HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpInfo(const RaUbGetTpInfoParam &param, const TpInfo &tpInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

}