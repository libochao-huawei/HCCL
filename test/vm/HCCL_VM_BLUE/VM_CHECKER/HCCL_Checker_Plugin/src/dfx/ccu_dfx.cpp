/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu dfx api implementation file
 * Create: 2025-04-16
 */

#include "ccu_dfx.h"

#include "hccl_common_v2.h"
#include "exception_util.h"
#include "ccu_error_handler.h"

namespace Hccl {
using namespace std;

HcclResult GetCcuErrorMsg(s32 deviceId, const ParaCcu &ccuTaskParam, std::vector<CcuErrorInfo> &errorInfo)
{
    TRY_CATCH_RETURN(
        HCCL_RUN_INFO(
            "[CcuDfx]GetCcuErrorMsg: deviceId[%d], dieId[%u], missionId[%u], execMissionId[%u], executeId[%llu].",
            deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.missionId),
            static_cast<u32>(ccuTaskParam.execMissionId), ccuTaskParam.executeId);

        // 入参校验
        CHK_PRT_RET((deviceId < 0 || static_cast<u32>(deviceId) >= MAX_MODULE_DEVICE_NUM),
                    HCCL_ERROR("[CcuDfx][GetCcuErrorMsg]deviceId[%d] error.", deviceId), HcclResult::HCCL_E_PARA);

        CcuErrorHandler::GetCcuErrorMsg(deviceId, ccuTaskParam, errorInfo);
    );
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl