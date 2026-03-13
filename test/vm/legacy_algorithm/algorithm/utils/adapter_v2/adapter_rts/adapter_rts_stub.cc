/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: adaper rts stub
 * Author:
 * Create: 2025-05-29
 */

#include "adapter_rts.h"
#include "exception_util.h"
#include "runtime_api_exception.h"

#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "rank_info_recorder.h"


using namespace std;
namespace Hccl {

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen)
{
    return RT_ERROR_NONE;
}

const std::unordered_map<std::string, DevType> SOC_VER_CONVERT{{"Ascend310P1", DevType::DEV_TYPE_V51_310_P1},
                                                               {"Ascend310P3", DevType::DEV_TYPE_V51_310_P3},
                                                               {"Ascend910", DevType::DEV_TYPE_910A},
                                                               {"Ascend910A", DevType::DEV_TYPE_910A},
                                                               {"Ascend910B", DevType::DEV_TYPE_910A},
                                                               {"Ascend910ProA", DevType::DEV_TYPE_910A},
                                                               {"Ascend910ProB", DevType::DEV_TYPE_910A},
                                                               {"Ascend910PremiumA", DevType::DEV_TYPE_910A},
                                                               {"Ascend910B1", DevType::DEV_TYPE_910A2},
                                                               {"Ascend910B2", DevType::DEV_TYPE_910A2},
                                                               {"Ascend910B3", DevType::DEV_TYPE_910A2},
                                                               {"Ascend910B4", DevType::DEV_TYPE_910A2},
                                                               {"Ascend910C1", DevType::DEV_TYPE_910A3},
                                                               {"Ascend910C2", DevType::DEV_TYPE_910A3},
                                                               {"Ascend910C3", DevType::DEV_TYPE_910A3},
                                                               {"Ascend910_9591", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_9599", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_9589", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_958a", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_958b", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_957b", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_957d", DevType::DEV_TYPE_910_95},
                                                               {"Ascend910_950z", DevType::DEV_TYPE_910_95},
                                                               {"nosoc", DevType::DEV_TYPE_NOSOC}};

// 添加编译宏，防止返回82类型芯片造成已有UT失效
DevType HrtGetDeviceType()
{
    char targetChipVer[CHIP_VERSION_MAX_LEN] = {0};
    HrtGetSocVer(targetChipVer, CHIP_VERSION_MAX_LEN);

    auto iter = SOC_VER_CONVERT.find(reinterpret_cast<char *>(targetChipVer));
    if (iter == SOC_VER_CONVERT.end()) {
        HCCL_ERROR("[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get "
                   "illegal chipver, chip_ver[%s].",
                   HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), targetChipVer);

        throw RuntimeApiException("call HrtGetSocVer failed.");
    }
    return iter->second;
}

void HrtGetSocVer(char_t *chipVer, const u32 size)
{
    rtError_t ret = rtGetSocVersion(reinterpret_cast<char_t *>(chipVer), size);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[Get][SocVer]errNo[0x%016llx] rtGet deviceVer failed, "
                   "return value[%d].",
                   HCCL_ERROR_CODE((HcclResult::HCCL_E_RUNTIME)), ret);
        throw RuntimeApiException("call rtGetSocVersion failed. ");
    }
}

void HrtUbDevQueryInfo(rtUbDevQueryCmd cmd, void *devInfo)
{ }

s32 HrtGetDevicePhyIdByIndex(s32 deviceLogicId)
{
    return deviceLogicId;
}

void *HrtMalloc(u64 size, rtMemType_t memType)
{
    void     *devPtr = (void*)0x01;
    return devPtr;
}

void HrtFree(void *devPtr)
{
    return;
}

void HrtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind)
{
    ;
}

s32 HrtGetDevice()
{
    u32 rankid = checker::RankInfoRecorder::Global()->GetRankId();
    return checker::RankInfoRecorder::Global()->rankId2phyId[rankid];
}

HcclResult HrtGetMainboardId(uint32_t deviceLogicId, HcclMainboardId &hcclMainboardId)
{
    hcclMainboardId = HcclMainboardId::MAINBOARD_A_K_SERVER;
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
