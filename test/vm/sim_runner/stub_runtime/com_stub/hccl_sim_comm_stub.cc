/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim public stub
 */

#include "hccl_sim_comm_stub.h"
#include "hccl_sim_pub_stub.h"
#include "sqe_v82.h"
#include "sqe.h"
#include <fstream>
#ifdef DEVICE_STUB
#endif
#include "SimRunnerMgr.h"

rtDataType_t ExtractDataType(uint8_t result)
{
    // 提取高 4 位（copyKind）
    uint8_t copyKind = (result)&0xF0;
    if (copyKind == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8)) {
        return rtDataType_t::RT_DATA_TYPE_INT8;
    }
    if (copyKind == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16)) {
        return rtDataType_t::RT_DATA_TYPE_INT16;
    }
    if (copyKind == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32)) {
        return rtDataType_t::RT_DATA_TYPE_INT32;
    }
    if (copyKind == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16)) {
        return rtDataType_t::RT_DATA_TYPE_FP16;
    }
    if (copyKind == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32)) {
        return rtDataType_t::RT_DATA_TYPE_FP32;
    }
    return rtDataType_t::RT_DATA_TYPE_END;
}

rtRecudeKind_t ExtractCopyKind(uint8_t result)
{
    // 提取低 4 位（copyDataType）
    uint8_t copyDataType = result & 0x0F;  // 直接与 0x0F 按位与，保留低 4 位
    if (copyDataType == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD)) {
        return rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD;
    }
    if (copyDataType == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX)) {
        return rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MAX;
    }
    if (copyDataType == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN)) {
        return rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MIN;
    }
    if (copyDataType == static_cast<uint8_t>(RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL)) {
        return rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_EQUAL;
    }
    return rtRecudeKind_t::RT_RECUDE_KIND_END;
}

// Hccl::RtStarsMemcpyAsyncOperationKind
std::map<uint32_t, rtRecudeKind_t> DavidReduceOpMap = {
    {0x01, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD},
    {0x02, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {0x03, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MIN},
    {0x04, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_EQUAL}
};

rtRecudeKind_t ExtractReduceTypeDavid(uint8_t result)
{
    uint8_t reduceType = result & 0x0F;  // 直接与 0x0F 按位与，保留低 4 位
    if (DavidReduceOpMap.find(static_cast<uint32_t>(reduceType)) != DavidReduceOpMap.end()) {
        return DavidReduceOpMap[reduceType];
    }
    return rtRecudeKind_t::RT_RECUDE_KIND_END;
}

// Hccl::RtStarsMemcpyAsyncDataType
std::map<uint32_t, rtDataType_t> DavidDataTypeMap = {
    {0x00, rtDataType_t::RT_DATA_TYPE_INT8},
    {0x10, rtDataType_t::RT_DATA_TYPE_INT16},
    {0x20, rtDataType_t::RT_DATA_TYPE_INT32},
    {0x60, rtDataType_t::RT_DATA_TYPE_FP16},
    {0x70, rtDataType_t::RT_DATA_TYPE_FP32}
};

rtDataType_t ExtractDataTypeDavid(uint8_t result)
{
    uint8_t dataType = static_cast<uint32_t>(result) & 0xF0;  // 提取高4位
    if (DavidDataTypeMap.find(static_cast<uint32_t>(dataType)) != DavidDataTypeMap.end()) {
        return DavidDataTypeMap[dataType];
    }
    return rtDataType_t::RT_DATA_TYPE_END;  // bfp16?
}

std::map<u32, rtRecudeKind_t> DavidUbReduceOpMap = {
    {0xA, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD},
    {0x8, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {0x9, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MIN}
};

rtRecudeKind_t ExtractUbReduceTypeDavid(uint32_t type)
{
    if (DavidUbReduceOpMap.find(type) != DavidUbReduceOpMap.end()) {
        return DavidUbReduceOpMap[type];
    }
    return rtRecudeKind_t::RT_RECUDE_KIND_END;
}

std::map<u32, rtDataType_t> DavidUbDataTypeMap = {
    {0x0, rtDataType_t::RT_DATA_TYPE_INT8},
    {0x1, rtDataType_t::RT_DATA_TYPE_INT16},
    {0x2, rtDataType_t::RT_DATA_TYPE_INT32},
    {0x3, rtDataType_t::RT_DATA_TYPE_UINT8},
    {0x4, rtDataType_t::RT_DATA_TYPE_UINT16},
    {0x5, rtDataType_t::RT_DATA_TYPE_UINT32},
    {0x6, rtDataType_t::RT_DATA_TYPE_FP16},
    {0x7, rtDataType_t::RT_DATA_TYPE_FP32}
};

rtDataType_t ExtractUbDataTypeDavid(uint32_t type)
{
    if (DavidUbDataTypeMap.find(type) != DavidUbDataTypeMap.end()) {
        return DavidUbDataTypeMap[type];
    }
    return rtDataType_t::RT_DATA_TYPE_END;
}

uint64_t GetFull64BitAddr(uint32_t lowAddr, uint32_t highAddr)
{
    return (static_cast<uint64_t>(highAddr) << SHIFT_BIT32) | static_cast<uint64_t>(lowAddr);
}

ShmCb * GetRankShmCb(int deviceid)
{
    return reinterpret_cast<ShmCb *>(SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetShmCbBaseByRank(deviceid));
}

int GetNotifyId(u64 notifyAddr)
{
    return SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetNotifyId(notifyAddr);
}

ShmPub *GetShmPub()
{
    return SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetShmPub();
}

ShmCb *GetShmCbBaseByRankTemp(int rankId)
{
    return reinterpret_cast<ShmCb *>(SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetShmCbBaseByRank(rankId));
}