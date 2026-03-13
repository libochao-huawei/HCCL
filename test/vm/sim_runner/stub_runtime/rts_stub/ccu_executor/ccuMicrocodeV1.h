/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu instruction header file
 * Author: sunzhepeng
 * Create: 2024-05-14
 */

#ifndef HCCL_SIM_CCU_MICROCODE_V1_H
#define HCCL_SIM_CCU_MICROCODE_V1_H

#include "CcuResourceCommon.h"

namespace SimCcuV1 {
    // 微码指令
    constexpr uint16_t LOAD_TYPE   = 0x0;
    constexpr uint16_t CTRL_TYPE   = 0x1;
    constexpr uint16_t TRANS_TYPE  = 0x2;
    constexpr uint16_t REDUCE_TYPE = 0x3;

    constexpr uint16_t LOADSQEARGSTOGSA_CODE = 0x0;
    constexpr uint16_t LOADSQEARGSTOXN_CODE  = 0x1;
    constexpr uint16_t LOADIMDTOGSA_CODE     = 0x2;
    constexpr uint16_t LOADIMDTOXN_CODE      = 0x3;
    constexpr uint16_t LOADGSAXN_CODE        = 0x4;
    constexpr uint16_t LOADGSAGSA_CODE       = 0x5;
    constexpr uint16_t LOADXX_CODE           = 0x6;

    constexpr uint16_t LOOP_CODE      = 0x0;
    constexpr uint16_t LOOPGROUP_CODE = 0x1;
    constexpr uint16_t SETCKE_CODE    = 0x2;
    constexpr uint16_t CLEARCKE_CODE  = 0x4;
    constexpr uint16_t JMP_CODE       = 0x5;

    constexpr uint16_t TRANSLOCMEMTOLOCMS_CODE  = 0x0;
    constexpr uint16_t TRANSRMTMEMTOLOCMS_CODE  = 0x1;
    constexpr uint16_t TRANSLOCMSTOLOCMEM_CODE  = 0x2;
    constexpr uint16_t TRANSLOCMSTORMTMEM_CODE  = 0x3;
    constexpr uint16_t TRANSRMTMSTOLOCMEM_CODE  = 0x4;
    constexpr uint16_t TRANSLOCMSTOLOCMS_CODE   = 0x5;
    constexpr uint16_t TRANSRMTMSTOLOCMS_CODE   = 0x6;
    constexpr uint16_t TRANSLOCMSTORMTMS_CODE   = 0x7;
    constexpr uint16_t TRANSRMTMEMTOLOCMEM_CODE = 0x8;
    constexpr uint16_t TRANSLOCMEMTORMTMEM_CODE = 0x9;
    constexpr uint16_t TRANSLOCMEMTOLOCMEM_CODE = 0xa;
    constexpr uint16_t SYNCCKE_CODE             = 0xb;
    constexpr uint16_t SYNCGSA_CODE             = 0xc;
    constexpr uint16_t SYNCXN_CODE              = 0xd;

    constexpr uint16_t ADD_CODE = 0x0;
    constexpr uint16_t MAX_CODE = 0x1;
    constexpr uint16_t MIN_CODE = 0x2;

    // CCU资源规格
    const uint16_t INVALID_INSTR_START_INDEX        = 0xFFFF;
    constexpr uint8_t MAX_CCU_CHANNEL_NUM           = 128;
    constexpr uint16_t CCU_RESOURCE_MS_NUM          = 1536;
    constexpr uint32_t CCU_RESOURCE_MS_SIZE         = CCU_RESOURCE_MS_NUM * BYTE_NUM_4K;
    constexpr uint16_t CCU_RESOURCE_GSA_NUM         = 3024; // max:3072
    constexpr uint16_t CCU_RESOURCE_XN_NUM          = 3024; // max:3072
    constexpr uint16_t CCU_RESOURCE_CKE_NUM         = 1008; // max:1024
    constexpr uint16_t CCU_RESOURCE_LOOP_ENGINE_NUM = 192;  // max:200

    constexpr uint16_t CCU_RESOURCE_GSA_MAX         = 3072;
    constexpr uint16_t CCU_RESOURCE_XN_MAX          = 3072;
    constexpr uint16_t CCU_RESOURCE_CKE_MAX         = 1024;
    constexpr uint16_t CCU_RESOURCE_LOOP_ENGINE_MAX = 200;
    constexpr uint16_t CCU_RESOURCE_MEM_SLICE_4K    = 4096;
};

#endif // HCCL_SIM_CCU_MICROCODE_V1_H