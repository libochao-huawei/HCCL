/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim public stub header
 */

#ifndef HCCL_SIM_COMMON_STUB_H
#define HCCL_SIM_COMMON_STUB_H
#include <iostream>
#include <string>
#include <vector>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <unistd.h>
#include "sal.h"
#include "../../virt_runtime_fwk/hccl_sim_situation.h"
#include <fcntl.h>
#include <sys/stat.h>
#include "../rts_stub/rts_stub.h"
#include "hccl_sim_pub_stub.h"

rtDataType_t ExtractDataType(uint8_t result);
rtRecudeKind_t ExtractCopyKind(uint8_t result);

// 910D数据类型/Reduce类型转换
rtDataType_t ExtractDataTypeDavid(uint8_t result);
rtRecudeKind_t ExtractReduceTypeDavid(uint8_t result);
rtDataType_t ExtractUbDataTypeDavid(uint32_t type);
rtRecudeKind_t ExtractUbReduceTypeDavid(uint32_t type);

uint64_t GetFull64BitAddr(uint32_t lowAddr, uint32_t highAddr);

ShmCb *GetRankShmCb(int deviceid);
int GetNotifyId(u64 notifyAddr);
ShmPub *GetShmPub();
ShmCb *GetShmCbBaseByRankTemp(int rankId);

#endif // HCCL_SIM_COMMON_STUB_H