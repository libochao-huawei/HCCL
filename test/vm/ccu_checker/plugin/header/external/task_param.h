/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: dfx common data
 * Create： 2025-04-15
 */
#ifndef TASK_PARAM_H
#define TASK_PARAM_H

#include <vector>
#include <string>
#include <memory>
#include "securec.h"
#include "hccl/base.h"
#include "const_val.h"
#include "enum_factory.h"

namespace Hccl {

MAKE_ENUM(DmaOp, HCCL_DMA_READ, HCCL_DMA_WRITE, HCCL_DMA_NOTIFY_WAIT)

MAKE_ENUM(AlgType, NOT_SPECIFIED, RING, MULTI_RING, MESH, RECURSIVE_HD, BINARY_HD, PAIR_WISE, INVALID_VAL)

MAKE_ENUM(TaskParamType, TASK_SDMA, TASK_RDMA, TASK_REDUCE_INLINE, TASK_REDUCE_TBE, TASK_NOTIFY_RECORD, TASK_NOTIFY_WAIT,
    TASK_SEND_NOTIFY, TASK_SEND_PAYLOAD, TASK_WRITE_WITH_NOTIFY, TASK_WRITE_REDUCE_WITH_NOTIFY, TASK_CCU, TASK_AICPU_KERNEL, TASK_AICPU_REDUCE)

MAKE_ENUM(DfxLinkType, ONCHIP, HCCS, PCIE, ROCE, SIO, HCCS_SW, STANDARD_ROCE, UB, UBoE, RESERVED)

MAKE_ENUM(CcuProfilinType, CCU_TASK_PROFILING, CCU_WAITCKE_PROFILING, CCU_LOOPGROUP_PROFILING, CCU_MAP_PROFILING)
 
constexpr uint16_t  CCU_MAX_CHANNEL_NUM     = 16;     // 最多16条link
constexpr uint16_t  INVALID_CKE_ID          = 0xFFFF; // CKE ID非法值
constexpr uint16_t  INVALID_VALUE_CHANNELID = 0xFFFF; // channel id非法值
constexpr u64       INVALID_VALUE_NOTIFYID  = 0xFFFFFFFFFFFFFFFF; // NOTIFY id非法值

struct CcuProfilingInfo {
    std::string name;          // CCU任务名或微码名
    uint8_t type;              // 枚举，0为Task粒度，1为WaitCKE，2为LoopGroup，3为channelId->RemoteRankId的映射
    uint8_t dieId;             // CCU任务执行的DieId
    uint8_t missionId;         // CCU任务执行的MissionId
    uint16_t instrId;
    uint8_t reduceOpType;      // 与HcclReduceOp类型保持一致
    uint8_t inputDataType;     // 与HcclDataType类型保持一致
    uint8_t outputDataType;    // 与HcclDataType类型保持一致
    uint64_t dataSize;         // 输入数据大小
    uint32_t ckeId;
    uint32_t mask;
    uint16_t channelId[CCU_MAX_CHANNEL_NUM];    // LoopGroup所包含的搬运指令使用的ChannelId
    uint32_t remoteRankId[CCU_MAX_CHANNEL_NUM]; // LoopGroup所包含的搬运指令的对端
 
    CcuProfilingInfo() : name(""), type(0), dieId(0), missionId(0), instrId(0), reduceOpType(0), inputDataType(0), outputDataType(0), dataSize(0), ckeId(0), mask(0) {
        (void)memset_s(channelId, sizeof(channelId), INVALID_VALUE_CHANNELID, sizeof(channelId));
        (void)memset_s(remoteRankId, sizeof(remoteRankId), INVALID_RANKID, sizeof(remoteRankId));
    }
};

struct ParaDMA {
    const void *src;
    const void *dst;
    std::size_t size;
    u64         notifyID;
    DfxLinkType linkType;
    DmaOp       dmaOp;
};

struct ParaReduce {
    const void  *src;
    const void  *dst;
    std::size_t  size;
    u64          notifyID;
    DfxLinkType  linkType;
    HcclReduceOp reduceOp;
    HcclDataType dataType;
};

struct ParaNotify {
    u64 notifyID;
    u32 value;
};

constexpr u32 CCU_COSTOM_ARGS_LEN = 32;
struct ParaCcu {
    u8  dieId;
    u8  missionId;
    u8  execMissionId;
    u32 instrId;
    u64 costumArgs[CCU_COSTOM_ARGS_LEN];
    u64 executeId;
};

struct TaskParam {
    TaskParamType taskType;
    u64           beginTime;
    u64           endTime;
    union {
        ParaDMA    DMA;    // taskType = SDMA/RDMA使用, 包括rtRDMASend写notify
        ParaReduce Reduce; // taskType = inline/CCE Reduce使用
        ParaNotify Notify; // taskType = Noitfy Record/Wait使用
        ParaCcu    Ccu;
    } taskPara;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo; // taskType为TASK_CCU时，ParaCcu的补充profiling信息
    std::string Describe() const
    {
        return StringFormat("TaskParam[taskType[%s] beginTime[%llu] endTime[%llu]]", taskType.Describe().c_str(), beginTime, endTime);
    }
};

} // namespace Hccl

#endif