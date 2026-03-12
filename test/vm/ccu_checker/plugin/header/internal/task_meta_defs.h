#ifndef HCCL_COMMON_DEFS_H
#define HCCL_COMMON_DEFS_H

#include "hccl_types.h"
#include <vector>
#include <cstdint>

#define RT_CCU_SQE_ARGS_LEN     (13U)

enum class LinkProto { 
    SDMA = 0,
    RDMA = 1,
    CCU = 2,
    INVALID_A = 3,
};

enum class HccLTaskMetaType : char {
    NOTIFY_WAIT,
    NOTIFY_RECORD,
    REDUCE,
    MEM_CPY,
    CCU_GRAPH,
    AIV_GRAPH,
    EVENT_WAIT,
    EVENT_RECORD
};

typedef enum {
    COMM_PROTOCOL_RESERVED = -1,  ///< 保留协议类型
    COMM_PROTOCOL_HCCS = 0,        ///< HCCS协议
    COMM_PROTOCOL_TCP = 1,        ///< 标准TCP协议
    COMM_PROTOCOL_ROCE = 2,       ///< RDMA over Converged Ethernet
    COMM_PROTOCOL_UB_CTP = 3,    ///< 华为统一总线UB_CTP
    COMM_PROTOCOL_UB_TP = 4,     ///< 华为统一总线UB_TP
    COMM_PROTOCOL_PCIE = 5,      ///< PCIE协议
    COMM_PROTOCOL_SIO = 6,        ///< SIO协议
} CommProtocol;

#pragma pack(push, 1)

typedef struct {
    uint32_t    srcRankId;
    uint64_t    srcOffset;
    uint32_t    dstRankId;
    uint64_t    dstOffset;
    uint64_t    len;
    uint8_t     protocol;
} TransMemTask;

typedef struct {
    uint32_t    srcRankId;
    uint64_t    srcOffset;
    uint32_t    dstRankId;
    uint64_t    dstOffset;
    uint64_t    dataCount;
    uint8_t     dataType;   // HcclDataType from hccl_types.h
    uint8_t     reduceOp;   // HcclReduceOp from hccl_types.h
    uint8_t     protocol;
    uint8_t     reserved;
} ReduceTask;

typedef struct {
    uint32_t    srcRankId;
    uint64_t    notifyId;
    uint32_t    dstRankId;
    uint8_t     notifyCount;
    uint8_t     protocol;
} NotifyTask;

typedef struct {
    uint8_t    dieId;
    uint8_t    missionId;
    uint16_t   timeout;
    uint16_t   instStartId;
    uint16_t   instCnt;
    uint32_t   key;
    uint32_t   argSize;
    uint64_t   args[RT_CCU_SQE_ARGS_LEN];
} CcuTask;

typedef struct {
    HccLTaskMetaType    taskType;
    uint16_t            commId;
    uint32_t            rankId;
    uint64_t            streamId;
    union {
        TransMemTask    transMem;
        ReduceTask      reduce;
        NotifyTask      notify;
        CcuTask         ccu;
    }                   taskData;
} HcclTaskMetaData;

typedef struct {
    uint64_t    taskCid;
    uint16_t    dispatchId;
} HcclTaskReq;

typedef struct {
    uint64_t    taskCid;
    uint16_t    status;
} HcclTaskRsp;

#pragma pack(pop)

#endif