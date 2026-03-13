/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 适配1.0的类型转换接口
 * Author: huangweihao, yinding
 * Create: 2025-05-06
 */

#include "transformer.h"

namespace hccl {
//转换函数
std::map<CheckerOpType, HcclCMDType> g_CheckerOpType2HcclCMDType = {
    {CheckerOpType::INVALID, HcclCMDType::HCCL_CMD_INVALID},
    {CheckerOpType::BROADCAST, HcclCMDType::HCCL_CMD_BROADCAST},
    {CheckerOpType::ALLREDUCE, HcclCMDType::HCCL_CMD_ALLREDUCE},
    {CheckerOpType::REDUCE, HcclCMDType::HCCL_CMD_REDUCE},
    {CheckerOpType::SEND, HcclCMDType::HCCL_CMD_SEND},
    {CheckerOpType::RECEIVE, HcclCMDType::HCCL_CMD_RECEIVE},
    {CheckerOpType::ALLGATHER, HcclCMDType::HCCL_CMD_ALLGATHER},
    {CheckerOpType::REDUCE_SCATTER, HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
    {CheckerOpType::ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLV},
    {CheckerOpType::ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLTOALLVC},
    {CheckerOpType::ALLTOALL, HcclCMDType::HCCL_CMD_ALLTOALL},
    {CheckerOpType::GATHER, HcclCMDType::HCCL_CMD_GATHER},
    {CheckerOpType::SCATTER, HcclCMDType::HCCL_CMD_SCATTER},
    {CheckerOpType::BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV},
    {CheckerOpType::BATCH_PUT, HcclCMDType::HCCL_CMD_BATCH_PUT},
    {CheckerOpType::BATCH_GET, HcclCMDType::HCCL_CMD_BATCH_GET},
    {CheckerOpType::ALLGATHER_V, HcclCMDType::HCCL_CMD_ALLGATHER_V},
    {CheckerOpType::REDUCE_SCATTER_V, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V},
    {CheckerOpType::BATCH_WRITE, HcclCMDType::HCCL_CMD_BATCH_WRITE},
    {CheckerOpType::ALL, HcclCMDType::HCCL_CMD_ALL},
    {CheckerOpType::MAX, HcclCMDType::HCCL_CMD_MAX}
};

//转换函数
std::map<HcclCMDType, CheckerOpType> g_HcclCMDType2CheckerOpType = {
    {HcclCMDType::HCCL_CMD_INVALID, CheckerOpType::INVALID},
    {HcclCMDType::HCCL_CMD_BROADCAST, CheckerOpType::BROADCAST},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, CheckerOpType::ALLREDUCE},
    {HcclCMDType::HCCL_CMD_REDUCE, CheckerOpType::REDUCE},
    {HcclCMDType::HCCL_CMD_SEND ,CheckerOpType::SEND},
    {HcclCMDType::HCCL_CMD_RECEIVE, CheckerOpType::RECEIVE},
    {HcclCMDType::HCCL_CMD_ALLGATHER, CheckerOpType::ALLGATHER},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, CheckerOpType::REDUCE_SCATTER},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, CheckerOpType::ALLTOALLV},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, CheckerOpType::ALLTOALLVC},
    {HcclCMDType::HCCL_CMD_ALLTOALL, CheckerOpType::ALLTOALL},
    {HcclCMDType::HCCL_CMD_GATHER, CheckerOpType::GATHER},
    {HcclCMDType::HCCL_CMD_SCATTER, CheckerOpType::SCATTER},
    {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, CheckerOpType::BATCH_SEND_RECV},
    {HcclCMDType::HCCL_CMD_BATCH_PUT, CheckerOpType::BATCH_PUT},
    {HcclCMDType::HCCL_CMD_BATCH_GET, CheckerOpType::BATCH_GET},
    {HcclCMDType::HCCL_CMD_ALLGATHER_V, CheckerOpType::ALLGATHER_V},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, CheckerOpType::REDUCE_SCATTER_V},
    {HcclCMDType::HCCL_CMD_BATCH_WRITE, CheckerOpType::BATCH_WRITE},
    {HcclCMDType::HCCL_CMD_ALL, CheckerOpType::ALL},
    {HcclCMDType::HCCL_CMD_MAX, CheckerOpType::MAX}
};

std::map<CheckerReduceOp, HcclReduceOp> g_CheckerReduceOp2HcclReduceOp = {
    {CheckerReduceOp::REDUCE_SUM, HcclReduceOp::HCCL_REDUCE_SUM},
    {CheckerReduceOp::REDUCE_PROD, HcclReduceOp::HCCL_REDUCE_PROD},
    {CheckerReduceOp::REDUCE_MAX, HcclReduceOp::HCCL_REDUCE_MAX},
    {CheckerReduceOp::REDUCE_MIN, HcclReduceOp::HCCL_REDUCE_MIN},
    {CheckerReduceOp::REDUCE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED}
};

std::map<HcclReduceOp, CheckerReduceOp> g_HcclReduceOp2CheckerReduceOp = {
    {HcclReduceOp::HCCL_REDUCE_SUM, CheckerReduceOp::REDUCE_SUM},
    {HcclReduceOp::HCCL_REDUCE_PROD, CheckerReduceOp::REDUCE_PROD},
    {HcclReduceOp::HCCL_REDUCE_MAX, CheckerReduceOp::REDUCE_MAX},
    {HcclReduceOp::HCCL_REDUCE_MIN, CheckerReduceOp::REDUCE_MIN},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, CheckerReduceOp::REDUCE_RESERVED}
};

std::map<CheckerDataType, HcclDataType> g_CheckerDataType2HcclDataType = {
    {CheckerDataType::DATA_TYPE_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
    {CheckerDataType::DATA_TYPE_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
    {CheckerDataType::DATA_TYPE_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
    {CheckerDataType::DATA_TYPE_FP16, HcclDataType::HCCL_DATA_TYPE_FP16},
    {CheckerDataType::DATA_TYPE_FP32, HcclDataType::HCCL_DATA_TYPE_FP32},
    {CheckerDataType::DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_INT64},
    {CheckerDataType::DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_UINT64},
    {CheckerDataType::DATA_TYPE_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {CheckerDataType::DATA_TYPE_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {CheckerDataType::DATA_TYPE_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {CheckerDataType::DATA_TYPE_FP64, HcclDataType::HCCL_DATA_TYPE_FP64},
    {CheckerDataType::DATA_TYPE_BFP16, HcclDataType::HCCL_DATA_TYPE_BFP16},
    {CheckerDataType::DATA_TYPE_INT128, HcclDataType::HCCL_DATA_TYPE_INT128},
    {CheckerDataType::DATA_TYPE_HIF8, HcclDataType::HCCL_DATA_TYPE_HIF8},
    {CheckerDataType::DATA_TYPE_FP8E4M3, HcclDataType::HCCL_DATA_TYPE_FP8E4M3},
    {CheckerDataType::DATA_TYPE_FP8E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E5M2},
    {CheckerDataType::DATA_TYPE_RESERVED, HcclDataType::HCCL_DATA_TYPE_RESERVED}
};

std::map<HcclDataType, CheckerDataType> g_HcclDataType2CheckerDataType = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, CheckerDataType::DATA_TYPE_INT8},
    {HcclDataType::HCCL_DATA_TYPE_INT16, CheckerDataType::DATA_TYPE_INT16},
    {HcclDataType::HCCL_DATA_TYPE_INT32, CheckerDataType::DATA_TYPE_INT32},
    {HcclDataType::HCCL_DATA_TYPE_FP16, CheckerDataType::DATA_TYPE_FP16},
    {HcclDataType::HCCL_DATA_TYPE_FP32, CheckerDataType::DATA_TYPE_FP32},
    {HcclDataType::HCCL_DATA_TYPE_INT64, CheckerDataType::DATA_TYPE_INT64},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, CheckerDataType::DATA_TYPE_UINT64},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, CheckerDataType::DATA_TYPE_UINT8},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, CheckerDataType::DATA_TYPE_UINT16},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, CheckerDataType::DATA_TYPE_UINT32},
    {HcclDataType::HCCL_DATA_TYPE_FP64, CheckerDataType::DATA_TYPE_FP64},
    {HcclDataType::HCCL_DATA_TYPE_BFP16, CheckerDataType::DATA_TYPE_BFP16},
    {HcclDataType::HCCL_DATA_TYPE_INT128, CheckerDataType::DATA_TYPE_INT128},
    {HcclDataType::HCCL_DATA_TYPE_HIF8, CheckerDataType::DATA_TYPE_HIF8},
    {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, CheckerDataType::DATA_TYPE_FP8E4M3},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, CheckerDataType::DATA_TYPE_FP8E5M2},
    {HcclDataType::HCCL_DATA_TYPE_RESERVED, CheckerDataType::DATA_TYPE_RESERVED}
};

std::map<CheckerDevType, DevType> g_CheckerDevType2HcclDevType = {
    {CheckerDevType::DEV_TYPE_910, DevType::DEV_TYPE_910},
    {CheckerDevType::DEV_TYPE_310P3, DevType::DEV_TYPE_310P3},
    {CheckerDevType::DEV_TYPE_910B, DevType::DEV_TYPE_910B},
    {CheckerDevType::DEV_TYPE_310P1, DevType::DEV_TYPE_310P1},
    {CheckerDevType::DEV_TYPE_910_93, DevType::DEV_TYPE_910_93},
    {CheckerDevType::DEV_TYPE_910_95, DevType::DEV_TYPE_910_95},
    {CheckerDevType::DEV_TYPE_HF, DevType::DEV_TYPE_910_95}
};

std::map<DevType, CheckerDevType> g_HcclDevType2CheckerDevType = {
    {DevType::DEV_TYPE_910, CheckerDevType::DEV_TYPE_910},
    {DevType::DEV_TYPE_310P3, CheckerDevType::DEV_TYPE_310P3},
    {DevType::DEV_TYPE_910B, CheckerDevType::DEV_TYPE_910B},
    {DevType::DEV_TYPE_310P1, CheckerDevType::DEV_TYPE_310P1},
    {DevType::DEV_TYPE_910_93, CheckerDevType::DEV_TYPE_910_93},
    {DevType::DEV_TYPE_NOSOC, CheckerDevType::DEV_TYPE_NOSOC},
    {DevType::DEV_TYPE_910_95, CheckerDevType::DEV_TYPE_910_95},
    {DevType::DEV_TYPE_COUNT, CheckerDevType::DEV_TYPE_COUNT},
};

}
