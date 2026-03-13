/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 用于1.0和2.0不同版本的兼容性处理
 * Author: huangweihao
 * Create: 2025-02-10
 */
#ifndef HCCLV2_TRANSFORM_FUNS_TYPE_H
#define HCCLV2_TRANSFORM_FUNS_TYPE_H

#include <map>
#include "checker_def.h"
#include "op_type.h"
#include "reduce_op.h"
#include "data_type.h"
#include "port.h"
#include "op_mode.h"
#include "proto_stub.h"
#include "buffer_type.h"
#include "checker_buffer_type.h"
#include "dev_type.h"

using namespace checker;

namespace Hccl {

extern  std::map<CheckerOpType, OpType> g_CheckerOpType2OpType_aicpu;
extern  std::map<OpType, CheckerOpType> g_OpType2CheckerOpType_aicpu;
extern  std::map<CheckerReduceOp, ReduceOp> g_CheckerReduceOp2ReduceOp_aicpu;
extern  std::map<ReduceOp, CheckerReduceOp> g_ReduceOp2CheckerReduceOp_aicpu;
extern  std::map<CheckerDataType, DataType> g_CheckerDataType2DataType_aicpu;
extern  std::map<DataType, CheckerDataType> g_DataType2CheckerDataType_aicpu;
extern  std::map<LinkProtoType, LinkProtoStub> g_LinkProtoType2LinkProtoStub_aicpu;
extern  std::map<CheckerOpMode, OpMode> g_CheckerOpMode2OpMode_aicpu;
extern  std::map<Hccl::BufferType, checker::BufferType> g_HcclBufferType2CheckerBufferType_aicpu;
extern  std::map<CheckerDevType, Hccl::DevType> g_CheckerDevType2HcclDevType_aicpu;

extern  std::map<uint16_t, CheckerReduceOp> g_ReduceOp2CheckerReduceOp_ccu;
extern  std::map<uint16_t, CheckerDataType> g_DataType2CheckerDataType_ccu;

} // namespace Hccl

#endif // HCCLV2_TRANSFORM_FUNS_TYPE_H