/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: transform function 头文件
 * Author:
 * Create: 2025-04-11
 */
#ifndef TRANSFORM_FUNSUNCTION_H
#define TRANSFORM_FUNSUNCTION_H

#include <map>
#include <vector>
#include <memory>
#include <queue>
#include <unordered_map>

#include "type_conversion.h"
#include "coll_operator.h"
#include "coll_alg_params.h"
#include "instruction.h"
#include "checker_data_slice.h"
#include "coll_service_stub.h"

using namespace checker;

namespace Hccl {

    HcclResult GenCollAlgOperator(CollAlgOperator &op, CheckerOpParam &checkerOpParam);
    HcclResult GenCollAlgParams(CollAlgParams &params, CheckerOpParam &checkerOpParam, DavidAlgConfig &config);
    HcclResult HcclDataSlice2CheckerDataSlice(Hccl::DataSlice &dataSlice, checker::DataSlice &checkerDataSlice);
    HcclResult TransformIns2Task(const Instruction &ins, RankId rankId, QId qId);


}// namespace Hccl

#endif
