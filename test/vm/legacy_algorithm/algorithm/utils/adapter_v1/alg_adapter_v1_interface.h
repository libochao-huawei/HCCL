/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 1.0适配器对外提供的编排接口
 * Author: huangweihao
 * Create: 2025-05-06
 */

#ifndef HCCL_ADAPTER_V1_INTERFACE_H
#define HCCL_ADAPTER_V1_INTERFACE_H

#include "hccl_types.h"
#include "checker_def.h"
#include "topo_meta.h"

using namespace checker;
namespace hccl {

class TaskQuesGenerator {
public:
    TaskQuesGenerator() = default;
    ~TaskQuesGenerator();
    HcclResult Run(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta);
};

}

#endif