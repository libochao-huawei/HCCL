/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 2.0适配器对外提供的编排接口
 * Author: yinding
 * Create: 2025-06-14
 */

#ifndef HCCL_ADAPTER_V2_INTERFACE_H
#define HCCL_ADAPTER_V2_INTERFACE_H

#include "task_def.h"

namespace HcclSim {
    TaskNodePtr GetCcuTaskHead(TaskNodePtr node);
    HcclResult CopyCcuSubGraph(TaskStub *originCcu, TaskStub **newCcu);
}
#endif