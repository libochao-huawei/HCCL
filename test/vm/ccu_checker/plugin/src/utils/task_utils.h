/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_UTILS_H
#define TASK_UTILS_H

#include "task_def.h"
#include "task_meta_defs.h"
#include "storage_manager.h"
#include <map>
#include <vector>

namespace HcclSim {

std::shared_ptr<TaskStub> ConvertTask(const HcclSim::StorageManager& storage, HcclTaskMetaData hcclTask);
void ConvertTaskQueue(AllRankTaskQueues& allRankTaskQueues);

} // namespace HcclSim

#endif