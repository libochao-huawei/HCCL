/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_GEN_TASK_NODE_UTILS_H
#define HCCLV1_GEN_TASK_NODE_UTILS_H

#include <map>
#include <set>
#include <vector>
#include <memory>

#include "log.h"
#include "task_ccu.h"
#include "llt_common.h"
#include "task_queue_stub.h"
using namespace checker;
namespace hccl {

class GenCcuTaskNodeGraphBase {
public:
    virtual ~GenCcuTaskNodeGraphBase();

public:
    std::vector<std::shared_ptr<TaskStub>> taskCCU;
    std::vector<std::shared_ptr<TaskNode>> ccuNode;
    std::shared_ptr<SingleRankTaskQueues> taskQeueus;
    TaskNodePtr GetCcuTaskHead(TaskNodePtr node);
    void PrintRankGraph(RankId rankId);
    void Init(TaskStubCcuGraph *curCcuTask, uint32_t rankSize, uint32_t queNum);

protected:
    TaskNodePtr head = nullptr;

    TaskNodePtr CreateHeadNode();
    TaskNodePtr CreateCcuHeadNode(RankId rankId, uint32_t queueId);
    TaskNodePtr AddCcuLocalCopy(RankId rankId, uint32_t queueId, DataSlice srcSlice, DataSlice dstSlice, TaskStubCcuGraph *curCcuTask);
    TaskNodePtr AddCcuWrite(RankId rankId, RankId rmtRankId, uint32_t queueId, DataSlice srcSlice, DataSlice dstSlice, TaskStubCcuGraph *curCcuTask);
    TaskNodePtr AddCcuRead(RankId rankId, RankId rmtRankId, uint32_t queueId, DataSlice srcSlice, DataSlice dstSlice, TaskStubCcuGraph *curCcuTask);
    TaskNodePtr AddCcuLocalWait(RankId rankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask);
    TaskNodePtr AddCcuLocalPost(RankId rankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask);
    TaskNodePtr AddCcuWait(RankId rankId, RankId rmtRankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask);
    TaskNodePtr AddCcuPost(RankId rankId, RankId rmtRankId, uint32_t queueId, uint32_t topicId, TaskStubCcuGraph *curCcuTask);

    void LinkNode(TaskNodePtr parent, TaskNodePtr node);
    void CreateCcuEndNode(RankId rankId, TaskNodePtr &node, TaskStubCcuGraph *curCcuTask);

private:
    std::vector<TaskNodePtr> toDeleteTaskNodeResource_;
};

void PrintGraphRevamp(TaskNodePtr head);

}

#endif