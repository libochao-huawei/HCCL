/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 2.0适配器对外提供的编排接口
 * Author: yinding
 * Create: 2025-06-14
 */

#ifndef HCCL_ADAPTER_V2_INTERFACE_H
#define HCCL_ADAPTER_V2_INTERFACE_H

#include "hccl_types.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "dma_mode.h"
#include "david_alg_config.h"
#include "task_def.h"

using namespace checker;

namespace Hccl {
using TaskStubPtr = TaskStub*;
using Ori2NewNodeMap = std::map<TaskNodePtr, TaskNodePtr>;
using CcuOri2NewNodeMap = std::map<TaskStubPtr, Ori2NewNodeMap>;

class TaskQuesGeneratorV2 {
public:
    TaskQuesGeneratorV2() = default;
    ~TaskQuesGeneratorV2();
    HcclResult Run(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta, DavidAlgConfig &config);
};

// 将CCU的task序列转换为CCU子图
HcclResult GenCcuGraph(TaskNode* dummyStart);
TaskNode* UpdateNodeForCcuGraph(TaskNode *node,  std::set<TaskNode *>& simulatedNodes);
TaskNodePtr GetCcuTaskHead(TaskNodePtr node);
// 拷贝ccu子图节点
HcclResult CopyCcuSubGraphNode(TaskStub *originCcu, TaskStub **newCcu,
    std::vector<std::pair<TaskStubPtr, TaskStubPtr>> &ccuGraphs, std::vector<CcuOri2NewNodeMap> &AllOri2NewNodeMap);
// 拷贝ccu子图连接关系
HcclResult CopyCcuSubGraphConnection(std::vector<std::pair<TaskStubPtr, TaskStubPtr>> &ccuGraphs,
    const std::vector<CcuOri2NewNodeMap> &AllOri2NewNodeMap);
}

#endif