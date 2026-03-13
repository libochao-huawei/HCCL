/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: coll service stub 头文件
 * Author:
 * Create: 2025-04-11
 */

#ifndef HCCLV2_COLL_SERVICE_STUB_H
#define HCCLV2_COLL_SERVICE_STUB_H

#include "checker_def.h"
#include "dma_mode.h"
#include "topo_meta.h"
#include "ccu_ins_preprocessor.h"
#include "david_alg_config.h"
#include "coll_operator.h"
#include "coll_alg_component.h"

using namespace checker;

namespace Hccl {

class CommunicatorImpl;
class CollServiceStub {
public:
    explicit CollServiceStub(CommunicatorImpl *comm)
        : comm_(comm), ccuInsPreprocessor_(comm)
    {
    }
    HcclResult GenRankSize(TopoMetaParam &topoMetaParam, TopoMeta &topoMate);
    HcclResult Orchestrate(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta, DavidAlgConfig &config);
    shared_ptr<InsQueue> Orchestrate(CollAlgOperator &op, std::string& algName);

    void LoadWithOpBasedMode(CollOperator &op, std::string& algName);
    void LoadWithOffloadMode(CollOperator &op);
    void Init();
    void CollAlgComponentInit();
    CcuInsPreprocessor* GetCcuInsPreprocessor();
    void TransformTask();
private:
    shared_ptr<InsQueue> insQue_;
    CommunicatorImpl *comm_;
    CcuInsPreprocessor ccuInsPreprocessor_;
    shared_ptr<CollAlgComponent> collAlgComponent_;
};


}// namespace Hccl
#endif