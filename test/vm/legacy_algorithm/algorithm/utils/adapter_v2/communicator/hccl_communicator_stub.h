/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: hccl communicator pub header file
 * Author: limengjiao
 * Create: 2024-02-04
 */

#ifndef ADAPTER_V2_COMMUNICATOR_PUB_H
#define ADAPTER_V2_COMMUNICATOR_PUB_H

#include <memory>
#include <mutex>
#include "hccl_params_pub.h"
#include "hccl_types.h"
#include "ccu_ins_preprocessor.h"

namespace Hccl {
class CommunicatorImpl;
class HcclCommunicator {
public:
    explicit HcclCommunicator(const CommParams &commParams);
    HcclCommunicator(const CommParams &commParams, const HcclCommConfig *config);
    ~HcclCommunicator();

    HcclResult Init(const std::string &ranktableM, std::string& topoPath);
    void DeInit() const;

    HcclResult LoadOpbasedCollOp(CollOpParams &opParams, std::string& algName);
    HcclResult LoadOffloadCollOp(std::string &opTag, CollOpParams &opParams);

    // MC2 流程专用
    HcclResult GetRankSize(uint32_t *rankSize);
    HcclResult GetRankId(uint32_t &rankId);
    CcuInsPreprocessor* GetCcuInsPreprocessor();
    void TransformTask();
private:
    CommParams                        commParams;
    HcclCommConfig                    config{};
    std::unique_ptr<CommunicatorImpl> pimpl;
    CommunicatorImpl *GetCommImpl();
    std::mutex serialMutex;
};

} // namespace Hccl

#endif