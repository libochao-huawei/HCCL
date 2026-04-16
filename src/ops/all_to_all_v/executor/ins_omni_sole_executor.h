/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_OMNI_SOLE_EXECUTOR_H
#define HCCLV2_INS_OMNI_SOLE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"
#include "ccu_temp_omni.h"
#include "aiv_temp_omni.h"

namespace ops_hccl {
struct ResRequest {
    uint16_t opCode : 5;    
    uint16_t slave : 5;  
    uint16_t notifyNumOnMainThread : 5;   
    uint16_t notifyNumPerThread : 5;    
    uint16_t netLayerNum : 2;    
    uint16_t chanCount : 8;    
    uint16_t reser : 2;   
};

struct CtrlOp {
    uint16_t opCode : 5;  
    uint16_t netlayerId : 2;  
    uint16_t linkProto : 3;  
    uint16_t sliceNum : 10;  
    uint16_t srcSliceNum : 4;  
    uint16_t dstSliceNum : 4;  
    uint16_t notifyFlag : 1;  
    uint16_t notifyThread : 4;  
    uint16_t waitFlag : 1;  
    uint16_t waitThread : 4;  
    uint16_t threadIdx : 5;  
    uint16_t reduceType : 2; 
    uint16_t inputDataType : 4; 
    uint16_t outputDataType : 4; 
    uint16_t instructionId : 10; 
    uint16_t reser : 1; 
};

struct SrcSlice {
    uint16_t bufferType : 2;  
    uint16_t sliceIdx : 10;  
    uint16_t rankId : 10;  
    uint16_t reser : 10;  
};

struct DstSlice {
    uint16_t bufferType : 2;  
    uint16_t sliceIdx : 10;  
    uint16_t rankId : 10;  
    uint16_t reser : 10;  
};

struct Channel {
    uint16_t netlayerId : 5;  
    uint16_t localRank : 10;  
    uint16_t remoteRank : 10;  
    uint16_t linkProto : 3;  
    uint16_t reser : 5;  
};






template <typename AlgTopoMatch, typename InsAlgTemplate> class InsOmniSoleExecutor : public InsCollAlgBase {
public:
    explicit InsOmniSoleExecutor();
    ~InsOmniSoleExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;
    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
    

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo);
    HcclResult ParseXmlInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo);

    std::vector<std::vector<std::vector<u32>>> algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;                 // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

private:
    XmlInfo xmlInfo_;
};
}

#endif
