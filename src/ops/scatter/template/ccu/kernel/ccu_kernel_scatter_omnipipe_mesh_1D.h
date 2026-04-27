/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_SCATTER_TEMPLATE_CCU_KERNEL_CCU_KERNEL_SCATTER_OMNIPIPE_MESH_1D_H_  
#define OPS_HCCL_SRC_OPS_SCATTER_TEMPLATE_CCU_KERNEL_CCU_KERNEL_SCATTER_OMNIPIPE_MESH_1D_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgScatterOmniPipeMesh1D : public hcomm::CcuKernelArg {
public:
    CcuKernelArgScatterOmniPipeMesh1D(uint64_t dimSize, uint32_t rankId, const OpParam &opParam,
        const std::vector<std::vector<uint32_t>> &subCommRanks,
        bool isSameAxisAsRoot)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          isSameAxisAsRoot(isSameAxisAsRoot)
    {
        HCCL_DEBUG("[%s]dimSize: %lu, rankId: %u, isSameAxisAsRoot: %d, dataType: %d",
                   __func__, dimSize_, rankId_, isSameAxisAsRoot, opParam.DataDes.dataType);
    }
    ~CcuKernelArgScatterOmniPipeMesh1D() override = default;
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgScatterOmniPipeMesh1D", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t dimSize_ = 0;
    uint32_t rankId_ = 0;
    OpParam opParam_;
    bool isSameAxisAsRoot = false;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgScatterOmniPipeMesh1D : public hcomm::CcuTaskArg {
public:
    CcuTaskArgScatterOmniPipeMesh1D(uint64_t inputAddr, uint64_t outputAddr, uint64_t sliceSize,
        uint64_t offset, uint64_t token, uint64_t localCopyFlag, uint64_t inputSliceStride,
        uint64_t outputSliceStride, uint64_t inputOmniPipeSliceStride, uint64_t outputOmniPipeSliceStride, uint32_t stepIndex)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), offset_(offset), token_(token),
          localCopyFlag_(localCopyFlag), inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),
          inputOmniPipeSliceStride_(inputOmniPipeSliceStride), outputOmniPipeSliceStride_(outputOmniPipeSliceStride), stepIndex_(stepIndex) {}
    ~CcuTaskArgScatterOmniPipeMesh1D() override = default;
    uint64_t inputAddr_ = 0;
    uint64_t outputAddr_ = 0;
    uint64_t sliceSize_ = 0;
    uint64_t offset_ = 0;
    uint64_t token_ = 0;
    uint64_t localCopyFlag_ = 0;
    uint64_t inputSliceStride_ = 0;
    uint64_t outputSliceStride_ = 0;
    uint64_t inputOmniPipeSliceStride_ = 0;
    uint64_t outputOmniPipeSliceStride_ = 0;
    uint32_t stepIndex_ = 0; // 当前通信步骤索引
};

class CcuKernelScatterOmniPipeMesh1D : public CcuKernelAlgBase {
public:
    explicit CcuKernelScatterOmniPipeMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelScatterOmniPipeMesh1D() override = default;
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;
    

private:
    void PreSync();
    void PostSync();
    void LoadArgs();
    HcclResult InitResource();
    void DoScatter();

    // 拓扑与身份信息
    uint64_t rankSize_ = 0;
    uint32_t rankId_ = 0;  // 用户rank号在子通信域中的indexID
    uint32_t rootId_ = 0;  // root的rank号
    bool isSameAxisAsRoot_ = false; // 是否为root的同轴线节点
    uint32_t currentStep_ = 0; // 当前通信步骤索引
    uint32_t userRank_ = 0; // 用户rank号

    // 数据与通信资源
    HcclDataType dataType_ = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    std::vector<ChannelHandle> channels_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    std::vector<hcomm::CcuRep::LocalAddr> inputMem_;
    std::vector<hcomm::CcuRep::RemoteAddr> outputMem_;

    // 切片与步长参数
    hcomm::CcuRep::Variable input_;
    hcomm::CcuRep::Variable offset_;
    hcomm::CcuRep::Variable localCopyFlag_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable inputSliceStride_;
    hcomm::CcuRep::Variable outputSliceStride_;
    hcomm::CcuRep::Variable inputOmniPipeSliceStride_;
    hcomm::CcuRep::Variable outputOmniPipeSliceStride_;

    // 事件
    hcomm::CcuRep::CompletedEvent event_;
};

} // namespace ops_hccl

#endif // OPS_HCCL_SRC_OPS_SCATTER_TEMPLATE_CCU_KERNEL_CCU_KERNEL_SCATTER_OMNIPIPE_MESH_1D_H_
