/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_ALLGATHER_BATCH_CCU_KERNEL_H
#define HCCL_CUSTOM_ALLGATHER_BATCH_CCU_KERNEL_H

#include <memory>
#include <vector>

#include "common.h"
#include "alg_param.h"
#include "ccu_kernel.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl_allgather_batch {

struct CcuAllGatherBatchItem {
    uint64_t inputAddr = 0;
    uint64_t outputAddr = 0;
    uint64_t token = 0;
    uint64_t offset = 0;
    uint64_t sliceSize = 0;
};

class CcuKernelArgAllGatherBatchMesh1D : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherBatchMesh1D(uint64_t rankSize, uint32_t rankId, uint32_t itemCount)
        : rankSize_(rankSize), rankId_(rankId), itemCount_(itemCount)
    {
        signatureParam_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        signatureParam_.engine = CommEngine::COMM_ENGINE_CCU;
        signatureParam_.opExecuteConfig = ops_hccl::OpExecuteConfig::CCU_MS;
        subCommRanks_.push_back({});
        subCommRanks_[0].reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; ++i) {
            subCommRanks_[0].push_back(i);
        }
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        ops_hccl::GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherBatchMesh1D", signatureParam_, subCommRanks_);
        signature.Append<uint32_t>(itemCount_);
        return signature;
    }

    uint64_t rankSize_ = 0;
    uint32_t rankId_ = 0;
    uint32_t itemCount_ = 0;
    ops_hccl::OpParam signatureParam_{};
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgAllGatherBatchMesh1D : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherBatchMesh1D(uint32_t itemCount, const CcuAllGatherBatchItem *items)
        : itemCount_(itemCount)
    {
        for (uint32_t i = 0; i < itemCount_ && i < MAX_ITEM_COUNT; ++i) {
            items_[i] = items[i];
        }
    }

    uint32_t itemCount_ = 0;
    CcuAllGatherBatchItem items_[MAX_ITEM_COUNT] = {};
};

class CcuKernelAllGatherBatchMesh1D : public ops_hccl::CcuKernelAlgBase {
public:
    explicit CcuKernelAllGatherBatchMesh1D(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherBatchMesh1D() override = default;

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    struct KernelAllGatherBatchItem {
        hcomm::CcuRep::Variable input;
        std::vector<hcomm::CcuRep::Variable> output;
        std::vector<hcomm::CcuRep::Variable> token;
        hcomm::CcuRep::Variable offset;
        GroupOpSize groupOpSize;
    };

    HcclResult InitResource();
    void LoadArgs();
    void PreSync(const KernelAllGatherBatchItem &item);
    void DoAllGather(const KernelAllGatherBatchItem &item);
    void PostSync();

    uint64_t rankSize_ = 0;
    uint32_t rankId_ = 0;
    uint32_t itemCount_ = 0;
    std::vector<KernelAllGatherBatchItem> items_;
    hcomm::CcuRep::LocalAddr src_;
    std::vector<hcomm::CcuRep::RemoteAddr> dst_;
};

} // namespace ops_hccl_allgather_batch

#endif
