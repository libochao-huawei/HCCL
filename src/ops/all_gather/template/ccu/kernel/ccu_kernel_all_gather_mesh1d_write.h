/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_WRITE_H
#define HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_WRITE_H

#include <vector>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

class CcuKernelArgAllGatherMesh1DWrite : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgAllGatherMesh1DWrite(uint64_t dimSize, uint32_t rankId, const OpParam &opParam,
                                               const std::vector<std::vector<uint32_t>> &subCommRanks)
        : dimSize_(dimSize), rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks)
    {
    }
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllGatherMesh1DWrite", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                           dimSize_;
    uint32_t                           rankId_;
    OpParam                            opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

// TaskArg: input + N output positions (pre-computed per-rank offsets) + token + sliceSize
class CcuTaskArgAllGatherMesh1DWrite : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllGatherMesh1DWrite(uint64_t inputAddr, uint64_t outputAddr,
                                             uint64_t token, uint64_t sliceSize)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize)
    {
    }
    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
};

// write 模式 AllGather Mesh-1D Kernel：
// 使用 MsWriteNb 将本端 slice 广播至所有 peers。
// 每个 rank 固定使用 bufs[rankId]，写完成后 bufs[R] = rank R 的数据，统一拷贝到各输出位置。
class CcuKernelAllGatherMesh1DWrite : public CcuKernelAlgBase {
public:
    explicit CcuKernelAllGatherMesh1DWrite(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllGatherMesh1DWrite() override {}

    HcclResult            Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void       LoadArgs();
    void       Presync();
    void       Postsync();
    void       DoGroupBroadcastWrite();

    uint64_t     rankSize_{0};
    uint32_t     rankId_{0};

    hcomm::CcuRep::Variable              input_;
    std::vector<hcomm::CcuRep::Variable> output_;  // output_[0..N-1] = per-rank output positions
    hcomm::CcuRep::Variable              token_;
    GroupOpSize                          groupOpSize_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_WRITE_H
