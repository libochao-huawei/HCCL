/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_WRITE_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_WRITE_H

#include <vector>
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

class CcuKernelArgReduceScatterMesh1DWrite : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterMesh1DWrite(uint64_t dimSize, uint32_t rankId, const OpParam &opParam,
                                                   const std::vector<std::vector<uint32_t>> &subCommRanks)
        : dimSize_(dimSize), rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks)
    {
    }
    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterMesh1DWrite", opParam_, subCommRanks_);
        return signature;
    }
    uint64_t                           dimSize_;
    uint32_t                           rankId_;
    OpParam                            opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgReduceScatterMesh1DWrite : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterMesh1DWrite(uint64_t inputAddr, uint64_t outputAddr,
                                                 uint64_t sliceSize, uint64_t token)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), sliceSize_(sliceSize), token_(token)
    {
    }
    uint64_t inputAddr_;  // 已加上 rankId * sliceSize + offset，即本 rank 负责的 slice 起始地址
    uint64_t outputAddr_;
    uint64_t sliceSize_;
    uint64_t token_;
};

// write 模式 ReduceScatter Mesh-1D Kernel：
// 每个 rank 将自己的 input slice 广播给所有 peers（同 AllReduce write 结构），
// 每个 rank 固定使用 bufs[rankId]，接收 N-1 个 peers 的写入，reduce 后写入 output。
class CcuKernelReduceScatterMesh1DWrite : public CcuKernelAlgBase {
public:
    explicit CcuKernelReduceScatterMesh1DWrite(const CcuKernelArg &arg);
    ~CcuKernelReduceScatterMesh1DWrite() override {}

    HcclResult            Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    HcclResult InitResource();
    void       LoadArgs();
    void       Presync();
    void       Postsync();
    void       DoGroupWrite();

    uint64_t     rankSize_{0};
    uint32_t     rankId_{0};
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;

    CcuRep::Variable input_;   // 本 rank 负责的 slice 起始地址
    CcuRep::Variable output_;  // 输出地址
    CcuRep::Variable token_;
    GroupOpSize      groupOpSize_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_WRITE_H
