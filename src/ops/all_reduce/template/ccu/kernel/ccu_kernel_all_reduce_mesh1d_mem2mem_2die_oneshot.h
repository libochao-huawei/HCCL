/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT

#include <array>
#include <ios>
#include <vector>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot : CcuKernelArgBase {
    uint64_t dimSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<uint32_t> kernelRanks_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    bool rmtReduceWithMyRank_;
};

struct LocalReduceVar {
    ccu::LocalAddr loopDst[2];
    std::array<std::vector<ccu::LocalAddr>, 2> loopSrc;
    ccu::Variable loopLen[2];
    ccu::Variable loopLenExp[2];
};

class CcuKernelAllReduceMesh1DMem2Mem2DieOneShot : public CcuKernelCtxBase {
public:
    explicit CcuKernelAllReduceMesh1DMem2Mem2DieOneShot(
        const CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *arg);
    ~CcuKernelAllReduceMesh1DMem2Mem2DieOneShot() = default;

    CcuResult Algorithm();

private:
    CcuResult InitResource();
    CcuResult LoadArgs();
    void InitDerivedArgs();
    CcuResult PreSync();
    CcuResult PostSync();
    CcuResult RmtReduce();
    CcuResult DoLocalReduce();
    CcuResult ReduceLoopGroup(ccu::LocalAddr outDstOrg, std::vector<ccu::LocalAddr> srcOrg,
        GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
        const std::string &loopName);
    std::string GetLoopBlockTag(std::string loopType, int32_t index) const;
    CcuResult CreateReduceLoop(LocalReduceVar &var, uint32_t size, HcclDataType dataType,
        HcclDataType outputDataType, HcclReduceOp opType, const std::string &loopName);
    void SetReduceLoopInput(LocalReduceVar &var, uint32_t index, const ccu::LocalAddr &dst,
        const std::vector<ccu::LocalAddr> &src, const ccu::Variable &len, const ccu::Variable &lenExp);
    CcuResult MissionSync(uint32_t maskIndex);

    const CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *kernelArg_;
    uint64_t rankSize_{0};
    uint32_t rankId_{0};
    std::vector<ChannelHandle> channels_;

    bool rmtReduceWithMyRank_;
    uint32_t rmtReduceRankNum_;

    uint32_t missionSyncMybit_;
    uint32_t missionSyncWaitBit_;

    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;

    ccu::Variable myInput_;
    ccu::Variable myOutput_;
    ccu::Variable myScratch_;
    ccu::Variable myToken_;
    std::vector<ccu::Variable> peerInput_;
    std::vector<ccu::Variable> peerToken_;

    ccu::Variable scratchBaseOffset0_;
    ccu::Variable scratchBaseOffset1_;
    ccu::Variable normalSliceSize_;
    ccu::Variable localReduceSliceOffset0_;
    ccu::Variable localReduceSliceOffset1_;

    GroupOpSizeVars localReduceGoSize_;
    GroupOpSizeVars localReduceGoSize0_;
    GroupOpSizeVars localReduceGoSize1_;

    ccu::Event event_;
};

CcuResult CcuAllReduceMesh1DMem2Mem2DieOneShotKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT
