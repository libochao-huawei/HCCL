/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H
#define HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H

#include <array>
#include <ios>
#include <vector>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllreduceMesh1D2DieOneShot : CcuKernelArgBase {
    uint64_t dimSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    bool rmtReduceWithMyRank_;
};

struct BufferedReduceVar {
    ccu::LocalAddr loopDst[2];
    std::array<std::vector<ccu::RemoteAddr>, 2> loopRemoteSrc;
    std::array<std::vector<ccu::LocalAddr>, 2> loopLocalSrc;
    ccu::Variable loopLen[2];
    ccu::Variable loopLenExp[2];
};

class CcuKernelAllreduceMesh1D2DieOneShot : public CcuKernelCtxBase {
public:
    explicit CcuKernelAllreduceMesh1D2DieOneShot(const CcuKernelArgAllreduceMesh1D2DieOneShot *arg);
    ~CcuKernelAllreduceMesh1D2DieOneShot() = default;

    CcuResult Algorithm();

private:
    CcuResult InitResource();
    CcuResult LoadArgs();
    CcuResult PreSync();
    CcuResult PostSync(uint32_t signalIndex);
    CcuResult MissionSync(uint32_t maskIndex);
    CcuResult DoRmtReduce();
    std::string GetLoopBlockTag(std::string loopType, int32_t index) const;
    CcuResult DoLocalReduce();
    CcuResult ReduceLoopGroup(ccu::LocalAddr outDstOrg, std::vector<ccu::RemoteAddr> remoteSrcOrg,
        std::vector<ccu::LocalAddr> localSrcOrg, GroupOpSizeVars goSize, HcclDataType dataType,
        HcclDataType outputDataType, HcclReduceOp opType, const std::string &loopName);
    CcuResult CreateReduceLoop(BufferedReduceVar &var, const std::string &loopName, uint32_t remoteCount,
        uint32_t localCount, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
    void SetReduceLoopInput(BufferedReduceVar &var, uint32_t index, const ccu::LocalAddr &dst,
        const std::vector<ccu::RemoteAddr> &remoteSrc, const std::vector<ccu::LocalAddr> &localSrc,
        const ccu::Variable &len, const ccu::Variable &lenExp);

    const CcuKernelArgAllreduceMesh1D2DieOneShot *kernelArg_;
    bool rmtReduceWithMyRank_;
    uint32_t myRankId_;
    uint32_t rankSize_;

    uint32_t rmtReduceRankNum_;

    HcclDataType dataType_;
    HcclDataType outputDataType_;
    HcclReduceOp reduceOp_;

    uint32_t missionSyncMybit_;
    uint32_t missionSyncWaitBit_;

    ccu::Variable myInput_;
    ccu::Variable myOutput_;
    ccu::Variable myScratch_;
    ccu::Variable myToken_;

    std::vector<ccu::Variable> input_;
    std::vector<ccu::Variable> remoteToken_;

    ccu::Variable scratchBaseOffset0_;
    ccu::Variable scratchBaseOffset1_;

    ccu::Variable localReduceSliceOffset0_;
    ccu::Variable localReduceSliceOffset1_;

    GroupOpSizeVars rmtReduceGoSize_;
    GroupOpSizeVars localReduceGoSize0_;
    GroupOpSizeVars localReduceGoSize1_;
    std::vector<ChannelHandle> channels_;
};

CcuResult CcuAllreduceMesh1D2DieOneShotKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H
