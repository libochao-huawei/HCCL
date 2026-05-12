/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <vector>
#include <hccl/hcomm_primitives.h>
#include "ccu_control_api.h"
#include "ccu_kernel.h"

#include "log.h"
#include "common.h"


using namespace ops_hccl_ag;

namespace ops_hccl_ag {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t UB_MAX_DATA_SIZE = 256*1024*1024;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

HcclResult ExecOp(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] start");

    uint32_t dataTypeSize = SIZE_TABLE[param.dataType];
    uint64_t dataSize = param.count * dataTypeSize;
    uint64_t count = param.count;

    if (count == 0) {
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem] DataCount == 0, ExecOp Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem] RankSize == 1, ExecOp Run Ends.");
        return HCCL_SUCCESS;
    }

    uint64_t maxDataSizePerLoop = UB_MAX_DATA_SIZE;
    uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    uint64_t loopCount = count / maxDataCountPerLoop + static_cast<uint64_t>(count % maxDataCountPerLoop != 0);
    uint64_t processedDataCount = 0;

    uint64_t token = 0;
    uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    if (param.inputPtr != nullptr) {
        token = hcomm::CcuRep::GetTokenInfo(baseInputAddr, static_cast<uint64_t>(dataSize));
    } else if (param.outputPtr != nullptr) {
        token = hcomm::CcuRep::GetTokenInfo(baseOutputAddr, static_cast<uint64_t>(dataSize));
    } 

    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = dataSize;

    for (uint64_t loop = 0; loop < loopCount; loop++) {
        uint64_t sliceCount = std::min(maxDataCountPerLoop, count - loop * maxDataCountPerLoop);
        uint64_t sliceSize = sliceCount * dataTypeSize;

        uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr) + processedDataCount * dataTypeSize;
        uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr) + processedDataCount * dataTypeSize;

        uint64_t currentRankSliceInputOffset = inputSliceStride * param.myRank;
        uint64_t currentRankSliceOutputOffset = outputSliceStride * param.myRank;

        std::vector<uint64_t> taskArgs = {
            inputAddr,
            outputAddr,
            token,
            currentRankSliceInputOffset,
            currentRankSliceOutputOffset,
            sliceSize
        };

        HCCL_INFO("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
                   "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
                   "sliceSize[%llu], loop[%llu], loopCount[%llu]",
                   inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, sliceSize, loop, loopCount);

        CcuResult launchRet = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0],
                                                    taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }

        processedDataCount += sliceCount;
    }

    HCCL_DEBUG("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] end");
    return HCCL_SUCCESS;
}
}