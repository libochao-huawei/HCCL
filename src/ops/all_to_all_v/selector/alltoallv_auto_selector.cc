/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

SelectorStatus AlltoAllVAutoSelector::SelectCcuScheduleAlgo(TopoInfoWithNetLayerDetails* topoInfo,
                                                    OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        HCCL_DEBUG("[AlltoAllVAutoSelector] levelNum > 1 is not supported yet for ccu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuAllToAllVMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_DEBUG("[AlltoAllVAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuAlltoAllVMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_DEBUG("[AlltoAllVAutoSelector] CheckMeshNumEqualToClosNum failed."),
                    SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= 4)) { // 同一组4P，走并发算法
            selectAlgName = "CcuAllToAllVMesh1DConcurrent";
        } else {
            HCCL_DEBUG("[AlltoAllVAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        HCCL_DEBUG("[AlltoAllVAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAicpuAlgo(TopoInfoWithNetLayerDetails* topoInfo,
                                                      OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAlltoAllVMesh1D";
        } else {
            HCCL_ERROR("[AlltoAllVAutoSelector] hccl algo no match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAlltoAllVMesh1D";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            bool isMeshNumEqualToClosNum = false;
            CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                        HCCL_ERROR("[AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
                        SelectorStatus::NOT_MATCH);
            if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= 4)) { // 同一组4P，走并发算法
                selectAlgName = "InsAllToAllVMesh1DConcurrent";
            } else {
                selectAlgName = "InsAlltoAllVMesh1D";
            }
        } else {
            HCCL_ERROR("[AlltoAllVAutoSelector] hccl algo no match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAivAlgo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                   std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;

    selectAlgName = "AivAlltoAllVMesh1D";
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALLV, 18, AlltoAllVAutoSelector);
} // namespace Hccl
