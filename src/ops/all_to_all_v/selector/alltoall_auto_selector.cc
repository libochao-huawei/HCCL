/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
SelectorStatus AlltoAllAutoSelector::SelectCcuMsAlgo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    HCCL_WARNING("[Algo][AlltoAllAutoSelector] is not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectCcuScheduleAlgo(TopoInfoWithNetLayerDetails* topoInfo,
                                                    OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][AlltoAllAutoSelector] levelNum > 1 is not supported yet for ccu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    }

    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT; 
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "CcuAlltoAllMesh1D";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            selectAlgName = "CcuAlltoAllMesh1D";
        } else {
            HCCL_WARNING("[Algo][AlltoAllAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[Algo][AlltoAllAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[Algo][AlltoAllAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    // if (IsDefaultAlg(levle0Algo) || levle0Algo ==  HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
    //        HCCL_INFO("Select CcuAlltoAllMesh1D!");
    //     selectAlgName = "CcuAlltoAllMesh1D";
    // } else {
    //     HCCL_ERROR("[Algo][AlltoAllAutoSelector] algo[%u] is not supported yet for ccu_schedule mode, reset to default.", levle0Algo);
    //     return SelectorStatus::NOT_MATCH;
    // }
    HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAicpuAlgo(TopoInfoWithNetLayerDetails* topoInfo,
                                                      OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[AlltoAll] hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos.at(0), algos.at(1), algos.at(2), algos.at(3));

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAlltoAllMesh1D";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS ||
            topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgName = "InsAlltoAllMesh1D";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());

    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAivAlgo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                       const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                       std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void) topoInfo;
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[AlltoAll] hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos.at(0), algos.at(1), algos.at(2), algos.at(3));

    selectAlgName = "AivAlltoAllMesh1D";

    HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALL, 18, AlltoAllAutoSelector);
} // namespace Hccl
