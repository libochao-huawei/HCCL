/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr u64 AG_2D_SMALL_DATA_SIZE = 1024 * 1024;

SelectorStatus AllGatherAutoSelector::SelectCcuMsAlgo(
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }

    if (topoInfo->topoLevelNums > 1) {
        // if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        //     selectAlgName = "CcuAllGatherParallelMeshNHR";
        // } else {
        //     HCCL_WARNING("[Algo][AllGatherAutoSelector] levelNum > 1 is not supported yet for 2d ccu_ms mode.");
        //     return SelectorStatus::NOT_MATCH;
        // }
        HCCL_WARNING("[Algo][AllGatherAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode");
        return SelectorStatus::NOT_MATCH;
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "CcuAllGatherMesh1D";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                selectAlgName = "CcuAllGatherMesh1D";
            } else {
                HCCL_WARNING("[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode",
                    topoInfo->level0Topo);
                return SelectorStatus::NOT_MATCH;
            }
        } else {
            HCCL_WARNING("[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    }
    
    // if (IsDefaultAlg(levle0Algo) || levle0Algo == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
    //     HCCL_DEBUG("[AllGatherAutoSelector][%s] SelectMeshAlgo", __func__);
    //     return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
    // } else {
    //     HCCL_WARNING("[Algo][AllGatherAutoSelector] algo[%u] is not supported yet for ccu_ms mode, reset to default.",
    //                  levle0Algo);
    //     return SelectorStatus::NOT_MATCH;
    // }
    HCCL_INFO("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectMeshAlgo(TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam,
                                                     std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (IsSmallData(opParam.inputSize)) {
            selectAlgName = "CcuAllGatherMesh1D";
        } else {
            selectAlgName = "CcuAllGatherMesh1D";
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectCcuScheduleAlgo(
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "CcuAllGatherNHR1DMem2Mem";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[Algo][AllGatherAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else if (topoInfo->deviceNumPerModule > 1) {
                selectAlgName = "CcuAllGatherParallelMesh1DNHR";
            } else {
                selectAlgName = "CcuAllGatherNHR1DMem2Mem";
            }
        } else {
            HCCL_WARNING("[Algo][AllGatherAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                         topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[Algo][AllGatherAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuAllGatherMesh1DMem2Mem";
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                selectAlgName = "CcuAllGatherMesh1DMem2Mem";
            } else {
                selectAlgName = "CcuAllGatherParallelMesh1DNHR";
            }
        } else {
            HCCL_WARNING("[Algo][AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    }

    // if ((IsDefaultAlg(levle0Algo) || levle0Algo == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) &&
    //     (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
    //     selectAlgName = "CcuAllGatherMesh1DMem2Mem";
    //     return SelectorStatus::MATCH;
    // } else if ((IsDefaultAlg(levle0Algo) || (levle0Algo == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH))) {
    //     selectAlgName = "CcuAllGatherMesh2DMem2Mem";
    //     return SelectorStatus::MATCH;
    // } else {
    //     HCCL_WARNING(
    //         "[Algo][AllGatherAutoSelector] algo[%u] is not supported yet for ccu_schedule mode, reset to default.",
    //         levle0Algo);
    //     return SelectorStatus::NOT_MATCH;
    // }
    HCCL_INFO("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAicpuAlgo(
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
    HCCL_INFO("[AllGatherAutoSelector][SelectAicpuAlgo] topoLevelNums=[%d], deviceNumPerModule=[%d], level0Topo=[%d]",
              topoInfo->topoLevelNums, topoInfo->deviceNumPerModule, topoInfo->level0Topo);
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->Level1Nhr) {
            selectAlgName = "InsAllGatherNHR";
        // } else if (topoInfo->Level0Nhr) {

        } else if (topoInfo->deviceNumPerModule == 1) {
            selectAlgName = "InsAllGatherNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllGatherParallelMesh1DNHR";
        } else {
            HCCL_WARNING("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllGatherMesh1D";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                selectAlgName = "InsAllGatherMesh1D";
            } else {
                selectAlgName = "InsAllGatherParallelMesh1DNHR";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAllGatherNHR";
        } else {
            HCCL_WARNING("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAivAlgo(
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);

    // if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
    selectAlgName = "AivAllGatherMesh1D";
    // } else {
    //     HCCL_WARNING("[AllGatherAutoSelector] topo not match for aiv algo");
    //     return SelectorStatus::NOT_MATCH;
    // }
    HCCL_INFO("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectDPUAlgo(
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->deviceNumPerModule == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "InsAllGatherMeshNhr";
            return SelectorStatus::MATCH;
        }
    }

    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER, 18, AllGatherAutoSelector);

}  // namespace ops_hccl
