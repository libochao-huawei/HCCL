/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr u64 REDUCE_AICPU_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;

SelectorStatus ReduceAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
<<<<<<< HEAD
    (void)configAlgMap;
=======
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[ReduceAutoSelector] layerNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
<<<<<<< HEAD
        HCCL_WARNING("[ReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
=======
        HCCL_WARNING("[Algo][ReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
            opParam.DataDes.dataType), SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING(
            "[ReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.", opParam.reduceType),
        SelectorStatus::NOT_MATCH);

<<<<<<< HEAD
    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
=======
    if (isInt64Type(opParam.DataDes.dataType)) {
        HCCL_WARNING("[Algo][ReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
<<<<<<< HEAD
        HCCL_WARNING("[ReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
=======
        HCCL_WARNING("[Algo][ReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
        return SelectorStatus::NOT_MATCH;
    }
    SelectorStatus ret = SelectMeshAlgoCcums(topoInfo, opParam, selectAlgName);
    if (ret == SelectorStatus::NOT_MATCH) {
        return ret;
    }
<<<<<<< HEAD
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
=======
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectMeshAlgoCcums(
<<<<<<< HEAD
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, std::string &selectAlgName) const
=======
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, std::string &selectAlgName) const
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
<<<<<<< HEAD
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu_ms mode.");
            return SelectorStatus::NOT_MATCH;
        } else if(dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
=======
        if (Is2DieFullMesh()) {
            HCCL_WARNING("[Algo][ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu_ms mode.");
            return SelectorStatus::NOT_MATCH;
        } else if(dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            HCCL_INFO("[Algo][ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            if(dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
<<<<<<< HEAD
                HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
=======
                HCCL_INFO("[Algo][ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuReduceMesh1D";
            }
        } else { // MS 不支持
<<<<<<< HEAD
            HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
=======
            HCCL_WARNING("[Algo][ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
<<<<<<< HEAD
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
=======
        HCCL_WARNING("[Algo][ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[Algo][ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
<<<<<<< HEAD
    (void)configAlgMap;
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING( "[ReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType), SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
=======
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING( "[Algo][ReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType), SelectorStatus::NOT_MATCH);

    if (isInt64Type(opParam.DataDes.dataType)) {
        HCCL_WARNING("[Algo][ReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) == 1) {
                // 每框出 1 卡
                selectAlgName = "CcuReduceNHR1DMem2Mem";
            } else if (topoInfo->is2DieFullMesh) {
<<<<<<< HEAD
                HCCL_WARNING("[ReduceAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
=======
                HCCL_WARNING("[Algo][ReduceAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuReduceParallelMesh1DNHR";
            }
        } else {
            HCCL_WARNING("[SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoCcuSchedule(topoInfo, opParam, selectAlgName);
    }
<<<<<<< HEAD
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
=======
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectMeshAlgoCcuSchedule(
    TopoInfoWithNetLayerDetails *topoInfo, OpParam &opParam, std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
<<<<<<< HEAD
            HCCL_WARNING("[ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
=======
            HCCL_WARNING("[Algo][ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            HCCL_INFO("[Algo][ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceMesh1DMem2Mem";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
<<<<<<< HEAD
                HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
=======
                HCCL_INFO("[Algo][ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuReduceMesh1DMem2Mem";
            }
        } else {
            selectAlgName = "CcuReduceNHR1DMem2Mem";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
<<<<<<< HEAD
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
=======
        HCCL_WARNING("[Algo][ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[Algo][ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
<<<<<<< HEAD
    (void)configAlgMap;
<<<<<<< HEAD
    if (Is64BitDataType(opParam.DataDes.dataType)) {
=======
    if (isInt64Type(opParam.DataDes.dataType)) {
>>>>>>> a62c957... reduce/reducescatter/reducescatterv selector fix 2
=======
    if (isInt64Type(opParam.DataDes.dataType)) {
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
        HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
        return SelectorStatus::NOT_MATCH;
    }
    if (topoInfo->topoLevelNums > 1) {
        CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
<<<<<<< HEAD
            HCCL_WARNING("[ReduceAutoSelector] ReduceOp[%d] is not supported yet for aicpu levelNum > 1.",
                opParam.reduceType), SelectorStatus::NOT_MATCH);

        CHK_PRT_RET(Is64BitDataType(opParam.DataDes.dataType),
            HCCL_WARNING("[ReduceAutoSelector] aicpu levelNum > 1 not support INT64, UINT64, FP64."),
=======
            HCCL_WARNING("[Algo][ReduceAutoSelector] ReduceOp[%d] is not supported yet for aicpu levelNum > 1.",
                opParam.reduceType), SelectorStatus::NOT_MATCH);

        CHK_PRT_RET(isInt64Type(opParam.DataDes.dataType),
            HCCL_WARNING("[Algo][ReduceAutoSelector] aicpu levelNum > 1 not support INT64, UINT64, FP64."),
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
            SelectorStatus::NOT_MATCH);
        if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) > 1 && topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "ReduceParallelMesh1DNHR";
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) == 1 || topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "ReduceNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }
<<<<<<< HEAD
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
=======
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectMeshAlgoAicpu(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_DEBUG("SelectMeshAlgoAicpu %u", topoInfo->level0Topo);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            selectAlgName = "ReduceMesh1D"; // twoshot预留
        } else {
            selectAlgName = "ReduceMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
<<<<<<< HEAD
            if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
=======
            if (isInt64Type(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
                selectAlgName = "ReduceMesh1D";
            } else {
                selectAlgName = "ReduceParallelMesh1DNHR";
            }
        } else {
<<<<<<< HEAD
            if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
=======
            if (isInt64Type(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
                selectAlgName = "ReduceMesh1D";
            } else {
                selectAlgName = "ReduceNHR";
            }
        }
    } else {
<<<<<<< HEAD
        HCCL_WARNING("[ReduceAutoSelector] level0Shape[%d] is not supported yet.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
=======
        HCCL_WARNING("[Algo][ReduceAutoSelector] level0Shape[%d] is not supported yet.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
>>>>>>> 789a2ad... Hybrid comm and selector conflict  fix
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    // aiv 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[ReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.", opParam.reduceType),
        SelectorStatus::NOT_MATCH);

<<<<<<< HEAD
    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceAutoSelector] aiv mode not support INT64, UINT64, FP64.");
=======
    if (isInt64Type(opParam.DataDes.dataType)) {
        HCCL_WARNING("[Algo][ReduceAutoSelector] aiv mode not support INT64, UINT64, FP64.");
>>>>>>> a62c957... reduce/reducescatter/reducescatterv selector fix 2
        return SelectorStatus::NOT_MATCH;
    }

    selectAlgName = "AivReduceMesh1D";
<<<<<<< HEAD
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
=======
    HCCL_INFO("[Algo][ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
>>>>>>> a62c957... reduce/reducescatter/reducescatterv selector fix 2
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE, 18, ReduceAutoSelector);
}  // namespace ops_hccl
