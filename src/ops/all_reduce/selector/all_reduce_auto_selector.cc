/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr u64 RS_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_ONESHOT_1D_MAX_DATA_SIZE = 16 * 1024;
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 32 * 1024 * 1024;

SelectorStatus AllReduceAutoSelector::SelectCcuMsAlgo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[Algo][AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.DataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
    }
    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectMeshAlgo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                    std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (IsInputOutputOverlap(opParam) == true) {
            // 不支持 inplace 场景
            return SelectorStatus::NOT_MATCH;
        }
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else if (dataSize / topoInfo->userRankSize > AR_ONESHOT_1D_MAX_DATA_SIZE) {
            selectAlgName = "CcuAllReduceMesh1D";
        } else {
            selectAlgName = "CcuAllReduceMesh1DOneShot";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            if (IsInputOutputOverlap(opParam) == true) {
                // 不支持 inplace 场景
                return SelectorStatus::NOT_MATCH;
            }
            if (dataSize / topoInfo->userRankSize > AR_ONESHOT_1D_MAX_DATA_SIZE) {
                selectAlgName = "CcuAllReduceMesh1D";
            } else {
                selectAlgName = "CcuAllReduceMesh1DOneShot";
            }
        } else { // MS 不支持
            HCCL_WARNING("[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                        topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                        topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                        topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleAlgo(TopoInfoWithNetLayerDetails* topoInfo,
                                                            OpParam &opParam,
                                                            const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                            std::string &selectAlgName) const
{   
    (void)configAlgMap;
    HCCL_DEBUG("[Algo][AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType), SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "CcuAllReduceNHR1D";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[Algo][AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                 // 性能优化改用MS做reduce后不支持int8
                CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                    HCCL_WARNING("[Algo][AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode with ms "
                        "reduce. levelNum[%u]", opParam.DataDes.dataType, topoInfo->topoLevelNums), SelectorStatus::NOT_MATCH);
                selectAlgName = "CcuAllReduceParallelMesh1DNHR";
            } 
        } else {
            HCCL_WARNING("[Algo][AllReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectCcuScheduleLevel0Algo(topoInfo, opParam, selectAlgName);
    }
    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus SelectCcuScheduleLevel0Algo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                 std::string &selectAlgName) const
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        // 性能优化改用MS做reduce后不支持int8
        CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
            HCCL_WARNING("[Algo][AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode "
                            "with ms reduce.", opParam.DataDes.dataType), SelectorStatus::NOT_MATCH);
        double ratio;
        if (topoInfo->userRankSize == 0) {
            HCCL_WARNING("[AllReduceAutoSelector] the selector userRankSize not set");
            ratio = 1;
        } else {
            ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
        }
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else if (dataSize * ratio > AR_M2M_1D_MAX_DATA_SIZE) {
            return SelectorStatus::NOT_MATCH;
        }
        selectAlgName = "CcuAllReduceMesh1DMem2Mem";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                HCCL_WARNING("[Algo][AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode "
                            "with ms reduce.", opParam.DataDes.dataType), SelectorStatus::NOT_MATCH);
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            double ratio;
            if (topoInfo->userRankSize == 0) {
                HCCL_WARNING("[AllReduceAutoSelector] the selector userRankSize not set");
                ratio = 1;
            } else {
                ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
            }
            if (dataSize * ratio > AR_M2M_1D_MAX_DATA_SIZE) {
                return SelectorStatus::NOT_MATCH;
            }
                selectAlgName = "CcuAllReduceMesh1DMem2Mem";
        } else {
            selectAlgName = "CcuAllReduceParallelMesh1DNHR";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                        topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                        topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAicpuAlgo(TopoInfoWithNetLayerDetails* topoInfo,
                                                      OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[Algo][AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp [PROD]] is not supported yet for aicpu mode."),
        SelectorStatus::NOT_MATCH);

    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "InsAllReduceNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllReduceParallelMesh1DNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
        return SelectorStatus::NOT_MATCH;
    }

    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectMeshAlgoAicpu(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                          std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (Is64BitDataType(opParam.DataDes.dataType)) {
            HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
            return SelectorStatus::NOT_MATCH;
    }
    double ratio;
    if (topoInfo->userRankSize == 0) {
        HCCL_WARNING("[AllReduceAutoSelector] the selector userRankSize not set");
        ratio = 1;
    } else {
        ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
            selectAlgName = "InsAllReduceMesh1DOneShot";
        } else if (dataSize * ratio > AR_AICPU_1D_MAX_DATA_SIZE) { 
            selectAlgName = "InsAllReduceMesh1DTwoShotMeshChunk";
        } else {
            selectAlgName = "InsAllReduceMesh1DTwoShot";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
                selectAlgName = "InsAllReduceMesh1DOneShot";
            } else if (dataSize * ratio > AR_AICPU_1D_MAX_DATA_SIZE) { 
                selectAlgName = "InsAllReduceMesh1DTwoShotMeshChunk";
            } else {
                selectAlgName = "InsAllReduceMesh1DTwoShot";
            }
        } else {
            selectAlgName = "InsAllReduceParallelMesh1DNHR";
        }
    } 
    else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (Is64BitDataType(opParam.DataDes.dataType)) {
            HCCL_WARNING("[AllReduceAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "InsAllReduceNHR";
        }
    } else {
        HCCL_WARNING("[AllReduceAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }

    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAivAlgo(TopoInfoWithNetLayerDetails* topoInfo, OpParam &opParam,
                                                       const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                       std::string &selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[Algo][AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    
    //aiv 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] aiv mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

 
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (IsSmallData(dataSize)) {
        selectAlgName = "AivAllReduceMesh1DOneShot";
    } else {
        selectAlgName = "AivAllReduceMesh1DTwoShot";
    }
    HCCL_INFO("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
} // namespace ops_hccl
