/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALGO_SELECTION_TABLE_H
#define ALGO_SELECTION_TABLE_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <optional>
#include "hccl.h"
#include "op_common.h"

namespace ops_hccl {

// ============================================================================
// 运行时上下文：包含所有算法选择所需的运行时信息
// ============================================================================
struct AlgoSelectContext {
    // 操作信息
    HcclCMDType opType = HCCL_CMD_ALLREDUCE;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM;
    u64 dataSize = 0;
    
    // 拓扑信息
    OpExecuteConfig execConfig = OpExecuteConfig::AICPU_TS;
    u32 topoLevelNums = 1;              // 1: 单层级, >1: 多层级
    Level0Shape level0Topo = Level0Shape::MESH_1D;
    Level0MeshType level0MeshType = Level0MeshType::SINGLE_DIE;
    
    // 拓扑特征标志
    bool level0PcieMix = false;         // PCIE 混合拓扑
    bool is2DieFullMesh = false;        // 双 Die 全连接 Mesh
    bool level1Nhr = false;             // Level1 NHR 拓扑
    bool level0Nhr = false;             // Level0 NHR 拓扑
    
    // Mesh 与 Clos 数量关系
    bool meshNumEqualToClosNum = false;
    bool closNumMultipleOfMeshNum = false;
    
    // 网络信息
    u32 localNetInsSizeOfLayer0 = 0;    // 第一层本地网络实例数
    u32 userRankSize = 0;               // 用户 Rank 数量
    
    // 内存信息
    bool isInputOutputOverlap = false;  // 输入输出内存是否重叠
    
    // 链路信息（Send/Recv 专用）
    bool isHostToDevice = false;        // Host -> Device
    bool isDeviceToHost = false;        // Device -> Host
};

// ============================================================================
// 条件检查函数类型：返回 true 表示条件满足
// ============================================================================
using ConditionChecker = std::function<bool(const AlgoSelectContext&)>;

// ============================================================================
// 算法规则：条件检查函数 + 算法名称
// ============================================================================
struct AlgoRule {
    std::string name;                   // 规则名称（用于调试）
    ConditionChecker checker;           // 条件检查函数
    std::string algoName;               // 匹配时选择的算法名
    
    AlgoRule() = default;
    AlgoRule(const std::string& ruleName, ConditionChecker cond, const std::string& algo)
        : name(ruleName), checker(cond), algoName(algo) {}
};

// ============================================================================
// 多级查找表结构
// 
// 层级结构：
//   Level 1: OpExecuteConfig (执行配置)
//   Level 2: HcclCMDType (操作类型)  
//   Level 3: TopoLevelCategory (拓扑层级: 单层/多层)
//   Level 4: Level0Shape (Level0 拓扑类型)
//   Level 5: 规则列表（按优先级排序，依次匹配）
// ============================================================================

// 拓扑层级分类
enum class TopoLevelCategory {
    SINGLE_LAYER,   // 单层级 (topoLevelNums == 1)
    MULTI_LAYER,    // 多层级 (topoLevelNums > 1)
    ANY             // 任意层级（用于默认规则）
};

// 第五层：规则列表（按优先级排序，第一个匹配的规则生效）
using RuleList = std::vector<AlgoRule>;

// 第四层：Level0 拓扑类型 -> 规则列表
using Level0TopoRules = std::map<Level0Shape, RuleList>;

// 第三层：拓扑层级 -> Level0 拓扑规则
using TopoLevelRules = std::map<TopoLevelCategory, Level0TopoRules>;

// 第二层：操作类型 -> 拓扑层级规则
using OpTypeRules = std::map<HcclCMDType, TopoLevelRules>;

// 第一层（顶级）：执行配置 -> 操作类型规则
using AlgoSelectionTable = std::map<OpExecuteConfig, OpTypeRules>;

// ============================================================================
// 算法选择器类
// ============================================================================
class TableBasedAlgoSelector {
public:
    TableBasedAlgoSelector();
    ~TableBasedAlgoSelector() = default;
    
    /**
     * @brief 初始化查找表，填充所有算法规则
     */
    void Initialize();
    
    /**
     * @brief 根据上下文选择算法
     * @param ctx 运行时上下文
     * @return 匹配到的算法名称，未匹配返回 std::nullopt
     */
    std::optional<std::string> SelectAlgo(const AlgoSelectContext& ctx) const;
    
    /**
     * @brief 添加算法规则
     */
    void AddRule(OpExecuteConfig execConfig,
                 HcclCMDType opType,
                 TopoLevelCategory topoLevel,
                 Level0Shape level0Topo,
                 const AlgoRule& rule);
    
    /**
     * @brief 添加通配规则（匹配任意 Level0 拓扑）
     */
    void AddRuleForAllTopo(OpExecuteConfig execConfig,
                           HcclCMDType opType,
                           TopoLevelCategory topoLevel,
                           const AlgoRule& rule);
    
    /**
     * @brief 打印查找表内容（用于调试）
     */
    void DumpTable() const;

private:
    AlgoSelectionTable table_;
    
    TopoLevelCategory CategorizeTopoLevel(u32 levelNums) const {
        return (levelNums == 1) ? TopoLevelCategory::SINGLE_LAYER : TopoLevelCategory::MULTI_LAYER;
    }
};

// ============================================================================
// 常用条件检查函数
// ============================================================================
namespace Conditions {

// ---- 数据类型相关 ----
inline bool Is64BitDataType(const AlgoSelectContext& ctx) {
    return ctx.dataType == HCCL_DATA_TYPE_INT64 ||
           ctx.dataType == HCCL_DATA_TYPE_UINT64 ||
           ctx.dataType == HCCL_DATA_TYPE_FP64;
}

inline bool IsInt8(const AlgoSelectContext& ctx) {
    return ctx.dataType == HCCL_DATA_TYPE_INT8;
}

// ---- 归约操作相关 ----
inline bool IsReduceProd(const AlgoSelectContext& ctx) {
    return ctx.reduceType == HCCL_REDUCE_PROD;
}

// ---- 数据大小区间判断 ----
constexpr u64 KB = 1024ULL;
constexpr u64 MB = 1024 * KB;

// 数据大小在区间 [minSize, maxSize)
inline ConditionChecker DataSizeInRange(u64 minSize, u64 maxSize) {
    return [minSize, maxSize](const AlgoSelectContext& ctx) {
        return ctx.dataSize >= minSize && ctx.dataSize < maxSize;
    };
}

// 数据大小 < threshold
inline ConditionChecker DataSizeBelow(u64 threshold) {
    return [threshold](const AlgoSelectContext& ctx) {
        return ctx.dataSize < threshold;
    };
}

// 数据大小 >= threshold
inline ConditionChecker DataSizeAbove(u64 threshold) {
    return [threshold](const AlgoSelectContext& ctx) {
        return ctx.dataSize >= threshold;
    };
}

// ---- Rank 数量区间判断 ----

// Rank 数量在区间 [minRank, maxRank]
inline ConditionChecker RankSizeInRange(u32 minRank, u32 maxRank) {
    return [minRank, maxRank](const AlgoSelectContext& ctx) {
        return ctx.userRankSize >= minRank && ctx.userRankSize <= maxRank;
    };
}

// Rank 数量 < threshold
inline ConditionChecker RankSizeBelow(u32 threshold) {
    return [threshold](const AlgoSelectContext& ctx) {
        return ctx.userRankSize < threshold;
    };
}

// Rank 数量 > threshold
inline ConditionChecker RankSizeAbove(u32 threshold) {
    return [threshold](const AlgoSelectContext& ctx) {
        return ctx.userRankSize > threshold;
    };
}

// Rank 数量 >= threshold
inline ConditionChecker RankSizeAtLeast(u32 threshold) {
    return [threshold](const AlgoSelectContext& ctx) {
        return ctx.userRankSize >= threshold;
    };
}

// ---- 拓扑特征相关 ----
inline bool IsPcieMix(const AlgoSelectContext& ctx) {
    return ctx.level0PcieMix;
}

inline bool IsNotPcieMix(const AlgoSelectContext& ctx) {
    return !ctx.level0PcieMix;
}

inline bool Is2DieFullMesh(const AlgoSelectContext& ctx) {
    return ctx.is2DieFullMesh;
}

inline bool IsNot2DieFullMesh(const AlgoSelectContext& ctx) {
    return !ctx.is2DieFullMesh;
}

inline bool IsLevel1Nhr(const AlgoSelectContext& ctx) {
    return ctx.level1Nhr;
}

inline bool IsMeshNumEqualToClosNum(const AlgoSelectContext& ctx) {
    return ctx.meshNumEqualToClosNum;
}

inline bool IsClosNumMultipleOfMeshNum(const AlgoSelectContext& ctx) {
    return ctx.closNumMultipleOfMeshNum;
}

// ---- 网络相关 ----
inline bool LocalNetInsSizeIs1(const AlgoSelectContext& ctx) {
    return ctx.localNetInsSizeOfLayer0 == 1;
}

inline bool LocalNetInsSizeNot1(const AlgoSelectContext& ctx) {
    return ctx.localNetInsSizeOfLayer0 != 1;
}

// ---- 内存相关 ----
inline bool IsInputOutputOverlap(const AlgoSelectContext& ctx) {
    return ctx.isInputOutputOverlap;
}

inline bool IsNotInputOutputOverlap(const AlgoSelectContext& ctx) {
    return !ctx.isInputOutputOverlap;
}

// ---- 链路相关（Send/Recv）----
inline bool IsHostToDevice(const AlgoSelectContext& ctx) {
    return ctx.isHostToDevice;
}

inline bool IsDeviceToHost(const AlgoSelectContext& ctx) {
    return ctx.isDeviceToHost;
}

// ---- Mesh 类型相关 ----
inline bool IsTwoDieRegular(const AlgoSelectContext& ctx) {
    return ctx.level0MeshType == Level0MeshType::TWO_DIE_REGULAR;
}

inline bool IsTwoDieNotRegular(const AlgoSelectContext& ctx) {
    return ctx.level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR;
}

inline bool IsSingleDie(const AlgoSelectContext& ctx) {
    return ctx.level0MeshType == Level0MeshType::SINGLE_DIE;
}

// ---- 组合条件 ----
inline ConditionChecker AllOf(std::vector<ConditionChecker> conditions) {
    return [conditions = std::move(conditions)](const AlgoSelectContext& ctx) {
        for (const auto& cond : conditions) {
            if (!cond(ctx)) return false;
        }
        return true;
    };
}

inline ConditionChecker AnyOf(std::vector<ConditionChecker> conditions) {
    return [conditions = std::move(conditions)](const AlgoSelectContext& ctx) {
        for (const auto& cond : conditions) {
            if (cond(ctx)) return true;
        }
        return false;
    };
}

inline ConditionChecker Not(ConditionChecker condition) {
    return [condition = std::move(condition)](const AlgoSelectContext& ctx) {
        return !condition(ctx);
    };
}

// 无条件匹配（始终为 true）
inline bool Always(const AlgoSelectContext& ctx) {
    (void)ctx;
    return true;
}

} // namespace Conditions

} // namespace ops_hccl

#endif // ALGO_SELECTION_TABLE_H