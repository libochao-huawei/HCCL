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
#include <optional>
#include <cstdint>

typedef uint64_t u64;
typedef uint32_t u32;

namespace ops_hccl {

// ============================================================================
// 枚举定义
// ============================================================================

enum class OpExecuteConfig {
    CCU_MS,
    CCU_SCHED,
    CCU_FAIL,
    AICPU_TS,
    AIV,
    AIV_ONLY,
    HOSTCPU,
    HOSTCPU_TS,
    UNKNOWN
};

enum class HcclCMDType {
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BROADCAST,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SCATTER,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_SEND,
    HCCL_CMD_RECV,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_UNKNOWN
};

enum class HcclDataType {
    HCCL_DATA_TYPE_INT8,
    HCCL_DATA_TYPE_INT32,
    HCCL_DATA_TYPE_FP16,
    HCCL_DATA_TYPE_BF16,
    HCCL_DATA_TYPE_FP32,
    HCCL_DATA_TYPE_INT64,
    HCCL_DATA_TYPE_UINT64,
    HCCL_DATA_TYPE_FP64,
    HCCL_DATA_TYPE_UNKNOWN
};

enum class HcclReduceOp {
    HCCL_REDUCE_SUM,
    HCCL_REDUCE_PROD,
    HCCL_REDUCE_MAX,
    HCCL_REDUCE_MIN,
    HCCL_REDUCE_UNKNOWN
};

enum class Level0Shape {
    MESH_1D,
    MESH_1D_CLOS,
    CLOS,
    UNKNOWN
};

enum class Level0MeshType {
    SINGLE_DIE,
    TWO_DIE_REGULAR,
    TWO_DIE_NOT_REGULAR,
    UNKNOWN
};

// ============================================================================
// 规则条目：所有条件平铺在一个 Map 中
//
// key = 条件名, value = 条件值（字符串形式）
// 特殊 key:
//   "_algo"        -> 算法名（结果）
//   "_execConfig"  -> 执行配置（必填）
//   "_opType"      -> 操作类型（必填）
//
// 支持的 key 列表：
//   _execConfig:     CCU_MS / CCU_SCHED / AICPU_TS / HOSTCPU / ...
//   _opType:         AllReduce / Broadcast / AllGather / ...
//   _topoLevel:      1 / 2 / multi
//   _level0Topo:     MESH_1D / CLOS / MESH_1D_CLOS
//   _level0MeshType: SINGLE_DIE / TWO_DIE_REGULAR / TWO_DIE_NOT_REGULAR
//   _level0PcieMix:  true / false
//   _is2DieFullMesh: true / false
//   _level1Nhr:      true / false
//   _level0Nhr:      true / false
//   _meshEqClos:     true / false
//   _closMulMesh:    true / false
//   _localNetIns:    <数字>
//   _rankSizeMin:    <数字>
//   _rankSizeMax:    <数字>
//   _dataSizeMin:    <数字>
//   _dataSizeMax:    <数字>
//   _dataTypes:      INT64,UINT64,FP64 (逗号分隔，匹配任一即可)
//   _reduceOp:       SUM / PROD / MAX / MIN
//   _overlap:        true / false
//   _hostToDevice:   true / false
//   _deviceToHost:   true / false
// ============================================================================
using RuleMap = std::map<std::string, std::string>;

// ============================================================================
// 运行时上下文
// ============================================================================
struct AlgoSelectContext {
    OpExecuteConfig execConfig = OpExecuteConfig::AICPU_TS;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    u64 dataSize = 0;
    u32 topoLevelNums = 1;
    Level0Shape level0Topo = Level0Shape::MESH_1D;
    Level0MeshType level0MeshType = Level0MeshType::SINGLE_DIE;
    bool level0PcieMix = false;
    bool is2DieFullMesh = false;
    bool level1Nhr = false;
    bool level0Nhr = false;
    bool meshNumEqualToClosNum = false;
    bool closNumMultipleOfMeshNum = false;
    u32 localNetInsSizeOfLayer0 = 0;
    u32 userRankSize = 0;
    bool isInputOutputOverlap = false;
    bool isHostToDevice = false;
    bool isDeviceToHost = false;
};

// ============================================================================
// 算法选择器
// ============================================================================
class TableBasedAlgoSelector {
public:
    TableBasedAlgoSelector() = default;

    // 初始化默认规则表
    void Initialize();

    // 初始化默认规则表 + 加载外部配置文件（外部规则优先）
    // configFilePath: 外部配置文件路径，为空字符串时仅加载默认规则
    void InitializeWithConfig(const std::string& configFilePath);

    // 从外部 txt 文件加载规则
    // 返回成功加载的规则数量，失败返回 -1
    int LoadRulesFromFile(const std::string& filePath);

    // 从字符串内容解析规则（用于测试或嵌入式场景）
    int LoadRulesFromString(const std::string& content);

    std::optional<std::string> SelectAlgo(const AlgoSelectContext& ctx) const;

    // 直接操作规则表
    std::vector<RuleMap>& GetRules() { return rules_; }
    const std::vector<RuleMap>& GetRules() const { return rules_; }
    void AddRule(const RuleMap& rule) { rules_.push_back(rule); }

    // 将外部规则插入到规则表头部（优先级高于默认规则）
    void PrependRule(const RuleMap& rule);

    void DumpTable() const;

private:
    std::vector<RuleMap> rules_;
    bool MatchRule(const RuleMap& rule, const AlgoSelectContext& ctx) const;

    // 解析单条规则的文本块为 RuleMap
    static RuleMap ParseRuleBlock(const std::vector<std::string>& lines);

    // 去除字符串首尾空白
    static std::string Trim(const std::string& s);
};

} // namespace ops_hccl

#endif // ALGO_SELECTION_TABLE_H