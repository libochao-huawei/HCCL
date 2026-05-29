/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "algo_selection_table.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ops_hccl {

// ============================================================================
// 辅助常量
// ============================================================================
static constexpr u64 KB = 1024ULL;
static constexpr u64 MB = 1024 * KB;

// ============================================================================
// 辅助函数：从 Map 获取值
// ============================================================================
static bool HasKey(const RuleMap& m, const std::string& key) {
    return m.find(key) != m.end();
}

static std::string GetVal(const RuleMap& m, const std::string& key) {
    auto it = m.find(key);
    return (it != m.end()) ? it->second : "";
}

static bool GetBool(const RuleMap& m, const std::string& key) {
    return GetVal(m, key) == "true";
}

static u32 GetU32(const RuleMap& m, const std::string& key, u32 def = 0) {
    if (!HasKey(m, key)) return def;
    return static_cast<u32>(std::stoul(GetVal(m, key)));
}

static u64 GetU64(const RuleMap& m, const std::string& key, u64 def = 0) {
    if (!HasKey(m, key)) return def;
    return static_cast<u64>(std::stoull(GetVal(m, key)));
}

// ============================================================================
// 匹配逻辑
// ============================================================================

// 枚举转字符串
static std::string ExecConfigToStr(OpExecuteConfig c) {
    switch (c) {
        case OpExecuteConfig::CCU_MS: return "CCU_MS";
        case OpExecuteConfig::CCU_SCHED: return "CCU_SCHED";
        case OpExecuteConfig::CCU_FAIL: return "CCU_FAIL";
        case OpExecuteConfig::AICPU_TS: return "AICPU_TS";
        case OpExecuteConfig::AIV: return "AIV";
        case OpExecuteConfig::AIV_ONLY: return "AIV_ONLY";
        case OpExecuteConfig::HOSTCPU: return "HOSTCPU";
        case OpExecuteConfig::HOSTCPU_TS: return "HOSTCPU_TS";
        default: return "UNKNOWN";
    }
}

static std::string CmdTypeToStr(HcclCMDType t) {
    switch (t) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE: return "AllReduce";
        case HcclCMDType::HCCL_CMD_ALLGATHER: return "AllGather";
        case HcclCMDType::HCCL_CMD_ALLGATHER_V: return "AllGatherV";
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER: return "ReduceScatter";
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V: return "ReduceScatterV";
        case HcclCMDType::HCCL_CMD_BROADCAST: return "Broadcast";
        case HcclCMDType::HCCL_CMD_REDUCE: return "Reduce";
        case HcclCMDType::HCCL_CMD_SCATTER: return "Scatter";
        case HcclCMDType::HCCL_CMD_ALLTOALL: return "AlltoAll";
        case HcclCMDType::HCCL_CMD_ALLTOALLV: return "AlltoAllV";
        case HcclCMDType::HCCL_CMD_ALLTOALLVC: return "AlltoAllVC";
        case HcclCMDType::HCCL_CMD_SEND: return "Send";
        case HcclCMDType::HCCL_CMD_RECV: return "Recv";
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV: return "BatchSendRecv";
        default: return "Unknown";
    }
}

static std::string DataTypeToStr(HcclDataType t) {
    switch (t) {
        case HcclDataType::HCCL_DATA_TYPE_INT8: return "INT8";
        case HcclDataType::HCCL_DATA_TYPE_INT32: return "INT32";
        case HcclDataType::HCCL_DATA_TYPE_FP16: return "FP16";
        case HcclDataType::HCCL_DATA_TYPE_BF16: return "BF16";
        case HcclDataType::HCCL_DATA_TYPE_FP32: return "FP32";
        case HcclDataType::HCCL_DATA_TYPE_INT64: return "INT64";
        case HcclDataType::HCCL_DATA_TYPE_UINT64: return "UINT64";
        case HcclDataType::HCCL_DATA_TYPE_FP64: return "FP64";
        default: return "Unknown";
    }
}

static std::string ReduceOpToStr(HcclReduceOp r) {
    switch (r) {
        case HcclReduceOp::HCCL_REDUCE_SUM: return "SUM";
        case HcclReduceOp::HCCL_REDUCE_PROD: return "PROD";
        case HcclReduceOp::HCCL_REDUCE_MAX: return "MAX";
        case HcclReduceOp::HCCL_REDUCE_MIN: return "MIN";
        default: return "Unknown";
    }
}

static std::string Level0ShapeToStr(Level0Shape s) {
    switch (s) {
        case Level0Shape::MESH_1D: return "MESH_1D";
        case Level0Shape::MESH_1D_CLOS: return "MESH_1D_CLOS";
        case Level0Shape::CLOS: return "CLOS";
        default: return "Unknown";
    }
}

static std::string Level0MeshTypeToStr(Level0MeshType t) {
    switch (t) {
        case Level0MeshType::SINGLE_DIE: return "SINGLE_DIE";
        case Level0MeshType::TWO_DIE_REGULAR: return "TWO_DIE_REGULAR";
        case Level0MeshType::TWO_DIE_NOT_REGULAR: return "TWO_DIE_NOT_REGULAR";
        default: return "Unknown";
    }
}

// 分割字符串
static std::vector<std::string> SplitStr(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool TableBasedAlgoSelector::MatchRule(const RuleMap& rule, const AlgoSelectContext& ctx) const {
    // 必填条件：_execConfig
    if (HasKey(rule, "_execConfig")) {
        if (GetVal(rule, "_execConfig") != ExecConfigToStr(ctx.execConfig)) {
            return false;
        }
    }
    
    // 必填条件：_opType
    if (HasKey(rule, "_opType")) {
        if (GetVal(rule, "_opType") != CmdTypeToStr(ctx.opType)) {
            return false;
        }
    }
    
    // 可选条件：_topoLevel
    if (HasKey(rule, "_topoLevel")) {
        std::string val = GetVal(rule, "_topoLevel");
        if (val == "multi") {
            if (ctx.topoLevelNums <= 1) return false;
        } else {
            if (ctx.topoLevelNums != GetU32(rule, "_topoLevel")) return false;
        }
    }
    
    // 可选条件：_level0Topo
    if (HasKey(rule, "_level0Topo")) {
        if (GetVal(rule, "_level0Topo") != Level0ShapeToStr(ctx.level0Topo)) {
            return false;
        }
    }
    
    // 可选条件：_level0MeshType
    if (HasKey(rule, "_level0MeshType")) {
        if (GetVal(rule, "_level0MeshType") != Level0MeshTypeToStr(ctx.level0MeshType)) {
            return false;
        }
    }
    
    // 可选条件：_level0PcieMix
    if (HasKey(rule, "_level0PcieMix")) {
        if (GetBool(rule, "_level0PcieMix") != ctx.level0PcieMix) return false;
    }
    
    // 可选条件：_is2DieFullMesh
    if (HasKey(rule, "_is2DieFullMesh")) {
        if (GetBool(rule, "_is2DieFullMesh") != ctx.is2DieFullMesh) return false;
    }
    
    // 可选条件：_level1Nhr
    if (HasKey(rule, "_level1Nhr")) {
        if (GetBool(rule, "_level1Nhr") != ctx.level1Nhr) return false;
    }
    
    // 可选条件：_level0Nhr
    if (HasKey(rule, "_level0Nhr")) {
        if (GetBool(rule, "_level0Nhr") != ctx.level0Nhr) return false;
    }
    
    // 可选条件：_meshEqClos
    if (HasKey(rule, "_meshEqClos")) {
        if (GetBool(rule, "_meshEqClos") != ctx.meshNumEqualToClosNum) return false;
    }
    
    // 可选条件：_closMulMesh
    if (HasKey(rule, "_closMulMesh")) {
        if (GetBool(rule, "_closMulMesh") != ctx.closNumMultipleOfMeshNum) return false;
    }
    
    // 可选条件：_localNetIns
    if (HasKey(rule, "_localNetIns")) {
        if (GetU32(rule, "_localNetIns") != ctx.localNetInsSizeOfLayer0) return false;
    }
    
    // 可选条件：_rankSizeMin
    if (HasKey(rule, "_rankSizeMin")) {
        if (ctx.userRankSize < GetU32(rule, "_rankSizeMin")) return false;
    }
    
    // 可选条件：_rankSizeMax
    if (HasKey(rule, "_rankSizeMax")) {
        if (ctx.userRankSize > GetU32(rule, "_rankSizeMax")) return false;
    }
    
    // 可选条件：_dataSizeMin
    if (HasKey(rule, "_dataSizeMin")) {
        if (ctx.dataSize < GetU64(rule, "_dataSizeMin")) return false;
    }
    
    // 可选条件：_dataSizeMax
    if (HasKey(rule, "_dataSizeMax")) {
        if (ctx.dataSize >= GetU64(rule, "_dataSizeMax")) return false;
    }
    
    // 可选条件：_dataTypes (逗号分隔，匹配任一即可)
    if (HasKey(rule, "_dataTypes")) {
        std::string types = GetVal(rule, "_dataTypes");
        std::vector<std::string> typeList = SplitStr(types, ',');
        std::string ctxType = DataTypeToStr(ctx.dataType);
        bool found = false;
        for (const auto& t : typeList) {
            if (t == ctxType) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    
    // 可选条件：_reduceOp
    if (HasKey(rule, "_reduceOp")) {
        if (GetVal(rule, "_reduceOp") != ReduceOpToStr(ctx.reduceOp)) return false;
    }
    
    // 可选条件：_overlap
    if (HasKey(rule, "_overlap")) {
        if (GetBool(rule, "_overlap") != ctx.isInputOutputOverlap) return false;
    }
    
    // 可选条件：_hostToDevice
    if (HasKey(rule, "_hostToDevice")) {
        if (GetBool(rule, "_hostToDevice") != ctx.isHostToDevice) return false;
    }
    
    // 可选条件：_deviceToHost
    if (HasKey(rule, "_deviceToHost")) {
        if (GetBool(rule, "_deviceToHost") != ctx.isDeviceToHost) return false;
    }
    
    return true;
}

std::optional<std::string> TableBasedAlgoSelector::SelectAlgo(const AlgoSelectContext& ctx) const {
    for (const auto& rule : rules_) {
        if (MatchRule(rule, ctx)) {
            if (HasKey(rule, "_algo")) {
                return GetVal(rule, "_algo");
            }
        }
    }
    return std::nullopt;
}

void TableBasedAlgoSelector::DumpTable() const {
    std::cout << "=== Algorithm Selection Table (" << rules_.size() << " rules) ===" << std::endl;
    for (size_t i = 0; i < rules_.size(); ++i) {
        const auto& rule = rules_[i];
        std::cout << "[" << i << "] ";
        if (HasKey(rule, "_execConfig")) std::cout << GetVal(rule, "_execConfig") << " ";
        if (HasKey(rule, "_opType")) std::cout << GetVal(rule, "_opType") << " ";
        std::cout << "-> " << GetVal(rule, "_algo") << std::endl;
    }
}

// ============================================================================
// 初始化规则表
//
// 每条规则是一个 Map，所有条件平铺：
//   - 必填：_execConfig, _opType
//   - 可选：其余条件
//   - 结果：_algo
//
// 规则按顺序匹配，先匹配的先返回
// ============================================================================

void TableBasedAlgoSelector::Initialize() {
    rules_.clear();
    
    // ==========================================================================
    // CCU_MS 模式 - AllReduce
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "CCU_MS"}, {"_opType", "AllReduce"}, {"_topoLevel", "1"}, {"_level0Topo", "MESH_1D"},
        {"_rankSizeMin", "1"}, {"_rankSizeMax", "7"}, {"_dataSizeMax", std::to_string(512 * KB)},
        {"_algo", "ccu_ms_ar_small_rank"}
    });
    
    rules_.push_back({
        {"_execConfig", "CCU_MS"}, {"_opType", "AllReduce"}, {"_topoLevel", "1"}, {"_level0Topo", "MESH_1D"},
        {"_rankSizeMin", "8"}, {"_dataSizeMax", std::to_string(512 * KB)},
        {"_algo", "ccu_ms_ar_large_rank"}
    });
    
    rules_.push_back({
        {"_execConfig", "CCU_MS"}, {"_opType", "AllReduce"}, {"_topoLevel", "1"}, {"_level0Topo", "MESH_1D"},
        {"_level0PcieMix", "false"},
        {"_dataSizeMin", std::to_string(512 * KB)}, {"_dataSizeMax", std::to_string(8 * MB)},
        {"_algo", "ccu_ms_ar_medium"}
    });
    
    rules_.push_back({
        {"_execConfig", "CCU_MS"}, {"_opType", "AllReduce"}, {"_topoLevel", "1"}, {"_level0Topo", "MESH_1D"},
        {"_dataSizeMin", std::to_string(8 * MB)},
        {"_algo", "ccu_ms_ar_large"}
    });
    
    rules_.push_back({
        {"_execConfig", "CCU_MS"}, {"_opType", "AllReduce"}, {"_level0Topo", "CLOS"},
        {"_algo", "ccu_ms_ar_clos"}
    });
    
    // ==========================================================================
    // CCU_SCHED 模式 - AllReduce
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "CCU_SCHED"}, {"_opType", "AllReduce"}, {"_level0Topo", "MESH_1D"},
        {"_is2DieFullMesh", "true"},
        {"_algo", "ccu_sched_ar_2die"}
    });
    
    rules_.push_back({
        {"_execConfig", "CCU_SCHED"}, {"_opType", "AllReduce"}, {"_level0Topo", "MESH_1D"},
        {"_algo", "ccu_sched_ar_default"}
    });
    
    // ==========================================================================
    // AICPU_TS 模式 - AllReduce
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllReduce"},
        {"_dataTypes", "INT64,UINT64,FP64"},
        {"_algo", "aicpu_ar_64bit"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllReduce"},
        {"_reduceOp", "PROD"},
        {"_algo", "aicpu_ar_prod"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllReduce"},
        {"_dataSizeMax", std::to_string(512 * KB)},
        {"_algo", "aicpu_ar_small"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllReduce"},
        {"_algo", "aicpu_ar_default"}
    });
    
    // 多层级 AllReduce
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllReduce"},
        {"_topoLevel", "multi"}, {"_level1Nhr", "true"},
        {"_algo", "aicpu_ar_l1nhr"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllReduce"},
        {"_topoLevel", "multi"},
        {"_algo", "aicpu_ar_multilvl"}
    });
    
    // ==========================================================================
    // AICPU_TS 模式 - Broadcast
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "Broadcast"},
        {"_dataSizeMax", std::to_string(1 * MB)},
        {"_algo", "broadcast_oneshot"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "Broadcast"},
        {"_algo", "broadcast_twoshot"}
    });
    
    // ==========================================================================
    // AICPU_TS 模式 - AllGather
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllGather"},
        {"_dataSizeMax", std::to_string(256 * KB)},
        {"_algo", "allgather_small"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AllGather"},
        {"_algo", "allgather_default"}
    });
    
    // ==========================================================================
    // AICPU_TS 模式 - ReduceScatter
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "ReduceScatter"},
        {"_dataSizeMax", std::to_string(256 * KB)},
        {"_algo", "reducescatter_small"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "ReduceScatter"},
        {"_algo", "reducescatter_default"}
    });
    
    // ==========================================================================
    // AICPU_TS 模式 - AlltoAll
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AlltoAll"},
        {"_level0PcieMix", "true"},
        {"_algo", "alltoall_pcie"}
    });
    
    rules_.push_back({
        {"_execConfig", "AICPU_TS"}, {"_opType", "AlltoAll"},
        {"_algo", "alltoall_default"}
    });
    
    // ==========================================================================
    // HOSTCPU 模式 - Send
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "HOSTCPU"}, {"_opType", "Send"},
        {"_hostToDevice", "true"},
        {"_algo", "send_host_dpu"}
    });
    
    rules_.push_back({
        {"_execConfig", "HOSTCPU"}, {"_opType", "Send"},
        {"_algo", "send_device_dpu"}
    });
    
    // ==========================================================================
    // HOSTCPU 模式 - Recv
    // ==========================================================================
    rules_.push_back({
        {"_execConfig", "HOSTCPU"}, {"_opType", "Recv"},
        {"_deviceToHost", "true"},
        {"_algo", "recv_host_dpu"}
    });
    
    rules_.push_back({
        {"_execConfig", "HOSTCPU"}, {"_opType", "Recv"},
        {"_algo", "recv_device_dpu"}
    });
}

// ============================================================================
// 字符串工具函数
// ============================================================================

std::string TableBasedAlgoSelector::Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ============================================================================
// 规则块解析
// ============================================================================

RuleMap TableBasedAlgoSelector::ParseRuleBlock(const std::vector<std::string>& lines) {
    RuleMap rule;
    for (const auto& rawLine : lines) {
        std::string line = Trim(rawLine);
        // 跳过空行和注释行
        if (line.empty() || line[0] == '#') continue;

        // 解析 key = value 格式
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;

        std::string key = Trim(line.substr(0, eqPos));
        std::string value = Trim(line.substr(eqPos + 1));

        if (!key.empty() && !value.empty()) {
            rule[key] = value;
        }
    }
    return rule;
}

// ============================================================================
// 从字符串内容加载规则
// ============================================================================

int TableBasedAlgoSelector::LoadRulesFromString(const std::string& content) {
    std::istringstream stream(content);
    std::string line;
    std::vector<std::string> block;
    int count = 0;

    while (std::getline(stream, line)) {
        std::string trimmed = Trim(line);

        // 遇到 [rule] 标记，开始新的规则块
        if (trimmed == "[rule]") {
            // 如果之前有未处理的块，先解析
            if (!block.empty()) {
                RuleMap rule = ParseRuleBlock(block);
                if (!rule.empty() && rule.count("_algo") > 0) {
                    PrependRule(rule);
                    count++;
                }
                block.clear();
            }
            continue;
        }

        // 累积当前块的行
        if (!trimmed.empty() || !block.empty()) {
            block.push_back(line);
        }
    }

    // 处理最后一个块
    if (!block.empty()) {
        RuleMap rule = ParseRuleBlock(block);
        if (!rule.empty() && rule.count("_algo") > 0) {
            PrependRule(rule);
            count++;
        }
    }

    return count;
}

// ============================================================================
// 从外部文件加载规则
// ============================================================================

int TableBasedAlgoSelector::LoadRulesFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[AlgoSelector] Failed to open config file: " << filePath << std::endl;
        return -1;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    int count = LoadRulesFromString(content);
    if (count > 0) {
        std::cout << "[AlgoSelector] Loaded " << count
                  << " external rules from: " << filePath << std::endl;
    }
    return count;
}

// ============================================================================
// 初始化 + 加载外部配置
// ============================================================================

void TableBasedAlgoSelector::InitializeWithConfig(const std::string& configFilePath) {
    // 1. 先加载默认规则
    Initialize();

    // 2. 如果提供了配置文件路径，加载外部规则（插入头部，优先匹配）
    if (!configFilePath.empty()) {
        LoadRulesFromFile(configFilePath);
    }
}

// ============================================================================
// 将规则插入到规则表头部
// ============================================================================

void TableBasedAlgoSelector::PrependRule(const RuleMap& rule) {
    rules_.insert(rules_.begin(), rule);
}

// ============================================================================
// ============================================================================
//
//  FastAlgoSelector 实现
//  高速算法选择器：分桶 + 预解析 + 哈希查表
//
// ============================================================================
// ============================================================================

// ============================================================================
// FastRule 匹配方法
// ============================================================================

bool FastRule::MatchDiscrete(const AlgoSelectContext& ctx) const {
    // 拓扑层级
    if (topoLevel.has_value()) {
        if (topoLevel.value() == -1) {
            if (ctx.topoLevelNums <= 1) return false;
        } else {
            if (ctx.topoLevelNums != static_cast<u32>(topoLevel.value())) return false;
        }
    }
    // Level0 拓扑形状
    if (level0Topo.has_value() && ctx.level0Topo != level0Topo.value()) return false;
    // Level0 Mesh 类型
    if (level0MeshType.has_value() && ctx.level0MeshType != level0MeshType.value()) return false;
    // 布尔条件
    if (level0PcieMix.has_value() && ctx.level0PcieMix != level0PcieMix.value()) return false;
    if (is2DieFullMesh.has_value() && ctx.is2DieFullMesh != is2DieFullMesh.value()) return false;
    if (level1Nhr.has_value() && ctx.level1Nhr != level1Nhr.value()) return false;
    if (level0Nhr.has_value() && ctx.level0Nhr != level0Nhr.value()) return false;
    if (meshEqClos.has_value() && ctx.meshNumEqualToClosNum != meshEqClos.value()) return false;
    if (closMulMesh.has_value() && ctx.closNumMultipleOfMeshNum != closMulMesh.value()) return false;
    // 本地网络实例数
    if (localNetIns.has_value() && ctx.localNetInsSizeOfLayer0 != localNetIns.value()) return false;
    // 归约操作
    if (reduceOp.has_value() && ctx.reduceOp != reduceOp.value()) return false;
    // 重叠
    if (overlap.has_value() && ctx.isInputOutputOverlap != overlap.value()) return false;
    // Host/Device 传输
    if (hostToDevice.has_value() && ctx.isHostToDevice != hostToDevice.value()) return false;
    if (deviceToHost.has_value() && ctx.isDeviceToHost != deviceToHost.value()) return false;
    // 数据类型白名单
    if (!dataTypes.empty()) {
        bool found = false;
        for (auto dt : dataTypes) {
            if (ctx.dataType == dt) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

bool FastRule::MatchFull(const AlgoSelectContext& ctx) const {
    // 先做快速离散匹配
    if (!MatchDiscrete(ctx)) return false;
    // 再做范围匹配
    if (hasRange) {
        if (ctx.userRankSize < rankSizeMin || ctx.userRankSize > rankSizeMax) return false;
        if (ctx.dataSize < dataSizeMin || ctx.dataSize >= dataSizeMax) return false;
    }
    return true;
}

// ============================================================================
// 字符串解析辅助
// ============================================================================

static OpExecuteConfig StrToExecConfig(const std::string& s) {
    if (s == "CCU_MS") return OpExecuteConfig::CCU_MS;
    if (s == "CCU_SCHED") return OpExecuteConfig::CCU_SCHED;
    if (s == "CCU_FAIL") return OpExecuteConfig::CCU_FAIL;
    if (s == "AICPU_TS") return OpExecuteConfig::AICPU_TS;
    if (s == "AIV") return OpExecuteConfig::AIV;
    if (s == "AIV_ONLY") return OpExecuteConfig::AIV_ONLY;
    if (s == "HOSTCPU") return OpExecuteConfig::HOSTCPU;
    if (s == "HOSTCPU_TS") return OpExecuteConfig::HOSTCPU_TS;
    return OpExecuteConfig::UNKNOWN;
}

static HcclCMDType StrToCmdType(const std::string& s) {
    if (s == "AllReduce") return HcclCMDType::HCCL_CMD_ALLREDUCE;
    if (s == "AllGather") return HcclCMDType::HCCL_CMD_ALLGATHER;
    if (s == "AllGatherV") return HcclCMDType::HCCL_CMD_ALLGATHER_V;
    if (s == "ReduceScatter") return HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    if (s == "ReduceScatterV") return HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    if (s == "Broadcast") return HcclCMDType::HCCL_CMD_BROADCAST;
    if (s == "Reduce") return HcclCMDType::HCCL_CMD_REDUCE;
    if (s == "Scatter") return HcclCMDType::HCCL_CMD_SCATTER;
    if (s == "AlltoAll") return HcclCMDType::HCCL_CMD_ALLTOALL;
    if (s == "AlltoAllV") return HcclCMDType::HCCL_CMD_ALLTOALLV;
    if (s == "AlltoAllVC") return HcclCMDType::HCCL_CMD_ALLTOALLVC;
    if (s == "Send") return HcclCMDType::HCCL_CMD_SEND;
    if (s == "Recv") return HcclCMDType::HCCL_CMD_RECV;
    if (s == "BatchSendRecv") return HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    return HcclCMDType::HCCL_CMD_UNKNOWN;
}

static Level0Shape StrToLevel0Shape(const std::string& s) {
    if (s == "MESH_1D") return Level0Shape::MESH_1D;
    if (s == "MESH_1D_CLOS") return Level0Shape::MESH_1D_CLOS;
    if (s == "CLOS") return Level0Shape::CLOS;
    return Level0Shape::UNKNOWN;
}

static Level0MeshType StrToLevel0MeshType(const std::string& s) {
    if (s == "SINGLE_DIE") return Level0MeshType::SINGLE_DIE;
    if (s == "TWO_DIE_REGULAR") return Level0MeshType::TWO_DIE_REGULAR;
    if (s == "TWO_DIE_NOT_REGULAR") return Level0MeshType::TWO_DIE_NOT_REGULAR;
    return Level0MeshType::UNKNOWN;
}

static HcclReduceOp StrToReduceOp(const std::string& s) {
    if (s == "SUM") return HcclReduceOp::HCCL_REDUCE_SUM;
    if (s == "PROD") return HcclReduceOp::HCCL_REDUCE_PROD;
    if (s == "MAX") return HcclReduceOp::HCCL_REDUCE_MAX;
    if (s == "MIN") return HcclReduceOp::HCCL_REDUCE_MIN;
    return HcclReduceOp::HCCL_REDUCE_UNKNOWN;
}

static HcclDataType StrToDataType(const std::string& s) {
    if (s == "INT8") return HcclDataType::HCCL_DATA_TYPE_INT8;
    if (s == "INT32") return HcclDataType::HCCL_DATA_TYPE_INT32;
    if (s == "FP16") return HcclDataType::HCCL_DATA_TYPE_FP16;
    if (s == "BF16") return HcclDataType::HCCL_DATA_TYPE_BF16;
    if (s == "FP32") return HcclDataType::HCCL_DATA_TYPE_FP32;
    if (s == "INT64") return HcclDataType::HCCL_DATA_TYPE_INT64;
    if (s == "UINT64") return HcclDataType::HCCL_DATA_TYPE_UINT64;
    if (s == "FP64") return HcclDataType::HCCL_DATA_TYPE_FP64;
    return HcclDataType::HCCL_DATA_TYPE_UNKNOWN;
}

// ============================================================================
// FastAlgoSelector 静态工具方法
// ============================================================================

std::string FastAlgoSelector::Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

RuleMap FastAlgoSelector::ParseRuleBlock(const std::vector<std::string>& lines) {
    RuleMap rule;
    for (const auto& rawLine : lines) {
        std::string line = Trim(rawLine);
        if (line.empty() || line[0] == '#') continue;
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;
        std::string key = Trim(line.substr(0, eqPos));
        std::string value = Trim(line.substr(eqPos + 1));
        if (!key.empty() && !value.empty()) {
            rule[key] = value;
        }
    }
    return rule;
}

// ============================================================================
// 从 RuleMap 编译为 FastRule
// ============================================================================

FastRule FastAlgoSelector::CompileRule(const RuleMap& rule) {
    FastRule fr;
    fr.originalRule = rule;
    fr.algo = GetVal(rule, "_algo");

    // 拓扑层级
    if (HasKey(rule, "_topoLevel")) {
        std::string val = GetVal(rule, "_topoLevel");
        fr.topoLevel = (val == "multi") ? -1 : static_cast<int>(std::stoul(val));
    }
    // Level0 拓扑
    if (HasKey(rule, "_level0Topo")) {
        fr.level0Topo = StrToLevel0Shape(GetVal(rule, "_level0Topo"));
    }
    // Level0 Mesh 类型
    if (HasKey(rule, "_level0MeshType")) {
        fr.level0MeshType = StrToLevel0MeshType(GetVal(rule, "_level0MeshType"));
    }
    // 布尔条件
    if (HasKey(rule, "_level0PcieMix")) fr.level0PcieMix = GetBool(rule, "_level0PcieMix");
    if (HasKey(rule, "_is2DieFullMesh")) fr.is2DieFullMesh = GetBool(rule, "_is2DieFullMesh");
    if (HasKey(rule, "_level1Nhr")) fr.level1Nhr = GetBool(rule, "_level1Nhr");
    if (HasKey(rule, "_level0Nhr")) fr.level0Nhr = GetBool(rule, "_level0Nhr");
    if (HasKey(rule, "_meshEqClos")) fr.meshEqClos = GetBool(rule, "_meshEqClos");
    if (HasKey(rule, "_closMulMesh")) fr.closMulMesh = GetBool(rule, "_closMulMesh");
    if (HasKey(rule, "_overlap")) fr.overlap = GetBool(rule, "_overlap");
    if (HasKey(rule, "_hostToDevice")) fr.hostToDevice = GetBool(rule, "_hostToDevice");
    if (HasKey(rule, "_deviceToHost")) fr.deviceToHost = GetBool(rule, "_deviceToHost");
    // 本地网络实例数
    if (HasKey(rule, "_localNetIns")) fr.localNetIns = GetU32(rule, "_localNetIns");
    // 归约操作
    if (HasKey(rule, "_reduceOp")) fr.reduceOp = StrToReduceOp(GetVal(rule, "_reduceOp"));
    // 数据类型白名单
    if (HasKey(rule, "_dataTypes")) {
        auto tokens = SplitStr(GetVal(rule, "_dataTypes"), ',');
        for (auto& t : tokens) {
            // 去除空格
            size_t start = t.find_first_not_of(" ");
            size_t end = t.find_last_not_of(" ");
            if (start != std::string::npos) {
                t = t.substr(start, end - start + 1);
            }
            auto dt = StrToDataType(t);
            if (dt != HcclDataType::HCCL_DATA_TYPE_UNKNOWN) {
                fr.dataTypes.push_back(dt);
            }
        }
    }
    // 范围条件
    if (HasKey(rule, "_rankSizeMin")) { fr.rankSizeMin = GetU32(rule, "_rankSizeMin"); fr.hasRange = true; }
    if (HasKey(rule, "_rankSizeMax")) { fr.rankSizeMax = GetU32(rule, "_rankSizeMax"); fr.hasRange = true; }
    if (HasKey(rule, "_dataSizeMin")) { fr.dataSizeMin = GetU64(rule, "_dataSizeMin"); fr.hasRange = true; }
    if (HasKey(rule, "_dataSizeMax")) { fr.dataSizeMax = GetU64(rule, "_dataSizeMax"); fr.hasRange = true; }

    return fr;
}

// ============================================================================
// composite key 构建
// ============================================================================

std::string FastAlgoSelector::BuildCompositeKey(const FastRule& rule) {
    // 将规则中所有离散条件按固定顺序拼接为规范化字符串
    // 格式: "key1=val1|key2=val2|..."
    // 未设置的条件不参与 key 生成
    std::string key;
    auto append = [&](const std::string& k, const std::string& v) {
        if (!key.empty()) key += "|";
        key += k + "=" + v;
    };

    if (rule.topoLevel.has_value()) {
        append("_topoLevel", rule.topoLevel.value() == -1 ? "multi" : std::to_string(rule.topoLevel.value()));
    }
    if (rule.level0Topo.has_value()) {
        append("_level0Topo", Level0ShapeToStr(rule.level0Topo.value()));
    }
    if (rule.level0MeshType.has_value()) {
        append("_level0MeshType", Level0MeshTypeToStr(rule.level0MeshType.value()));
    }
    if (rule.level0PcieMix.has_value()) {
        append("_level0PcieMix", rule.level0PcieMix.value() ? "true" : "false");
    }
    if (rule.is2DieFullMesh.has_value()) {
        append("_is2DieFullMesh", rule.is2DieFullMesh.value() ? "true" : "false");
    }
    if (rule.level1Nhr.has_value()) {
        append("_level1Nhr", rule.level1Nhr.value() ? "true" : "false");
    }
    if (rule.level0Nhr.has_value()) {
        append("_level0Nhr", rule.level0Nhr.value() ? "true" : "false");
    }
    if (rule.meshEqClos.has_value()) {
        append("_meshEqClos", rule.meshEqClos.value() ? "true" : "false");
    }
    if (rule.closMulMesh.has_value()) {
        append("_closMulMesh", rule.closMulMesh.value() ? "true" : "false");
    }
    if (rule.localNetIns.has_value()) {
        append("_localNetIns", std::to_string(rule.localNetIns.value()));
    }
    if (rule.reduceOp.has_value()) {
        append("_reduceOp", ReduceOpToStr(rule.reduceOp.value()));
    }
    if (rule.overlap.has_value()) {
        append("_overlap", rule.overlap.value() ? "true" : "false");
    }
    if (rule.hostToDevice.has_value()) {
        append("_hostToDevice", rule.hostToDevice.value() ? "true" : "false");
    }
    if (rule.deviceToHost.has_value()) {
        append("_deviceToHost", rule.deviceToHost.value() ? "true" : "false");
    }
    if (!rule.dataTypes.empty()) {
        std::string dtStr;
        for (size_t i = 0; i < rule.dataTypes.size(); ++i) {
            if (i > 0) dtStr += ",";
            dtStr += DataTypeToStr(rule.dataTypes[i]);
        }
        append("_dataTypes", dtStr);
    }
    return key;
}

std::vector<std::string> FastAlgoSelector::GetDiscreteKeyNames(const RuleMap& rule) {
    static const std::vector<std::string> allDiscreteKeys = {
        "_topoLevel", "_level0Topo", "_level0MeshType", "_level0PcieMix",
        "_is2DieFullMesh", "_level1Nhr", "_level0Nhr", "_meshEqClos", "_closMulMesh",
        "_localNetIns", "_reduceOp", "_overlap", "_hostToDevice", "_deviceToHost",
        "_dataTypes"
    };
    std::vector<std::string> result;
    for (const auto& k : allDiscreteKeys) {
        if (HasKey(rule, k)) {
            result.push_back(k);
        }
    }
    return result;
}

std::string FastAlgoSelector::BuildQueryKey(const AlgoSelectContext& ctx,
                                             const std::vector<std::string>& keyNames) {
    // 根据指定的 key 名称列表，从 ctx 中提取对应值，生成与 BuildCompositeKey 相同格式的查询字符串
    std::string key;
    auto append = [&](const std::string& k, const std::string& v) {
        if (!key.empty()) key += "|";
        key += k + "=" + v;
    };

    for (const auto& k : keyNames) {
        if (k == "_topoLevel") {
            append(k, ctx.topoLevelNums > 1 ? "multi" : std::to_string(ctx.topoLevelNums));
        } else if (k == "_level0Topo") {
            append(k, Level0ShapeToStr(ctx.level0Topo));
        } else if (k == "_level0MeshType") {
            append(k, Level0MeshTypeToStr(ctx.level0MeshType));
        } else if (k == "_level0PcieMix") {
            append(k, ctx.level0PcieMix ? "true" : "false");
        } else if (k == "_is2DieFullMesh") {
            append(k, ctx.is2DieFullMesh ? "true" : "false");
        } else if (k == "_level1Nhr") {
            append(k, ctx.level1Nhr ? "true" : "false");
        } else if (k == "_level0Nhr") {
            append(k, ctx.level0Nhr ? "true" : "false");
        } else if (k == "_meshEqClos") {
            append(k, ctx.meshNumEqualToClosNum ? "true" : "false");
        } else if (k == "_closMulMesh") {
            append(k, ctx.closNumMultipleOfMeshNum ? "true" : "false");
        } else if (k == "_localNetIns") {
            append(k, std::to_string(ctx.localNetInsSizeOfLayer0));
        } else if (k == "_reduceOp") {
            append(k, ReduceOpToStr(ctx.reduceOp));
        } else if (k == "_overlap") {
            append(k, ctx.isInputOutputOverlap ? "true" : "false");
        } else if (k == "_hostToDevice") {
            append(k, ctx.isHostToDevice ? "true" : "false");
        } else if (k == "_deviceToHost") {
            append(k, ctx.isDeviceToHost ? "true" : "false");
        } else if (k == "_dataTypes") {
            append(k, DataTypeToStr(ctx.dataType));
        }
    }
    return key;
}

// ============================================================================
// 将规则索引到对应的桶中
// ============================================================================

void FastAlgoSelector::IndexRule(const RuleMap& rule, bool prepend) {
    std::string execStr = GetVal(rule, "_execConfig");
    std::string opStr = GetVal(rule, "_opType");
    if (execStr.empty() || opStr.empty() || !HasKey(rule, "_algo")) return;

    BucketKey bk = {execStr, opStr};

    // 编译规则
    FastRule fr = CompileRule(rule);
    fr.priority = nextPriority_++;
    fr.compositeKey = BuildCompositeKey(fr);

    // 存储到 fastRules_ 中（保证指针稳定性）
    fastRules_.push_back(std::move(fr));
    const FastRule* frPtr = &fastRules_.back();

    auto& bucket = buckets_[bk];
    if (!frPtr->hasRange) {
        // 纯离散条件 → 放入 exactMap
        if (prepend) {
            // prepend 时直接覆盖已有相同 compositeKey 的规则（外部优先）
            bucket.exactMap[frPtr->compositeKey] = frPtr;
        } else {
            // 非 prepend 时仅在不存在时插入
            if (bucket.exactMap.find(frPtr->compositeKey) == bucket.exactMap.end()) {
                bucket.exactMap[frPtr->compositeKey] = frPtr;
            }
        }
    } else {
        // 含范围条件 → 放入 rangeRules
        bucket.rangeRules.push_back(frPtr);
    }
}

// ============================================================================
// SelectAlgo：O(1) 桶查找 + O(1) 哈希精确匹配 + O(k) 范围回退
// ============================================================================

std::optional<std::string> FastAlgoSelector::SelectAlgo(const AlgoSelectContext& ctx) const {
    std::string execStr = ExecConfigToStr(ctx.execConfig);
    std::string opStr = CmdTypeToStr(ctx.opType);
    BucketKey bk = {execStr, opStr};

    auto bucketIt = buckets_.find(bk);
    if (bucketIt == buckets_.end()) {
        return std::nullopt;
    }

    const auto& bucket = bucketIt->second;

    // --- 第 1 层：exactMap 哈希精确匹配 ---
    // 需要尝试桶内所有 exactMap 条目的 key 组合（因为规则可能使用不同子集的离散条件）
    // 策略：遍历 exactMap，对每条规则用其自身的离散 key 集合构建查询 key
    const FastRule* bestExact = nullptr;
    int bestExactPriority = INT32_MAX;

    for (const auto& [compositeKey, rulePtr] : bucket.exactMap) {
        // 从规则的 originalRule 获取它使用的离散 key 名称
        auto keyNames = GetDiscreteKeyNames(rulePtr->originalRule);
        std::string queryKey = BuildQueryKey(ctx, keyNames);
        if (queryKey == compositeKey && rulePtr->MatchDiscrete(ctx)) {
            if (rulePtr->priority < bestExactPriority) {
                bestExactPriority = rulePtr->priority;
                bestExact = rulePtr;
            }
        }
    }

    // --- 第 2 层：rangeRules 顺序遍历 ---
    const FastRule* bestRange = nullptr;
    int bestRangePriority = INT32_MAX;

    for (const auto* rulePtr : bucket.rangeRules) {
        if (rulePtr->MatchFull(ctx)) {
            if (rulePtr->priority < bestRangePriority) {
                bestRangePriority = rulePtr->priority;
                bestRange = rulePtr;
            }
        }
    }

    // 选择优先级最高的（priority 值最小的）
    if (bestExact && bestRange) {
        return (bestExactPriority <= bestRangePriority) ? bestExact->algo : bestRange->algo;
    }
    if (bestExact) return bestExact->algo;
    if (bestRange) return bestRange->algo;

    return std::nullopt;
}

// ============================================================================
// Initialize：加载默认规则并构建索引
// ============================================================================

void FastAlgoSelector::Initialize() {
    buckets_.clear();
    fastRules_.clear();
    originalRules_.clear();
    nextPriority_ = 0;

    // 复用 TableBasedAlgoSelector 的默认规则
    TableBasedAlgoSelector base;
    base.Initialize();
    for (const auto& rule : base.GetRules()) {
        IndexRule(rule, false);
        originalRules_.push_back(rule);
    }
}

// ============================================================================
// InitializeWithConfig
// ============================================================================

void FastAlgoSelector::InitializeWithConfig(const std::string& configFilePath) {
    Initialize();
    if (!configFilePath.empty()) {
        LoadRulesFromFile(configFilePath);
    }
}

// ============================================================================
// LoadRulesFromFile
// ============================================================================

int FastAlgoSelector::LoadRulesFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[FastAlgoSelector] Failed to open config file: " << filePath << std::endl;
        return -1;
    }
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();
    return LoadRulesFromString(content);
}

// ============================================================================
// LoadRulesFromString
// ============================================================================

int FastAlgoSelector::LoadRulesFromString(const std::string& content) {
    std::istringstream stream(content);
    std::string line;
    std::vector<std::string> block;
    int count = 0;

    while (std::getline(stream, line)) {
        std::string trimmed = Trim(line);
        if (trimmed == "[rule]") {
            if (!block.empty()) {
                RuleMap rule = ParseRuleBlock(block);
                if (!rule.empty() && rule.count("_algo") > 0) {
                    PrependRule(rule);
                    count++;
                }
                block.clear();
            }
            continue;
        }
        if (!trimmed.empty() || !block.empty()) {
            block.push_back(line);
        }
    }
    if (!block.empty()) {
        RuleMap rule = ParseRuleBlock(block);
        if (!rule.empty() && rule.count("_algo") > 0) {
            PrependRule(rule);
            count++;
        }
    }
    if (count > 0) {
        std::cout << "[FastAlgoSelector] Loaded " << count << " external rules." << std::endl;
    }
    return count;
}

// ============================================================================
// AddRule / PrependRule
// ============================================================================

void FastAlgoSelector::AddRule(const RuleMap& rule) {
    IndexRule(rule, false);
    originalRules_.push_back(rule);
}

void FastAlgoSelector::PrependRule(const RuleMap& rule) {
    // 外部规则使用更低的 priority 值，确保优先匹配
    IndexRule(rule, true);
    originalRules_.insert(originalRules_.begin(), rule);
}

// ============================================================================
// DumpTable
// ============================================================================

void FastAlgoSelector::DumpTable() const {
    std::cout << "=== FastAlgoSelector (" << fastRules_.size() << " rules, "
              << buckets_.size() << " buckets) ===" << std::endl;
    for (const auto& [bk, bucket] : buckets_) {
        std::cout << "Bucket [" << bk.first << " | " << bk.second << "]:" << std::endl;
        std::cout << "  exact (" << bucket.exactMap.size() << "):" << std::endl;
        for (const auto& [key, rulePtr] : bucket.exactMap) {
            std::cout << "    P" << rulePtr->priority << " key=\""
                      << (key.empty() ? "(empty)" : key) << "\" -> " << rulePtr->algo << std::endl;
        }
        std::cout << "  range (" << bucket.rangeRules.size() << "):" << std::endl;
        for (const auto* rulePtr : bucket.rangeRules) {
            std::cout << "    P" << rulePtr->priority << " -> " << rulePtr->algo << std::endl;
        }
    }
}

} // namespace ops_hccl
