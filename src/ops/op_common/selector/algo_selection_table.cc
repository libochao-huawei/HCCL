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

} // namespace ops_hccl