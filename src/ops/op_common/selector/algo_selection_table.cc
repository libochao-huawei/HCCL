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
#include "log.h"
#include <algorithm>

namespace ops_hccl {

TableBasedAlgoSelector::TableBasedAlgoSelector() {
    Initialize();
}

void TableBasedAlgoSelector::AddRule(OpExecuteConfig execConfig,
                                     HcclCMDType opType,
                                     TopoLevelCategory topoLevel,
                                     Level0Shape level0Topo,
                                     const AlgoRule& rule) {
    table_[execConfig][opType][topoLevel][level0Topo].push_back(rule);
}

void TableBasedAlgoSelector::AddRuleForAllTopo(OpExecuteConfig execConfig,
                                               HcclCMDType opType,
                                               TopoLevelCategory topoLevel,
                                               const AlgoRule& rule) {
    // 对所有 Level0 拓扑类型添加规则
    for (auto shape : {Level0Shape::MESH_1D, Level0Shape::MESH_1D_CLOS, Level0Shape::CLOS}) {
        AddRule(execConfig, opType, topoLevel, shape, rule);
    }
}

std::optional<std::string> TableBasedAlgoSelector::SelectAlgo(const AlgoSelectContext& ctx) const {
    // 第一层查找：执行配置
    auto configIt = table_.find(ctx.execConfig);
    if (configIt == table_.end()) {
        HCCL_DEBUG("[AlgoSelector] No rules for execConfig=%d", static_cast<int>(ctx.execConfig));
        return std::nullopt;
    }
    
    // 第二层查找：操作类型
    auto opIt = configIt->second.find(ctx.opType);
    if (opIt == configIt->second.end()) {
        HCCL_DEBUG("[AlgoSelector] No rules for opType=%d", static_cast<int>(ctx.opType));
        return std::nullopt;
    }
    
    // 第三层查找：拓扑层级
    TopoLevelCategory levelCat = CategorizeTopoLevel(ctx.topoLevelNums);
    auto levelIt = opIt->second.find(levelCat);
    if (levelIt == opIt->second.end()) {
        // 尝试查找 ANY 层级的规则
        levelIt = opIt->second.find(TopoLevelCategory::ANY);
        if (levelIt == opIt->second.end()) {
            HCCL_DEBUG("[AlgoSelector] No rules for topoLevel=%d", static_cast<int>(levelCat));
            return std::nullopt;
        }
    }
    
    // 第四层查找：Level0 拓扑类型
    auto topoIt = levelIt->second.find(ctx.level0Topo);
    if (topoIt == levelIt->second.end()) {
        HCCL_DEBUG("[AlgoSelector] No rules for level0Topo=%d", static_cast<int>(ctx.level0Topo));
        return std::nullopt;
    }
    
    // 第五层：遍历规则列表，找到第一个匹配的规则
    for (const auto& rule : topoIt->second) {
        if (rule.checker && rule.checker(ctx)) {
            HCCL_DEBUG("[AlgoSelector] Rule '%s' matched, selecting algo: %s",
                       rule.name.c_str(), rule.algoName.c_str());
            return rule.algoName;
        }
    }
    
    HCCL_DEBUG("[AlgoSelector] No rule matched");
    return std::nullopt;
}

void TableBasedAlgoSelector::DumpTable() const {
    HCCL_INFO("[AlgoSelector] Dumping algorithm selection table:");
    for (const auto& [config, opRules] : table_) {
        HCCL_INFO("  ExecConfig=%d:", static_cast<int>(config));
        for (const auto& [opType, levelRules] : opRules) {
            HCCL_INFO("    OpType=%d:", static_cast<int>(opType));
            for (const auto& [level, topoRules] : levelRules) {
                const char* levelStr = (level == TopoLevelCategory::SINGLE_LAYER) ? "SINGLE" :
                                       (level == TopoLevelCategory::MULTI_LAYER) ? "MULTI" : "ANY";
                HCCL_INFO("      TopoLevel=%s:", levelStr);
                for (const auto& [topo, rules] : topoRules) {
                    HCCL_INFO("        Level0Topo=%d: %zu rules", static_cast<int>(topo), rules.size());
                }
            }
        }
    }
}

// ============================================================================
// 初始化查找表 - 填充所有算法规则
// ============================================================================
void TableBasedAlgoSelector::Initialize() {
    using namespace Conditions;
    
    // ========================================================================
    // CCU_MS 模式规则
    // ========================================================================
    
    // AllReduce - CCU_MS - 单层级 - Mesh
    {
        // 规则1: 小数据量，RankSize < 8
        AddRule(OpExecuteConfig::CCU_MS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("CCU_MS_Mesh_SmallData_SmallRank",
                AllOf({DataSizeBelow(512 * KB), RankSizeBelow(8)}),
                "ccu_ms_mesh_small_small"));
        
        // 规则2: 小数据量，RankSize >= 8
        AddRule(OpExecuteConfig::CCU_MS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("CCU_MS_Mesh_SmallData_LargeRank",
                AllOf({DataSizeBelow(512 * KB), RankSizeAtLeast(8)}),
                "ccu_ms_mesh_small_large"));
        
        // 规则3: 中等数据量
        AddRule(OpExecuteConfig::CCU_MS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("CCU_MS_Mesh_MediumData",
                AllOf({DataSizeInRange(512 * KB, 8 * MB), IsNotPcieMix}),
                "ccu_ms_mesh_medium"));
        
        // 规则4: 大数据量
        AddRule(OpExecuteConfig::CCU_MS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("CCU_MS_Mesh_LargeData",
                DataSizeAbove(8 * MB),
                "ccu_ms_mesh_large"));
    }
    
    // AllReduce - CCU_MS - 单层级 - Clos
    {
        AddRule(OpExecuteConfig::CCU_MS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::CLOS,
            AlgoRule("CCU_MS_Clos_Default",
                Always,
                "ccu_ms_clos_default"));
    }
    
    // ========================================================================
    // CCU_SCHED 模式规则
    // ========================================================================
    
    // AllReduce - CCU_SCHED - 单层级
    {
        // 双 Die 全连接 Mesh
        AddRule(OpExecuteConfig::CCU_SCHED, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("CCU_SCHED_2DieFullMesh",
                Is2DieFullMesh,
                "ccu_sched_2die_fullmesh"));
        
        // 默认规则
        AddRule(OpExecuteConfig::CCU_SCHED, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("CCU_SCHED_Default",
                Always,
                "ccu_sched_default"));
    }
    
    // ========================================================================
    // AICPU_TS 模式规则
    // ========================================================================
    
    // AllReduce - AICPU_TS - 单层级
    {
        // 64 位数据类型或 PROD 归约
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AICPU_TS_SpecialType",
                AnyOf({Is64BitDataType, IsReduceProd}),
                "aicpu_special_type"));
        
        // 小数据量
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AICPU_TS_SmallData",
                DataSizeBelow(512 * KB),
                "aicpu_small_data"));
        
        // 默认
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AICPU_TS_Default",
                Always,
                "aicpu_default"));
    }
    
    // AllReduce - AICPU_TS - 多层级
    {
        // Level1 NHR
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::MULTI_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AICPU_TS_Level1Nhr",
                IsLevel1Nhr,
                "aicpu_level1_nhr"));
        
        // 默认
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLREDUCE, TopoLevelCategory::MULTI_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AICPU_TS_MultiLayer_Default",
                Always,
                "aicpu_multilayer_default"));
    }
    
    // ========================================================================
    // Broadcast 规则
    // ========================================================================
    
    // Broadcast - AICPU_TS - 单层级
    {
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_BROADCAST, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("Broadcast_OneShot",
                DataSizeBelow(1 * MB),
                "broadcast_oneshot"));
        
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_BROADCAST, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("Broadcast_TwoShot",
                Always,
                "broadcast_twoshot"));
    }
    
    // ========================================================================
    // AllGather 规则
    // ========================================================================
    
    // AllGather - AICPU_TS - 单层级
    {
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLGATHER, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AllGather_SmallData",
                DataSizeBelow(256 * KB),
                "allgather_small"));
        
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLGATHER, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AllGather_Default",
                Always,
                "allgather_default"));
    }
    
    // ========================================================================
    // ReduceScatter 规则
    // ========================================================================
    
    // ReduceScatter - AICPU_TS - 单层级
    {
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_REDUCE_SCATTER, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("ReduceScatter_SmallData",
                DataSizeBelow(256 * KB),
                "reducescatter_small"));
        
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_REDUCE_SCATTER, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("ReduceScatter_Default",
                Always,
                "reducescatter_default"));
    }
    
    // ========================================================================
    // AlltoAll 规则
    // ========================================================================
    
    // AlltoAll - AICPU_TS - 单层级
    {
        // PCIE 混合拓扑
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLTOALL, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AlltoAll_PcieMix",
                IsPcieMix,
                "alltoall_pcie"));
        
        AddRule(OpExecuteConfig::AICPU_TS, HCCL_CMD_ALLTOALL, TopoLevelCategory::SINGLE_LAYER, Level0Shape::MESH_1D,
            AlgoRule("AlltoAll_Default",
                Always,
                "alltoall_default"));
    }
    
    // ========================================================================
    // Send/Recv 规则
    // ========================================================================
    
    // Send - HOSTCPU - Host DPU
    {
        AddRule(OpExecuteConfig::HOSTCPU, HCCL_CMD_SEND, TopoLevelCategory::ANY, Level0Shape::MESH_1D,
            AlgoRule("Send_HostToDevice",
                IsHostToDevice,
                "send_host_dpu"));
        
        AddRule(OpExecuteConfig::HOSTCPU, HCCL_CMD_SEND, TopoLevelCategory::ANY, Level0Shape::MESH_1D,
            AlgoRule("Send_Default",
                Always,
                "send_device_dpu"));
    }
    
    // Recv - HOSTCPU - Host DPU
    {
        AddRule(OpExecuteConfig::HOSTCPU, HCCL_CMD_RECV, TopoLevelCategory::ANY, Level0Shape::MESH_1D,
            AlgoRule("Recv_DeviceToHost",
                IsDeviceToHost,
                "recv_host_dpu"));
        
        AddRule(OpExecuteConfig::HOSTCPU, HCCL_CMD_RECV, TopoLevelCategory::ANY, Level0Shape::MESH_1D,
            AlgoRule("Recv_Default",
                Always,
                "recv_device_dpu"));
    }
    
    HCCL_INFO("[AlgoSelector] Algorithm selection table initialized");
}

} // namespace ops_hccl

/*
## src/ops 目录下算法选择条件总结

通过分析 src/ops 目录下所有 selector 文件夹中的算法选择文件（共 15 个文件），总结出共有 **15 种主要条件类型** 用于算法选择：

### 一、拓扑相关条件（7种）

1. **拓扑层级数 (topoLevelNums)**
   - 判断是否跨多个层级（单层级 vs 多层级）
   - 例如：`topoInfo->topoLevelNums > 1`

2. **Level0 拓扑类型 (level0Topo)**
   - `MESH_1D`: 一维 Mesh 拓扑
   - `MESH_1D_CLOS`: Mesh + Clos 混合拓扑
   - `CLOS`: Clos 拓扑

3. **Level0 Mesh 类型 (level0MeshType)**
   - `TWO_DIE_REGULAR`: 双 Die 规则 Mesh
   - `TWO_DIE_NOT_REGULAR`: 双 Die 非规则 Mesh

4. **PCIE 混合拓扑标志 (level0PcieMix)**
   - 判断是否为 PCIE-SW 定制机型

5. **两 Die 全连接 Mesh (is2DieFullMesh)**
   - 判断是否为双 Die 全连接 Mesh 拓扑

6. **Mesh 数量与 Clos 数量关系**
   - `isMeshNumEqualToClosNum`: Mesh 数量是否等于 Clos 数量
   - `isClosNumMultipleOfMeshNum`: Clos 数量是否为 Mesh 数量的倍数

7. **网络层详情 (netLayerDetails)**
   - `localNetInsSizeOfLayer[0]`: 第一层的本地网络实例大小
   - `Level1Nhr`: Level1 是否为 NHR 拓扑

### 二、数据相关条件（2种）

8. **数据大小 (dataSize)**
   - 小数据量：< 512KB (SMALL_COUNT_512KB)
   - 大数据量：>= 1024KB (LARGE_COUNT_1024KB)
   - CCU 并行最大数据量：<= 64MB

9. **数据类型 (dataType)**
   - 64 位数据类型：`INT64, UINT64, FP64`
   - 特殊类型：`INT8`

### 三、操作相关条件（2种）

10. **归约操作类型 (reduceType)**
    - 特别关注 `HCCL_REDUCE_PROD` (乘法归约)

11. **操作类型 (opType)**
    - 如 `HCCL_CMD_ALLTOALL`, `HCCL_CMD_ALLTOALLV` 等

### 四、执行模式条件（1种）

12. **执行配置 (opExecuteConfig)**
    - `CCU_MS`: CCU Mesh 模式
    - `CCU_SCHED`: CCU Schedule 模式
    - `AICPU_TS`: AICPU 模式
    - `AIV / AIV_ONLY`: AIV 模式
    - `HOSTCPU`: 主机 CPU 模式

### 五、其他条件（3种）

13. **Rank 数量 (userRankSize)**
    - 判断是否在 4P 范围内（<= 4）
    - 用于选择并发算法

14. **输入输出内存重叠 (IsInputOutputOverlap)**
    - 判断输入输出缓冲区是否重叠（inplace 场景）

15. **链路端点位置类型 (locType)**
    - `ENDPOINT_LOC_TYPE_HOST`: 主机端
    - `ENDPOINT_LOC_TYPE_DEVICE`: 设备端
    - 用于 Send/Recv 选择 Host DPU 或 Device DPU 算法

### 总结

算法选择系统采用**多层次决策树结构**，按优先级依次检查上述条件，为不同硬件配置和操作参数自动选择最优通信算法。*/