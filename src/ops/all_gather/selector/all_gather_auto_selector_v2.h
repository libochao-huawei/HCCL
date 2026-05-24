/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ALLGATHER_AUTO_SELECTOR_V2
#define HCCLV2_ALLGATHER_AUTO_SELECTOR_V2

#include "rule_based_selector.h"

namespace ops_hccl {

/**
 * @brief 使用规则链重构的 AllGather 算法选择器
 * 
 * 相比原版本，新增算法选择分支只需在 BuildXxxRules() 方法中添加一行规则，
 * 无需编写复杂的 if-else 嵌套逻辑。
 */
class AllGatherAutoSelectorV2 : public RuleBasedSelector {
protected:
    /**
     * @brief 构建 CCU Schedule 模式的规则链
     * 
     * 规则按优先级顺序添加，先匹配的规则优先
     */
    void BuildCcuScheduleRules(RuleChain& chain) override {
        // ============ 多级拓扑规则 ============
        
        // 规则1: 多级 + MESH_1D + Level1Nhr
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithLevel1Nhr(),
            "CCU_MultiLevel_Mesh1D_Level1Nhr", "CcuAllGatherNHR1DMem2Mem");
        
        // 规则2: 多级 + MESH_1D + localNetInsSize == 1
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithLocalNetInsSizeOfLayer0(1),
            "CCU_MultiLevel_Mesh1D_LocalNetIns1", "CcuAllGatherNHR1DMem2Mem");
        
        // 规则3: 多级 + MESH_1D + 小数据 + 小规模
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithDataSizeLessThan(AG_FLATTEN_MAX_DATA_SIZE)
            .WithRankSizeLessOrEqual(64),
            "CCU_MultiLevel_Mesh1D_SmallData", "CcuAllGatherMesh1DMem2Mem");
        
        // 规则4: 多级 + MESH_1D + 大数据
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D),
            "CCU_MultiLevel_Mesh1D_LargeData", "CcuAllGatherParallelMesh1DNHR");
        
        // 规则5: 多级 + CLOS
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::CLOS),
            "CCU_MultiLevel_CLOS", "CcuAllGatherNHR1DMem2Mem");
        
        // ============ 单级拓扑规则 ============
        
        // 规则6: 单级 + MESH_1D + 2DieRegular
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithLevel0MeshType(Level0MeshType::TWO_DIE_REGULAR),
            "CCU_SingleLevel_Mesh1D_2DieRegular", "CcuAllGatherMesh2Die");
        
        // 规则7: 单级 + MESH_1D
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D),
            "CCU_SingleLevel_Mesh1D", "CcuAllGatherMesh1DMem2Mem");
        
        // 规则8: 单级 + MESH_1D_CLOS + PCIe混合 + 全连接
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithPcieMix()
            .WithCustomCondition([this](const RuleContext& ctx) {
                return IsLayerAllConnetedWithTopo(ctx.topoInfo, 0, CommTopo::COMM_TOPO_1DMESH);
            }),
            "CCU_SingleLevel_Mesh1DClos_PcieMix_FullConn", "CcuAllGatherMesh1DMem2Mem");
        
        // 规则9: 单级 + MESH_1D_CLOS + UBX机型 + 大数据 + 特定条件
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(SMALL_COUNT_512KB)
            .WithCustomCondition([this](const RuleContext& ctx) {
                bool isEqual = false;
                CheckMeshNumEqualToClosNum(ctx.topoInfo, isEqual);
                return isEqual && ctx.topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO;
            }),
            "CCU_SingleLevel_Mesh1DClos_UBX_Concurrent", "CcuAllGatherConcurrentMesh1DNHRMem");
        
        // 规则10: 单级 + MESH_1D_CLOS + UBX机型 + 大数据 + ClosNum是MeshNum倍数
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(SMALL_COUNT_512KB)
            .WithCustomCondition([this](const RuleContext& ctx) {
                bool isMultiple = false;
                CheckClosNumMultipleOfMeshNum(ctx.topoInfo, isMultiple);
                return isMultiple;
            }),
            "CCU_SingleLevel_Mesh1DClos_UBX_Parallel", "CcuAllGatherParallelMesh1DNHRMemMultiJetty");
        
        // 规则11: 单级 + MESH_1D_CLOS + UBX机型 + 大数据 + 默认
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(SMALL_COUNT_512KB),
            "CCU_SingleLevel_Mesh1DClos_UBX_Default", "CcuAllGatherNHR1DMem2MemMultiJetty");
        
        // 规则12: 单级 + MESH_1D_CLOS + UBX机型 + 小数据
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix(),
            "CCU_SingleLevel_Mesh1DClos_UBX_SmallData", "CcuAllGatherMesh1DMem2Mem");
        
        // 规则13: 单级 + CLOS + 大数据
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(AG_CCU_CLOS_SMALL_DATA_SIZE),
            "CCU_SingleLevel_CLOS_LargeData", "CcuAllGatherMesh1DMem2Mem");
        
        // 规则14: 单级 + CLOS + 小数据
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::CLOS)
            .WithoutPcieMix(),
            "CCU_SingleLevel_CLOS_SmallData", "CcuAllGatherNHR1DMem2Mem");
    }
    
    /**
     * @brief 构建 AICPU 模式的规则链
     */
    void BuildAicpuRules(RuleChain& chain) override {
        // ============ 多级拓扑规则 ============
        
        // 规则1: 多级 + Level1Nhr
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel1Nhr(),
            "AICPU_MultiLevel_Level1Nhr", "InsAllGatherNHR");
        
        // 规则2: 多级 + Level0Nhr
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Nhr(),
            "AICPU_MultiLevel_Level0Nhr", "InsAllGatherNHR");
        
        // 规则3: 多级 + localNetInsSize == 1
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLocalNetInsSizeOfLayer0(1),
            "AICPU_MultiLevel_LocalNetIns1", "InsAllGatherNHR");
        
        // 规则4: 多级 + MESH_1D + 大数据 + 超大数据量
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithDataSizeGreaterThan(AG_AICPU_SMALL_DATA_SIZE)
            .WithCustomCondition([](const RuleContext& ctx) {
                return ctx.dataSize * ctx.topoInfo->userRankSize > AG_AICPU_SEQUENCE_DATA_SIZE;
            }),
            "AICPU_MultiLevel_Mesh1D_SequenceNHR", "InsAllGatherSequenceNHRMesh1D");
        
        // 规则5: 多级 + MESH_1D + 大数据
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithDataSizeGreaterThan(AG_AICPU_SMALL_DATA_SIZE),
            "AICPU_MultiLevel_Mesh1D_Parallel", "InsAllGatherParallelMesh1DNHR");
        
        // 规则6: 多级 + MESH_1D + 小数据
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D),
            "AICPU_MultiLevel_Mesh1D_SmallData", "InsAllGatherNHR");
        
        // 规则7: 多级 + CLOS
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::CLOS),
            "AICPU_MultiLevel_CLOS", "InsAllGatherNHR");
        
        // ============ 单级拓扑规则 ============
        
        // 规则8: 单级 + MESH_1D + 二级网络 + 超大数据
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithCustomCondition([this](const RuleContext& ctx) {
                return IsTwoLevelNetLayer(ctx.topoInfo) && 
                       ctx.dataSize * ctx.topoInfo->userRankSize > AG_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD;
            }),
            "AICPU_SingleLevel_Mesh1D_2Level_ZAxisDetour", "InsAllGatherMesh1D1DZAxisDetour");
        
        // 规则9: 单级 + MESH_1D
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D),
            "AICPU_SingleLevel_Mesh1D", "InsAllGatherMesh1D");
        
        // 规则10: 单级 + MESH_1D_CLOS + PCIe混合 + 全连接
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithPcieMix()
            .WithCustomCondition([this](const RuleContext& ctx) {
                return IsLayerAllConnetedWithTopo(ctx.topoInfo, 0, CommTopo::COMM_TOPO_1DMESH);
            }),
            "AICPU_SingleLevel_Mesh1DClos_PcieMix_FullConn", "InsAllGatherMesh1D");
        
        // 规则11: 单级 + MESH_1D_CLOS + PCIe混合 + 非全连接
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithPcieMix(),
            "AICPU_SingleLevel_Mesh1DClos_PcieMix_NotFullConn", "InsAllGatherParallelMesh1DNHRPcie");
        
        // 规则12: 单级 + MESH_1D_CLOS + UBX + 大数据 + 并发条件
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(SMALL_COUNT_512KB)
            .WithCustomCondition([this](const RuleContext& ctx) {
                bool isEqual = false;
                CheckMeshNumEqualToClosNum(ctx.topoInfo, isEqual);
                return isEqual && ctx.topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO;
            }),
            "AICPU_SingleLevel_Mesh1DClos_UBX_Concurrent", "InsAllGatherConcurrentMesh1DNHR");
        
        // 规则13: 单级 + MESH_1D_CLOS + UBX + 小数据 + 并发条件
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithCustomCondition([this](const RuleContext& ctx) {
                bool isEqual = false;
                CheckMeshNumEqualToClosNum(ctx.topoInfo, isEqual);
                return isEqual && ctx.topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO;
            }),
            "AICPU_SingleLevel_Mesh1DClos_UBX_SmallConcurrent", "InsAllGatherMesh1D");
        
        // 规则14: 单级 + MESH_1D_CLOS + UBX + 大数据 + ClosNum倍数
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(SMALL_COUNT_512KB)
            .WithCustomCondition([this](const RuleContext& ctx) {
                bool isMultiple = false;
                CheckClosNumMultipleOfMeshNum(ctx.topoInfo, isMultiple);
                return isMultiple;
            }),
            "AICPU_SingleLevel_Mesh1DClos_UBX_Parallel", "InsAllGatherParallelMesh1DNHRMultiJetty");
        
        // 规则15: 单级 + MESH_1D_CLOS + UBX + 默认
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix(),
            "AICPU_SingleLevel_Mesh1DClos_UBX_Default", "InsAllGatherNHR");
        
        // 规则16: 单级 + CLOS
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::CLOS),
            "AICPU_SingleLevel_CLOS", "InsAllGatherNHR");
    }
    
    /**
     * @brief 构建 CCU MS 模式的规则链
     */
    void BuildCcuMsRules(RuleChain& chain) override {
        // 规则1: 单级 + MESH_1D
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D),
            "CCUMS_SingleLevel_Mesh1D", "CcuAllGatherMesh1D");
        
        // 规则2: 单级 + MESH_1D_CLOS + PCIe混合 + 全连接
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithPcieMix()
            .WithCustomCondition([this](const RuleContext& ctx) {
                return IsLayerAllConnetedWithTopo(ctx.topoInfo, 0, CommTopo::COMM_TOPO_1DMESH);
            }),
            "CCUMS_SingleLevel_Mesh1DClos_PcieMix_FullConn", "CcuAllGatherMesh1D");
        
        // 规则3: 单级 + MESH_1D_CLOS + UBX + 大数据 + 并发条件
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix()
            .WithDataSizeGreaterThan(SMALL_COUNT_512KB)
            .WithCustomCondition([this](const RuleContext& ctx) {
                bool isEqual = false;
                CheckMeshNumEqualToClosNum(ctx.topoInfo, isEqual);
                return isEqual && ctx.topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO;
            }),
            "CCUMS_SingleLevel_Mesh1DClos_UBX_Concurrent", "CcuAllGatherConcurrentMesh1DNHR");
        
        // 规则4: 单级 + MESH_1D_CLOS + UBX + 小数据
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS)
            .WithoutPcieMix(),
            "CCUMS_SingleLevel_Mesh1DClos_UBX_SmallData", "CcuAllGatherMesh1D");
    }
    
    /**
     * @brief 构建 AIV 模式的规则链
     */
    void BuildAivRules(RuleChain& chain) override {
        // 规则1: rankSize 限制
        chain.AddRule(ConditionBuilder()
            .WithRankSizeLessOrEqual(MAX_RANK_SIZE),
            "AIV_Default", "AivAllGatherMesh1D");
    }
    
    /**
     * @brief 构建 DPU 模式的规则链
     */
    void BuildDPURules(RuleChain& chain) override {
        // 规则1: 多级 + localNetInsSize == 1
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLocalNetInsSizeOfLayer0(1),
            "DPU_MultiLevel_LocalNetIns1", "InsAllGatherMeshNhrDPU");
        
        // 规则2: 多级 + MESH_1D
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D),
            "DPU_MultiLevel_Mesh1D", "InsAllGatherMeshNhrDPU");
        
        // 规则3: 多级 + MESH_1D_CLOS
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D_CLOS),
            "DPU_MultiLevel_Mesh1DClos", "InsV2AllGatherOmniPipe");
    }

private:
    // 常量定义
    static constexpr u64 AG_FLATTEN_MAX_DATA_SIZE = 1 * 1024 * 1024;
    static constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
    static constexpr u64 AG_CCU_CLOS_SMALL_DATA_SIZE = 1 * 1024 * 1024;
    static constexpr u64 AG_AICPU_SMALL_DATA_SIZE = 1 * 1024 * 1024;
    static constexpr u64 AG_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD = 1 * 1024 * 1024 * 1024;
    static constexpr u64 AG_AICPU_SEQUENCE_DATA_SIZE = 1 * 1024 * 1024 * 1024;
};

}  // namespace ops_hccl

#endif  // HCCLV2_ALLGATHER_AUTO_SELECTOR_V2
