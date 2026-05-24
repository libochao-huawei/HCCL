# 基于规则链的算法选择器设计文档

## 1. 设计背景

原有的 `AllGatherAutoSelector` 存在以下问题：
- 大量嵌套的 if-else 逻辑，代码可读性差
- 新增算法选择分支需要修改多个函数
- 条件判断重复，难以维护
- 难以进行单元测试

## 2. 新设计方案：规则链模式

### 2.1 核心思想

将每个算法选择条件封装为独立的**规则（Rule）**，按优先级顺序组成**规则链（RuleChain）**。
算法选择时依次尝试匹配规则，第一个匹配的规则即为最终选择的算法。

### 2.2 核心组件

| 组件 | 说明 |
|------|------|
| `RuleResult` | 规则匹配结果，包含是否匹配和算法名称 |
| `RuleContext` | 规则匹配上下文，包含拓扑信息、操作参数等 |
| `AlgoRule` | 规则基类，定义 TryMatch 接口 |
| `FunctionalRule` | 函数式规则实现，支持 lambda 表达式 |
| `ConditionBuilder` | 条件构建器，提供声明式的条件构建 API |
| `RuleChain` | 规则链，按优先级顺序执行规则匹配 |
| `RuleBasedSelector` | 基于规则的选择器基类 |

### 2.3 架构关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    RuleBasedSelector                             │
│  - ccuScheduleRules_                                            │
│  - ccuMsRules_                                                  │
│  - aicpuRules_                                                  │
│  - aivRules_                                                    │
│  - dpuRules_                                                    │
│                                                                  │
│  + BuildCcuScheduleRules(chain)  // 子类实现                     │
│  + BuildAicpuRules(chain)        // 子类实现                     │
│  + ...                                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 包含
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RuleChain                                  │
│  - rules_: vector<unique_ptr<AlgoRule>>                         │
│                                                                  │
│  + AddRule(rule)                                                │
│  + Execute(context) -> RuleResult                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 包含
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AlgoRule                                  │
│  + TryMatch(context) -> RuleResult                              │
│  + GetRuleName() -> string                                      │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 使用方法

### 3.1 定义新的算法选择器

继承 `RuleBasedSelector` 并重写 `BuildXxxRules()` 方法：

```cpp
#include "rule_based_selector.h"

namespace ops_hccl {

class MyAlgoSelector : public RuleBasedSelector {
protected:
    void BuildCcuScheduleRules(RuleChain& chain) override {
        // 规则1: 多级拓扑 + MESH_1D + Level1Nhr
        chain.AddRule(ConditionBuilder()
            .WithMultiLevel()
            .WithLevel0Topo(Level0Shape::MESH_1D)
            .WithLevel1Nhr(),
            "MyRule_MultiLevel_NHR", 
            "MyAlgoNHR1D");
        
        // 规则2: 单级拓扑 + 大数据
        chain.AddRule(ConditionBuilder()
            .WithSingleLevel()
            .WithDataSizeGreaterThan(1024 * 1024),
            "MyRule_SingleLevel_LargeData", 
            "MyAlgoLargeData");
        
        // 规则3: 默认规则
        chain.AddRule(ConditionBuilder(),
            "MyRule_Default", 
            "MyAlgoDefault");
    }
};

}  // namespace ops_hccl
```

### 3.2 添加新算法分支

**只需在相应的 `BuildXxxRules()` 方法中添加一行规则**：

```cpp
// 新增：支持新拓扑类型 NEW_TOPO
chain.AddRule(ConditionBuilder()
    .WithLevel0Topo(Level0Shape::NEW_TOPO)
    .WithDataSizeGreaterThan(THRESHOLD),
    "Rule_NewTopo_LargeData", 
    "NewAlgoForNewTopo");
```

### 3.3 ConditionBuilder 可用的条件方法

| 方法 | 说明 |
|------|------|
| `WithTopoLevelNums(n)` | 拓扑层级等于 n |
| `WithTopoLevelNumsGreaterThan(n)` | 拓扑层级大于 n |
| `WithSingleLevel()` | 单级拓扑 (levelNums == 1) |
| `WithMultiLevel()` | 多级拓扑 (levelNums > 1) |
| `WithLevel0Topo(topo)` | Level0 拓扑类型 |
| `WithLevel0MeshType(type)` | Level0 Mesh 类型 |
| `WithDataSizeGreaterThan(size)` | 数据量大于 size |
| `WithDataSizeLessThan(size)` | 数据量小于等于 size |
| `WithDataSizeInRange(min, max)` | 数据量在范围内 |
| `WithRankSizeLessOrEqual(n)` | Rank 数量小于等于 n |
| `WithRankSizeGreaterThan(n)` | Rank 数量大于 n |
| `WithPcieMix()` | PCIe 混合拓扑 |
| `WithoutPcieMix()` | 非 PCIe 混合拓扑 |
| `WithLevel1Nhr()` | Level1 NHR |
| `WithLevel0Nhr()` | Level0 NHR |
| `With2DieFullMesh()` | 2 Die Full Mesh |
| `WithLocalNetInsSizeOfLayer0(n)` | Layer0 本地网络实例数 |
| `WithCustomCondition(pred)` | 自定义条件 (lambda) |

### 3.4 使用自定义条件

对于复杂条件，可以使用 `WithCustomCondition`：

```cpp
chain.AddRule(ConditionBuilder()
    .WithSingleLevel()
    .WithCustomCondition([this](const RuleContext& ctx) {
        // 可以调用 selector 的辅助方法
        bool isEqual = false;
        CheckMeshNumEqualToClosNum(ctx.topoInfo, isEqual);
        return isEqual && ctx.topoInfo->userRankSize <= 4;
    }),
    "Rule_ComplexCondition", 
    "ComplexAlgo");
```

## 4. 对比：旧方案 vs 新方案

### 4.1 旧方案（if-else 嵌套）

```cpp
SelectorStatus SelectCcuScheduleAlgo(...) const {
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->Level1Nhr) {
                selectAlgName = "Algo1";
                return SelectorStatus::MATCH;
            } else if (topoInfo->is2DieFullMesh) {
                return SelectorStatus::NOT_MATCH;
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "Algo2";
                return SelectorStatus::MATCH;
            } else if (dataSize < threshold && topoInfo->userRankSize <= 64) {
                selectAlgName = "Algo3";
                return SelectorStatus::MATCH;
            } else {
                selectAlgName = "Algo4";
                return SelectorStatus::MATCH;
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "Algo5";
        }
        // ... 更多嵌套
    } else {
        // 单级拓扑逻辑...
    }
}
```

### 4.2 新方案（规则链）

```cpp
void BuildCcuScheduleRules(RuleChain& chain) override {
    chain.AddRule(ConditionBuilder()
        .WithMultiLevel()
        .WithLevel0Topo(Level0Shape::MESH_1D)
        .WithLevel1Nhr(),
        "Rule_MultiLevel_NHR", "Algo1");
    
    chain.AddRule(ConditionBuilder()
        .WithMultiLevel()
        .WithLevel0Topo(Level0Shape::MESH_1D)
        .WithLocalNetInsSizeOfLayer0(1),
        "Rule_MultiLevel_LocalNetIns1", "Algo2");
    
    chain.AddRule(ConditionBuilder()
        .WithMultiLevel()
        .WithLevel0Topo(Level0Shape::MESH_1D)
        .WithDataSizeLessThan(threshold)
        .WithRankSizeLessOrEqual(64),
        "Rule_MultiLevel_SmallData", "Algo3");
    
    chain.AddRule(ConditionBuilder()
        .WithMultiLevel()
        .WithLevel0Topo(Level0Shape::MESH_1D),
        "Rule_MultiLevel_Mesh1D_Default", "Algo4");
    
    chain.AddRule(ConditionBuilder()
        .WithMultiLevel()
        .WithLevel0Topo(Level0Shape::CLOS),
        "Rule_MultiLevel_CLOS", "Algo5");
}
```

## 5. 优势总结

| 方面 | 旧方案 | 新方案 |
|------|--------|--------|
| **添加新分支** | 需要修改多处 if-else | 添加一行规则 |
| **可读性** | 嵌套深，难以理解 | 声明式，一目了然 |
| **维护性** | 修改容易引入 bug | 规则独立，互不影响 |
| **测试性** | 难以单独测试条件 | 可单独测试每个规则 |
| **扩展性** | 需要理解整体逻辑 | 只需了解规则 API |

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| `src/ops/op_common/selector/algo_rule.h` | 规则基础设施 |
| `src/ops/op_common/selector/rule_based_selector.h` | 规则选择器基类 |
| `src/ops/all_gather/selector/all_gather_auto_selector_v2.h` | 重构示例 |

## 7. 迁移指南

1. 将选择器类改为继承 `RuleBasedSelector`
2. 删除原有的 `SelectXxxAlgo()` 方法
3. 实现对应的 `BuildXxxRules()` 方法
4. 将原有的 if-else 逻辑转换为规则链
5. 使用 `REGISTER_SELECTOR_BY_OPTYPE` 注册新选择器
