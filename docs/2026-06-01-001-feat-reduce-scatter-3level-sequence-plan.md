---
title: Reduce Scatter 3 级 Sequence Executor (Mesh + NHR + NHR)
type: feat
status: active
date: 2026-06-01
origin: files/three_sequence/reduce-scatter-3-level-sequence-requirements.md
deepened: 2026-06-01
---

# Reduce Scatter 3 级 Sequence Executor (Mesh + NHR + NHR)

## Summary

为 reduce_scatter 新增 3 级线性串行 sequence executor（Mesh→NHR→NHR），在 3 层拓扑场景下逐级规约数据量，避免合并后两层为大 NHR 时每步受限于跨 pod 4 端口瓶颈。实现包括：TopoMatchMultilevel 扩展为 3 层、新注册宏 `REGISTER_EXECUTOR_BY_THREE_TEMPS`、新 executor 类、Selector 分支。

---

## Problem Frame

当前 2 级 sequence executor（Mesh+NHR）在 3 层拓扑（8P框→64P pod→跨pod）下合并 level1+level2 为一个大 NHR，每步通信受限于跨 pod 仅 4 端口收敛，框内 7+8 端口和 pod 内 8 端口的带宽被浪费。3 级 executor 逐级规约使跨 pod 只传输已规约的子集数据。（详见 origin: files/three_sequence/reduce-scatter-3-level-sequence-requirements.md）

---

## Requirements

- R1. 新建 `InsV2ReduceScatterSequenceExecutorAicpu3Level` 类，3 个算法模板参数 `<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>`，`SEQUENCE_EXECUTOR_LEVEL_NUM = 3`
- R2. 数据流为线性串行 3 步：INPUT → Mesh(level0) → CCL → NHR(level1) → CCL → NHR(level2) → OUTPUT
- R3. 支持 loop 分片循环处理大数据量
- R4. 新增 `REGISTER_EXECUTOR_BY_THREE_TEMPS` 宏，注册名 `InsReduceScatterSequenceMesh1DNHRNHR`
- R5. CalcRes 为 3 级分别计算 AlgResourceRequest，合并 slaveThreadNum 为 max(res0,res1,res2)，notifyNumPerThread 为各级 max，channels 拆为 3 层
- R6. scratchMultiple = multiplier0 * multiplier1 * multiplier2
- R7. GenIntraTemplateParams (level0 Mesh)：repeatNum = rankSizeLevel1 * rankSizeLevel2
- R8. GenInterTemplateParams (level1 NHR)：repeatNum = rankSizeLevel2
- R9. GenInterTemplateParams (level2 NHR)：repeatNum = 1
- R10. 扩展 TopoMatchMultilevel 支持 3 层 algHierarchyInfo
- R11. Selector 新增分支：3 层拓扑 + 大数据量选择 `InsReduceScatterSequenceMesh1DNHRNHR`

**Origin actors:** A1 (Selector), A2 (TopoMatchMultilevel), A3 (SequenceExecutor3Level), A4 (AlgTemplate)
**Origin flows:** F1 (3 级 Reduce Scatter 执行流 — R1,R2,R3,R4)
**Origin acceptance examples:** AE1 (8P×8Pod×2Cluster 128卡验证 — R1,R2,R7,R8,R9), AE2 (3 级资源合并验证 — R5), AE3 (Selector 分支验证 — R11)

---

## Scope Boundaries

- 不实现流水线（pipeline）模式——作为后续优化方向
- 不修改现有 2 级 sequence executor 的行为——3 级是纯新增
- 不改变 NHR 或 Mesh 算法模板的内部实现——只新增编排层
- 不支持 4 级及以上拓扑

### Deferred to Follow-Up Work

- Pipeline 式 3 级 executor：后续规划中考虑，当前不纳入

---

## Context & Research

### Relevant Code and Patterns

- **现有 2 级 executor（aicpu版）**: `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu.cc` — 数据流 INPUT→Mesh→CCL→NHR→OUTPUT，**整段 CCL 复用**（不分区），3 级应沿用此模式
- **现有 2 级 executor（非aicpu版）**: `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor.cc` — 使用 `ccl_in + ccl_out` 双段分区模式，这是因为 DPU 模式需要 npu2DpuShmemPtr/dpu2NpuShmemPtr 共享内存机制，3 级 aicpu executor 不涉及 DPU，不需要此模式
- **注册宏**: `src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.h` — `REGISTER_EXECUTOR_BY_TWO_TEMPS` 使用 `__COUNTER__` + `DefaultExecCreatorV2` 模式
- **TopoMatchMultilevel**: `src/ops/op_common/topo/topo_match_multilevel.cc` — 当前 `infos.resize(COMM_LAYER_SIZE_2)` 拒绝 >2 层拓扑，`COMM_LAYER_SIZE_3=3` 常量已定义但未使用
- **Selector**: `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc` — `SelectAicpuAlgo` 在 `topoLevelNums>1` 且大数据量时选择 `InsReduceScatterSequenceMesh1DNhr`
- **算法模板**: `InsTempReduceScatterMesh1DZAxisDetour` (level0), `InsTempReduceScatterNHR` (level1/level2)
- **旧 A3 拓扑**: `CalcGeneralTopoInfoForA3` 在 `src/ops/op_common/topo/topo.cc` 已支持 3 层 `AlgHierarchyInfo`（flat localRank/localRankSize 模式），可作为 `TopoForLayer2` 的参考

### Institutional Learnings

- CCL Buffer 整段复用模式已在 2 级 aicpu executor 中验证可行——NHR 算法是原地累加，每个 rank 只修改自己的 slice 区域，不会覆盖其他 rank 的数据，因此 3 级 aicpu 同样可以使用整段 CCL 而无需分区
- 非 aicpu executor 的双段分区是因为 DPU 模式需要共享内存机制，与 3 级 aicpu executor 无关
- `AlgHierarchyInfoForAllLevel.infos` 是 3D vector `[level][instance_group][rank_list]`，Mesh1D 对应 1 个 inner vector，Mesh2D 对应 2 个
- `AlgResourceRequest.channels` 外层维度对应层级数，3 级需要 3 个元素

---

## Key Technical Decisions

- **注册宏**: 新建 `REGISTER_EXECUTOR_BY_THREE_TEMPS`，与现有 `BY_TWO_TEMPS`/`BY_FOUR_TEMPS` 模式一致，命名清晰（而非复用 variadic `REGISTER_EXEC_V2_MULTI`）
- **CCL Buffer 管理**: 采用整段复用模式（与 2 级 aicpu executor 一致）——NHR 是原地累加算法，每个 rank 只修改自己的 slice，3 步串行中 Mesh 结果被 NHR1 结果原地替代，NHR1 结果被 NHR2 结果原地替代，无需分区隔离。非 aicpu 的双段分区是因为 DPU 共享内存需求，与本 executor 无关
- **TopoMatch 扩展方式**: 扩展 `TopoMatchMultilevel`（新增 `TopoForLayer2`），而非新建类
- **rankIdx/rankSize 计算**: 3 级需要 `rankIdxLevel0`, `rankIdxLevel1`, `rankIdxLevel2` 和对应的 `rankSizeLevel0/1/2`

---

## Open Questions

### Resolved During Planning

- `REGISTER_EXECUTOR_BY_THREE_TEMPS` 宏不存在 → 新建专用宏，与 `BY_TWO_TEMPS` 模式一致
- TopoMatchMultilevel 如何支持 3 层 → 扩展为接受 `topoLevelNums==3`，新增 `TopoForLayer2` 方法，参照 `TopoForLayer1` 和旧 `CalcGeneralTopoInfoForA3`

### Deferred to Implementation

- CCL Buffer 整段复用，无需分区——NHR 原地累加，每步只修改本 rank 的 slice 区域，串行执行天然避免冲突
- GenIntraTemplateParams / GenInterTemplateParams1 / GenInterTemplateParams2 中 repeatNum 和 stride 的精确偏移 → 需根据实际拓扑规模和 dataTypeSize 计算
- TopoForLayer2 的 rank 筛选逻辑（level2 的子通信域划分）→ 需根据实际 3 层拓扑数据验证
- Level2Nhr 标志是否需要新增 → 当前不需要：3 层拓扑统一走 3 级 sequence 不区分数据量，Level1Nhr 退化场景已由 BRANCH 2 处理

---

## Output Structure

    src/ops/reduce_scatter/executor/
        ins_v2_reduce_scatter_sequence_executor_aicpu_3level.h   (新增)
        ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc  (新增)
        CMakeLists.txt                                           (修改: 新增 3level.cc 到 src_list)
    src/ops/op_common/executor/registry/
        coll_alg_v2_exec_registry.h                              (修改: 新增宏)
    src/ops/op_common/topo/
        topo_match_multilevel.h                                  (修改: 新增 TopoForLayer2)
        topo_match_multilevel.cc                                 (修改: 实现 TopoForLayer2, 支持3层)
    src/ops/reduce_scatter/selector/
        reduce_scatter_auto_selector.cc                          (修改: 新增分支)

---

## Change List

> 基于需求文档与 Plan 文档的对比分析，梳理所有代码级别的变化点。

### U1. 注册宏变更

| # | 文件 | 变化类型 | 变化点 | 需求映射 |
|---|------|---------|--------|----------|
| U1-1 | `src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.h` | 修改 | 新增 `REGISTER_EXECUTOR_BY_THREE_TEMPS` 宏定义，接受 7 个参数（`type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2`），参照 `REGISTER_EXECUTOR_BY_TWO_TEMPS` 的 `__COUNTER__` + `_HELPER` + `_HELPER_1` 三层嵌套模式 | R4 |

### U2. TopoMatchMultilevel 扩展

| # | 文件 | 变化类型 | 变化点 | 需求映射 |
|---|------|---------|--------|----------|
| U2-1 | `src/ops/op_common/topo/topo_match_multilevel.h` | 修改 | 新增 `TopoForLayer2` 方法声明 | R10 |
| U2-2 | `src/ops/op_common/topo/topo_match_multilevel.cc` | 修改 | `MatchTopo` 拒绝条件从 `topoLevelNums > COMM_LAYER_SIZE_2` 改为 `topoLevelNums > COMM_LAYER_SIZE_3` | R10 |
| U2-3 | `src/ops/op_common/topo/topo_match_multilevel.cc` | 修改 | `infos.resize` 从固定 `COMM_LAYER_SIZE_2` 改为按 `topoLevelNums` 动态选择：`topoLevelNums >= 3` 时 `infos.resize(COMM_LAYER_SIZE_3)`，`topoLevelNums == 2` 时 `infos.resize(COMM_LAYER_SIZE_2)`（保持不变） | R10 |
| U2-4 | `src/ops/op_common/topo/topo_match_multilevel.cc` | 修改 | 新增 `TopoForLayer2` 实现：从网络层 layer 2 提取同 index peer ranks（跨 pod），筛选条件为同一 level0 position + 同一 level1 position + 跨 level2 直连链路，参照 `TopoForLayer1` 和旧 `CalcGeneralTopoInfoForA3`（`localRank = serverIdx / serverNumPerSuperPod, localRankSize = superPodNum`） | R10 |

### U3. Executor 类创建

| # | 文件 | 变化类型 | 变化点 | 需求映射 |
|---|------|---------|--------|----------|
| U3-1 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.h` | 新增 | 新建类声明 `InsV2ReduceScatterSequenceExecutorAicpu3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>`，继承 `InsCollAlgBase`，`SEQUENCE_EXECUTOR_LEVEL_NUM = 3` | R1 |
| U3-2 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `CalcAlgHierarchyInfo` 覆写：调用 `TopoMatchMultilevel::MatchTopo` 生成 3 层 `algHierarchyInfo.infos.size() == 3` | R1, R10 |
| U3-3 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `CalcRes` 实现：3 级模板分别调用 CalcRes，合并 `slaveThreadNum = max(res0, res1, res2)`，`notifyNumPerThread` 各级取 max，`channels[0/1/2]` 对应各级独立 channel 映射 | R5 |
| U3-4 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `CalcScratchMultiple` 实现：`templateScratchMultiplier = multiplier0 * multiplier1 * multiplier2` | R6 |
| U3-5 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `Orchestrate` 实现：计算 `rankIdxLevel0 = myRank % rankSizeLevel0`、`rankIdxLevel1 = (myRank / rankSizeLevel0) % rankSizeLevel1`、`rankIdxLevel2 = myRank / (rankSizeLevel0 * rankSizeLevel1)`，以及对应的 `rankSizeLevel0/1/2` | R1 |
| U3-6 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `OrchestrateLoop` 实现：整段 CCL 复用，3 步串行执行，`maxCountPerLoop` 基于 `cclMem.size / templateScratchMultiplier` 计算 | R2, R3 |
| U3-7 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `GenIntraTemplateParams` (level0)：`buffInfo.inBuffType=INPUT, outBuffType=HCCL_BUFFER, hcclBuffType=HCCL_BUFFER, repeatNum = rankSizeLevel1 * rankSizeLevel2, inputSliceStride = dataSize_, outputSliceStride = currDataCount * dataTypeSize_, inputRepeatStride = rankSizeLevel0 * dataSize_, outputRepeatStride = rankSizeLevel0 * currDataCount * dataTypeSize_` | R7 |
| U3-8 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `GenInterTemplateParams1` (level1)：`buffInfo.inBuffType=HCCL_BUFFER, outBuffType=HCCL_BUFFER, hcclBuffType=HCCL_BUFFER, repeatNum = rankSizeLevel2, inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_, inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_, outputSliceStride = 0` | R8 |
| U3-9 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 新增 | `GenInterTemplateParams2` (level2)：`buffInfo.inBuffType=HCCL_BUFFER, outBuffType=OUTPUT, hcclBuffType=HCCL_BUFFER, repeatNum = 1, inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_, inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_, outputSliceStride = 0` | R9 |
| U3-10 | `src/ops/reduce_scatter/executor/CMakeLists.txt` | 修改 | 在 `src_list` 中追加 `${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | R1 |

### U4. 注册 + Selector 更新

| # | 文件 | 变化类型 | 变化点 | 需求映射 |
|---|------|---------|--------|----------|
| U4-1 | `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` | 修改 | 文件末尾新增注册调用：`REGISTER_EXECUTOR_BY_THREE_TEMPS(HCCL_CMD_REDUCE_SCATTER, InsReduceScatterSequenceMesh1DNHRNHR, InsV2ReduceScatterSequenceExecutorAicpu3Level, TopoMatchMultilevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR, InsTempReduceScatterNHR)` | R4 |
| U4-2 | `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc` | 修改 | BRANCH 4（line 288-290 `localNetInsSizeOfLayer[0] > 1 && level0Topo == MESH_1D`）入口处新增 `if (topoInfo->topoLevelNums >= 3)` 分支，返回 `"InsReduceScatterSequenceMesh1DNHRNHR"`，3 层拓扑统一走 3 级不区分数据量，其余所有逻辑不变 | R11 |

### 汇总

| 类型 | 数量 | 文件 |
|------|------|------|
| 新增文件 | 2 | `ins_v2_reduce_scatter_sequence_executor_aicpu_3level.h`, `ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` |
| 修改文件 | 5 | `coll_alg_v2_exec_registry.h`, `topo_match_multilevel.h`, `topo_match_multilevel.cc`, `reduce_scatter_auto_selector.cc`, `reduce_scatter/executor/CMakeLists.txt` |
| 涉及模块 | 3 | 注册宏、拓扑匹配、reduce_scatter executor/selector |

---

## Gap Analysis

> 针对 Plan 文档中的变化清单，分析当前方案是否完善，是否存在遗漏。

### G1. 构建系统变更缺失 [严重度: 高] → 已解决

**问题:** Output Structure 和 Change List 未包含构建系统文件变更。新增的 `.h` 和 `.cc` 文件需要加入 CMakeLists.txt / SConscript 等构建配置中，否则无法编译链接。

**解决:** 在 Output Structure 中补充 `src/ops/reduce_scatter/executor/CMakeLists.txt`（修改: 新增 3level.cc 到 src_list）；在 U3 Files 中补充该文件；在 Change List 中新增 U3-10 条目。

**建议:** 在 U3 的 Files 列表中补充：
- Modify: `src/ops/reduce_scatter/executor/CMakeLists.txt`（或对应构建文件）— 新增 `ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc` 到编译目标

---

### G2. CalcAlgHierarchyInfo 方法未显式描述 [严重度: 中] → 已解决

**问题:** Plan 在 U3 Approach 中列出了 CalcRes、Orchestrate、OrchestrateLoop、GenXxxTemplateParams、CalcScratchMultiple 等方法，但**缺少 CalcAlgHierarchyInfo 的实现说明**。

**解决:** 在 U3 Approach 中补充 CalcAlgHierarchyInfo 覆写描述：调用 `AlgTopoMatch::MatchTopo(comm, topoInfo, algHierarchyInfo)` 生成 3 层 algHierarchyInfo，验证 `infos.size() == 3`，与 2 级 aicpu executor 的 CalcAlgHierarchyInfo 实现模式一致。

---

### G3. currDataCount 在各级间的演变未详细说明 [严重度: 中] → 已解决

**问题:** Plan 在 OrchestrateLoop pseudo-flow 中对所有 3 步使用同一 `currDataCount`，但在 reduce_scatter 场景中，每级规约后每 rank 拥有的数据量递减。

**解决:** 在 U3 Approach 中补充 currDataCount 语义定义：currDataCount 表示每轮 loop 中每 rank 最终输出的数据量（`dataCount_ = dataSize_ / userRankSize_` 的一个分片），所有 3 步共用同一 currDataCount，各级间每 rank 数据量递减靠 repeatNum 体现而非 currDataCount 变化。同时补充了各级的输入数据量说明。

---

### G4. GenInterTemplateParams1/2 的 stride 和 inputSliceStride 缺失 [严重度: 中] → 已解决

**问题:** Plan 为 GenIntraTemplateParams (level0) 指定了 `inputSliceStride = dataSize_`，但 GenInterTemplateParams1 (level1) 和 GenInterTemplateParams2 (level2) **缺少 stride 和 inputSliceStride 参数说明**。

**解决:** 在 U3 Approach 中补充完整参数：
- GenIntraTemplateParams (level0): `inputSliceStride = dataSize_, outputSliceStride = currDataCount * dataTypeSize_, inputRepeatStride = rankSizeLevel0 * dataSize_, outputRepeatStride = rankSizeLevel0 * currDataCount * dataTypeSize_`
- GenInterTemplateParams1 (level1): `inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_, outputSliceStride = 0`
- GenInterTemplateParams2 (level2): `inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_, outputSliceStride = 0`
同时在 Change List U3-7/8/9 条目中补充 stride 参数。

---

### G5. topoLevelNums 在 Selector 中的可访问性未确认 [严重度: 中] → 已解决

**问题:** Plan 在 U4-2 中新增 `if (topoLevelNums >= 3)` 分支，但未确认 `topoLevelNums` 变量在 `SelectAicpuAlgo` 方法中是否已可用。

**解决:** 代码调研确认 `topoInfo->topoLevelNums` 在 Selector 中已广泛使用（line 33, 35, 136, 276, 400, 402），无需修改 `.h` 文件。修正 U4-2 变化点中的变量名为 `topoInfo->topoLevelNums >= 3`，移除 U4-3（`.h` 修改已确认不需要）。

---

### G6. 各级 AlgorithmTemplate 的资源分配未细化 [严重度: 低] → 已解决

**问题:** Plan 在 CalcRes 中说明了合并策略，但未说明 3 级模板各自的 `templateResource0/1/2` 如何从合并后的总资源中分配。

**解决:** 在 U3 Approach CalcRes 描述中补充：3 级模板共享 `slaveThreadNum`（取 max 后足够任一级使用），`notify` 按各级 max 合并后由 Orchestrator 在各级 KernelRun 前通过 `GenTempResource(resCtx, level, algTemplate, templateResource)` 分配，`channels[0/1/2]` 直接映射到各级模板的 channel 需求。

---

### G7. 单元测试文件未列入 Output Structure [严重度: 低] → 待实现时确定

**问题:** 每个 Implementation Unit 都描述了 test scenarios，但 Output Structure 和 Change List 中未列出任何测试文件。

**说明:** HCCL 项目的单测文件路径和命名惯例需根据实际测试框架确认（st 测试目录在 `test/st/algorithm/`）。待实现时根据项目惯例补充，当前在 Output Structure 中暂不列出具体测试文件路径。

---

### G8. Executor 类头文件 include 依赖未指定 [严重度: 低] → 已解决

**问题:** 新增的 `.h` 文件需要正确的 `#include` 依赖，Plan 未列出。

**解决:** 在 U3 Approach 中补充关键 include 依赖说明：参照 `ins_v2_reduce_scatter_sequence_executor_aicpu.h`，需包含 `InsCollAlgBase`、`InsTempReduceScatterMesh1DZAxisDetour`、`InsTempReduceScatterNHR`、`TopoMatchMultilevel`、`TemplateDataParams`、`AlgResourceRequest` 等对应头文件。

---

### Gap Analysis 总结

| 严重度 | 数量 | 状态 | 关键遗漏 |
|--------|------|------|----------|
| 高 | 1 | 已解决 | G1: 构建系统变更缺失 |
| 中 | 4 | 已解决 | G2: CalcAlgHierarchyInfo、G3: currDataCount 语义、G4: stride 参数、G5: topoLevelNums 可访问性 |
| 低 | 3 | 2已解决,1待定 | G6: 资源分配细化(已解决)、G7: 测试文件(待实现时确定)、G8: include 依赖(已解决) |

**全部遗漏项已解决或已明确处理方式，仅 G7（单测文件路径）待实现时确定。**

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification.*

### 3 级 OrchestrateLoop 数据流

```
for each loop chunk:
  ┌─ Step0 (level0 Mesh) ──────────────────────────┐
  │ INPUT ──[Mesh1DZAxisDetour]──→ CCL             │
  │ repeatNum = rankSizeLevel1 × rankSizeLevel2     │
  │ inputSliceStride = dataSize_                    │
  │ outBuffType = HCCL_BUFFER                       │
  └────────────────────────────────────────────────┘
            ↓  (CCL 中本 rank 的 slice → NHR1 输入)
  ┌─ Step1 (level1 NHR) ──────────────────────────┐
  │ CCL ──[NHR]──→ CCL (原地累加)                   │
  │ repeatNum = rankSizeLevel2                      │
  │ inBuffType = HCCL_BUFFER, outBuffType =         │
  │             HCCL_BUFFER                         │
  │ inBuffBaseOff = rankIdxLevel0 × currDataCount × │
  │                dataTypeSize_                    │
  │ NHR 原地替代 Mesh 结果，不覆盖其他 rank 的 slice │
  └────────────────────────────────────────────────┘
            ↓  (CCL 中本 rank 的 slice → NHR2 输入)
  ┌─ Step2 (level2 NHR) ──────────────────────────┐
  │ CCL ──[NHR]──→ OUTPUT                          │
  │ repeatNum = 1                                   │
  │ inBuffType = HCCL_BUFFER, outBuffType = OUTPUT  │
  │ inBuffBaseOff = rankIdxLevel0 × currDataCount × │
  │                dataTypeSize_                    │
  │ NHR 原地累加后写 OUTPUT                         │
  └────────────────────────────────────────────────┘
```

### CCL Buffer 管理

```
整段 CCL 复用，不分区：

Step0: hcclBuff = cclMem(整段), inputPtr = INPUT, outputPtr = cclMem.addr
       outBuffType = HCCL_BUFFER → Mesh 规约结果写入 CCL
Step1: hcclBuff = cclMem(整段), inputPtr = cclMem.addr, outputPtr = cclMem.addr
       inBuffType = HCCL_BUFFER, outBuffType = HCCL_BUFFER → NHR 原地累加回 CCL
Step2: hcclBuff = cclMem(整段), inputPtr = cclMem.addr, outputPtr = outputPtr
       inBuffType = HCCL_BUFFER, outBuffType = OUTPUT → NHR 结果写 OUTPUT

maxCountPerLoop = cclMem.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
                  * HCCL_MIN_SLICE_ALIGN / dataTypeSize_
```

NHR 原地累加原理：每个 rank 只修改自己的 slice 区域（offset = rankIdxLevel0 × sliceSize），不同 rank 的 slice 按 stride 间隔排列互不重叠。Step1 的 NHR 结果替代 Mesh 结果（Mesh 数据已被消耗），Step2 的 NHR 结果替代 Step1 结果（Step1 数据已被消耗），串行执行天然保证安全。

---

## Implementation Units

### U1. 新增 REGISTER_EXECUTOR_BY_THREE_TEMPS 注册宏

**Goal:** 在注册宏头文件中新增支持 3 个算法模板参数的注册宏，与现有 BY_TWO_TEMPS 模式一致。

**Requirements:** R4

**Dependencies:** None

**Files:**
- Modify: `src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.h`

**Approach:**
- 参照 `REGISTER_EXECUTOR_BY_TWO_TEMPS` 的 `__COUNTER__` + `_HELPER` + `_HELPER_1` 三层嵌套模式
- 新宏接受 7 个参数：`type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2`
- `DefaultExecCreatorV2` 已是通用工厂，无需修改

**Patterns to follow:**
- `REGISTER_EXECUTOR_BY_TWO_TEMPS` 在同一文件中的定义模式
- `REGISTER_EXECUTOR_BY_FOUR_TEMPS` 也参照此模式（5 个模板参数）

**Test scenarios:**
- Happy path: 使用新宏注册一个测试 executor 类，验证 `CollAlgExecRegistryV2::Instance().GetAlgExec()` 能正确找到并创建实例
- Edge case: 同一 name 多次注册（`__COUNTER__` 应保证唯一变量名）

**Verification:**
- 新宏定义存在且编译通过
- 注册的 executor 名可通过 `GetAlgExec(HCCL_CMD_REDUCE_SCATTER, "InsReduceScatterSequenceMesh1DNHRNHR")` 查询到

---

### U2. 扩展 TopoMatchMultilevel 支持 3 层拓扑

**Goal:** 使 `TopoMatchMultilevel::MatchTopo` 在 `topoLevelNums == 3` 时生成 `infos.size() == 3` 的 algHierarchyInfo，新增 `TopoForLayer2` 方法。

**Requirements:** R10

**Dependencies:** None（executor 依赖此变更，但可独立开发验证）

**Files:**
- Modify: `src/ops/op_common/topo/topo_match_multilevel.h`
- Modify: `src/ops/op_common/topo/topo_match_multilevel.cc`

**Approach:**
- 修改 `MatchTopo` 验证逻辑：将 `topoLevelNums > COMM_LAYER_SIZE_2` 的拒绝条件改为 `topoLevelNums > COMM_LAYER_SIZE_3`
- 在 `topoLevelNums >= 3` 时 `infos.resize(COMM_LAYER_SIZE_3)` 而非固定 `COMM_LAYER_SIZE_2`
- 新增 `TopoForLayer2` 方法：参照 `TopoForLayer1` 的实现模式，从网络层 layer 2 提取同 index peer ranks（跨 pod 层面）
- `TopoForLayer2` 筛选条件：rank 在同一 level0 position 且同一 level1 position，跨 level2 实例间有直连链路
- 参照旧 `CalcGeneralTopoInfoForA3` 中 level2 的 `localRank = serverIdx / serverNumPerSuperPod` 和 `localRankSize = superPodNum` 计算方式

**Patterns to follow:**
- `TopoForLayer1` 方法（筛选 same-index peers with direct links）
- `CalcGeneralTopoInfoForA3`（旧 3 层拓扑计算逻辑）
- `COMM_LAYER_SIZE_3` 常量（已在 `topo_match_base.h` 定义）

**Test scenarios:**
- Happy path: 8P×8Pod×2Cluster（128卡）3 层拓扑，验证 `infos.size() == 3`，`infos[0]` 包含 8 个框内 ranks，`infos[1]` 包含同 index 的 8 个 pod 内 peers，`infos[2]` 包含同 index 的 2 个跨 pod peers
- Edge case: 不对称 pod 规模（GCD 场景），验证 level2 的 rank 划分正确
- Error path: `topoLevelNums == 4` 时仍被拒绝（`COMM_LAYER_SIZE_3` 上限）
- Integration: 2 层拓扑场景仍正常工作（`infos.size() == 2`）

**Verification:**
- `topoLevelNums == 3` 时 `infos.size() == 3`，每层包含正确的 rank ID 列表
- 2 层拓扑不受影响

---

### U3. 创建 InsV2ReduceScatterSequenceExecutorAicpu3Level 类

**Goal:** 新建 3 级 sequence executor 类，实现 3 步串行数据流编排、3 级资源计算、3 级模板参数生成。

**Requirements:** R1, R2, R3, R5, R6, R7, R8, R9

**Dependencies:** U1（注册宏），U2（TopoMatchMultilevel 3 层支持）

**Files:**
- Create: `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.h`
- Create: `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc`
- Modify: `src/ops/reduce_scatter/executor/CMakeLists.txt`（在 `src_list` 中追加 `${CMAKE_CURRENT_SOURCE_DIR}/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc`，参照现有 2 级 aicpu executor 的添加位置，置于 `ins_v2_reduce_scatter_sequence_executor_aicpu.cc` 之后）

**Approach:**
- 类模板参数 `<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>`，继承 `InsCollAlgBase`
- `SEQUENCE_EXECUTOR_LEVEL_NUM = 3`
- `CalcAlgHierarchyInfo`: 覆写该方法，调用 `AlgTopoMatch::MatchTopo(comm, topoInfo, algHierarchyInfo)` 生成 3 层 algHierarchyInfo，验证 `algHierarchyInfo.infos.size() == 3`（与 2 级 aicpu executor 的 CalcAlgHierarchyInfo 实现模式一致：`myRank_ = topoInfo->userRank; rankSize_ = topoInfo->userRankSize; AlgTopoMatch topoMatch; CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));`）
- `CalcRes`: 3 级模板分别调用 `CalcRes`，合并 `slaveThreadNum = max(res0,res1,res2)`，`notifyNumPerThread` 各级取 max，`channels[0/1/2]` 对应各级。3 级模板共享 `slaveThreadNum`（取 max 后足够任一级使用），`notify` 按各级 max 合并后由 Orchestrator 在各级 KernelRun 前通过 `GenTempResource(resCtx, level, algTemplate, templateResource)` 分配，`channels[0/1/2]` 直接映射到各级模板的 channel 需求（与 2 级 aicpu executor 的 CalcRes 合并模式一致）
- `Orchestrate`: 计算 `rankIdxLevel0/1/2` 和 `rankSizeLevel0/1/2`，3 级索引为 `myRank % rankSizeLevel0`、`(myRank / rankSizeLevel0) % rankSizeLevel1`、`myRank / (rankSizeLevel0 * rankSizeLevel1)`
- `OrchestrateLoop`: 整段 CCL 复用（与 2 级 aicpu 一致），3 步串行
- `CalcScratchMultiple`: `templateScratchMultiplier = multiplier0 * multiplier1 * multiplier2`
- `maxCountPerLoop`: 基于 `cclMem.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_` 计算（整段 CCL，不分区，与 2 级 aicpu 一致）
- **currDataCount 语义**: `currDataCount` 表示每轮 loop 中每 rank 最终输出的数据量（即 `dataCount_` 的一个分片，`dataCount_ = dataSize_ / userRankSize_）。所有 3 步共用同一 `currDataCount`，与 2 级 aicpu executor 的语义一致。各级间每 rank 数据量递减是靠 repeatNum 体现而非 currDataCount 变化：Step0 Mesh 输入每 rank 为 `currDataCount × rankSizeLevel0 × rankSizeLevel1 × rankSizeLevel2` 的总输入数据分片（通过 repeatNum=rankSizeLevel1×rankSizeLevel2 拆分为多组 Mesh 操作），Step1 NHR 输入每 rank 为 `currDataCount × rankSizeLevel0` 的 CCL 数据（通过 repeatNum=rankSizeLevel2 拆分为多组 NHR 操作），Step2 NHR 输入每 rank 为 `currDataCount` 的 CCL 数据（repeatNum=1）
- `GenIntraTemplateParams` (level0): `buffInfo.inBuffType=INPUT, outBuffType=HCCL_BUFFER, hcclBuffType=HCCL_BUFFER, repeatNum = rankSizeLevel1 * rankSizeLevel2, inputSliceStride = dataSize_, outputSliceStride = currDataCount * dataTypeSize_, inputRepeatStride = rankSizeLevel0 * dataSize_, outputRepeatStride = rankSizeLevel0 * currDataCount * dataTypeSize_`（参照 2 级 aicpu 的 GenIntraTemplateParams，2 级的 repeatNum=rankSizeLevel1，3 级扩展为 rankSizeLevel1×rankSizeLevel2）
- `GenInterTemplateParams1` (level1): `buffInfo.inBuffType=HCCL_BUFFER, outBuffType=HCCL_BUFFER, hcclBuffType=HCCL_BUFFER, repeatNum = rankSizeLevel2, inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_, hcclBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_, inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_, outputSliceStride = 0`（参照 2 级 aicpu 的 GenInterTemplateParams，2 级的 inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_，3 级的 repeatNum 从 1 扩展为 rankSizeLevel2，需要新增 inputRepeatStride/outputRepeatStride）
- `GenInterTemplateParams2` (level2): `buffInfo.inBuffType=HCCL_BUFFER, outBuffType=OUTPUT, hcclBuffType=HCCL_BUFFER, repeatNum = 1, inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_, hcclBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_, inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_, outputSliceStride = 0, inputRepeatStride = 0, outputRepeatStride = 0`（与 2 级 aicpu 的 GenInterTemplateParams 结构一致，repeatNum=1 无需 repeatStride）
- **关键 include 依赖**（新 .h 文件需包含）: 参照 `ins_v2_reduce_scatter_sequence_executor_aicpu.h`，需包含 `InsCollAlgBase`、`InsTempReduceScatterMesh1DZAxisDetour`、`InsTempReduceScatterNHR`、`TopoMatchMultilevel`、`TemplateDataParams`、`AlgResourceRequest` 等对应头文件

**Technical design:**

> *Directional guidance, not implementation specification.*

```
OrchestrateLoop pseudo-flow:
  // 整段 CCL 复用，与 2 级 aicpu 一致
  algTemplateLevel0 = InsAlgTemplate0(param, myRank_, infos[0])
  algTemplateLevel1 = InsAlgTemplate1(param, myRank_, infos[1])
  algTemplateLevel2 = InsAlgTemplate2(param, myRank_, infos[2])

  scratchMultiplier = CalcScratchMultiple(INPUT, HCCL_BUFFER)
                    * CalcScratchMultiple(HCCL_BUFFER, HCCL_BUFFER)
                    * CalcScratchMultiple(HCCL_BUFFER, OUTPUT)

  maxCountPerLoop = cclMem.size / scratchMultiplier / ... / dataTypeSize_

  for each loop chunk:
    // Step0: Mesh 规约 → CCL
    GenIntraTemplateParams(level0, processedDataCount, currDataCount)
    algTemplateLevel0.KernelRun(param, tempAlgParamsLevel0, templateResource0)

    // Step1: NHR 原地累加 → CCL (替代 Mesh 结果)
    GenInterTemplateParams1(level1, processedDataCount, currDataCount)
    algTemplateLevel1.KernelRun(param, tempAlgParamsLevel1, templateResource1)

    // Step2: NHR 原地累加 → OUTPUT (替代 Step1 结果)
    GenInterTemplateParams2(level2, processedDataCount, currDataCount)
    algTemplateLevel2.KernelRun(param, tempAlgParamsLevel2, templateResource2)

    processedDataCount += currDataCount
```

**Patterns to follow:**
- `InsV2ReduceScatterSequenceExecutorAicpu`（现有 2 级 aicpu 版——框架结构、CalcRes 模式、整段 CCL 复用模式）

**Test scenarios:**
- Happy path: 8P×8Pod×2Cluster，reduce_scatter fp32 数据量 > 1GB，验证 3 步串行执行后 outputPtr 数据正确 Covers AE1
- Happy path: loop 分片场景，数据量超过 maxCountPerLoop，验证多轮循环后数据正确
- Edge case: 小数据量（不需要 loop 分片），验证单轮执行正确
- Integration: CalcRes 合并 3 级资源，验证 slaveThreadNum = max(3级)，channels 有 3 层 Covers AE2

**Verification:**
- executor 编译通过
- 3 级 KernelRun 依次执行，outputPtr 包含正确规约结果
- 整段 CCL 原地累加正确，每步 NHR 只修改本 rank slice 区域

---

### U4. 注册 executor 并更新 Selector

**Goal:** 使用新宏注册 3 级 executor，在 Selector 中精确设计 3 层拓扑的算法选择分支树。

**Requirements:** R4, R11

**Dependencies:** U1（注册宏），U2（TopoMatchMultilevel 3 层支持），U3（executor 类）

**Files:**
- Modify: `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu_3level.cc`（注册调用）
- Modify: `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc`（line 288-290 else 分支内新增 topoLevelNums >= 3 判断）

**Approach:**

#### 4a. 注册调用

在 executor .cc 文件末尾添加：
```
REGISTER_EXECUTOR_BY_THREE_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
    InsReduceScatterSequenceMesh1DNHRNHR,
    InsV2ReduceScatterSequenceExecutorAicpu3Level,
    TopoMatchMultilevel,
    InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR,
    InsTempReduceScatterNHR)
```

#### 4b. Selector 最小修改：在 BRANCH 4 入口处新增 3 层拓扑判断

在 `localNetInsSizeOfLayer[0] > 1 && level0Topo == MESH_1D` 分支（BRANCH 4）的**入口处**，优先判断 `topoLevelNums >= 3`，3 层拓扑直接走 3 级 algorithm，不区分数据量大小。其余所有逻辑不变。

**原始代码（line 284-294）：**
```
} else if (localNetInsSizeOfLayer[0] > 1 && level0Topo == MESH_1D) {
    if (dataSize > RS_AICPU_1D_MIN_DATA_SIZE) {
        if (Level1Hd) {
            selectAlgName = "InsReduceScatterParallelMesh1DHD";
        } else {
            selectAlgName = (dataSize * userRankSize > THRESHOLD) ?
                "InsReduceScatterSequenceMesh1DNhr" : "InsReduceScatterParallelMesh1DNHR";
        }
    } else {
        selectAlgName = "InsReduceScatterNHR";
    }
}
```

**修改后：在 BRANCH 4 入口处新增一行判断，其余完全不变：**
```
} else if (localNetInsSizeOfLayer[0] > 1 && level0Topo == MESH_1D) {
    if (topoLevelNums >= 3) {
        selectAlgName = "InsReduceScatterSequenceMesh1DNHRNHR";  ← 3 层拓扑统一走 3 级
    } else if (dataSize > RS_AICPU_1D_MIN_DATA_SIZE) {
        if (Level1Hd) {
            selectAlgName = "InsReduceScatterParallelMesh1DHD";
        } else {
            selectAlgName = (dataSize * userRankSize > THRESHOLD) ?
                "InsReduceScatterSequenceMesh1DNhr" : "InsReduceScatterParallelMesh1DNHR";
        }
    } else {
        selectAlgName = "InsReduceScatterNHR";
    }
}
```

**影响范围：**
- 只新增一个 `if (topoLevelNums >= 3)` 分支，放在 BRANCH 4 最前面
- `Level1Hd` 分支不变，`dataSize <= 4MB` 分支不变，2 层拓扑的所有条件不变
- 3 层拓扑时：`topoLevelNums >= 3` 直接匹配，不区分数据量大小、不判断 Level1Hd
- Level1Nhr 退化（BRANCH 2）和 64-bit/PROD（BRANCH 1）不受影响（它们在 BRANCH 4 之前）

**Patterns to follow:**
- 现有 `InsReduceScatterSequenceMesh1DNhr` 的注册方式
- Selector line 288-290 的 else 分支结构（仅在此处新增 `topoLevelNums >= 3` 判断）

**test scenarios:**
- Happy path: `topoLevelNums == 3`, level0Topo==MESH_1D → "InsReduceScatterSequenceMesh1DNHRNHR" Covers AE3
- Level1Nhr: `topoLevelNums == 3`, Level1Nhr=true → "InsReduceScatterNHR" (退化，走 BRANCH 2)
- 64-bit data: `topoLevelNums == 3`, dataType=FP64 → "InsReduceScatterAicpuReduceNHR" (走 BRANCH 1)
- Edge case: `topoLevelNums == 2`, 所有现有逻辑不变（无回归）
- Edge case: `topoLevelNums == 3`, level0Topo==CLOS → "InsReduceScatterNHR" (走 BRANCH 5)
- Integration: 注册名能通过 `CollAlgExecRegistryV2::Instance().GetAlgExec()` 查询到 executor 实例

**Verification:**
- Selector 在 3 层拓扑 + MESH_1D 场景始终返回 `"InsReduceScatterSequenceMesh1DNHRNHR"`
- Selector 在 3 层拓扑 + Level1Nhr 退化场景返回 `"InsReduceScatterNHR"`
- 2 层拓扑所有分支不变（无回归）

---

## System-Wide Impact

- **Interaction graph:** 新 executor 通过 `CollAlgExecRegistryV2` 注册，由 Selector 选中后调用 `CalcAlgHierarchyInfo→CalcRes→Orchestrate` 流程。与现有 2 级 executor 无交互冲突。
- **Error propagation:** 3 级 KernelRun 任何一步失败返回 HCCL_E_INTERNAL，OrchestrateLoop 使用 CHK_RET 传播错误
- **State lifecycle risks:** 整段 CCL 原地累加——Step0 写入 CCL 后 Step1 原地替代（串行保证安全）；Step1 写入 CCL 后 Step2 原地替代并最终输出到 OUTPUT。每个 rank 只修改自己的 slice，无跨 rank 冲突
- **API surface parity:** 无新公共 API。executor 接口与 `InsCollAlgBase` 一致。
- **Integration coverage:** 3 级 executor 的 `CalcAlgHierarchyInfo` 调用 `TopoMatchMultilevel::MatchTopo`，需验证 3 层 topology 数据正确传递到 executor
- **Unchanged invariants:** 现有 2 级 executor 的注册名和行为不变；NHR/Mesh 模板内部逻辑不变

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| TopoMatchMultilevel 扩展可能影响现有 2 层拓扑匹配逻辑 | 2 层场景保持 `infos.resize(2)` 不变，只在 `topoLevelNums>=3` 时 resize 为 3 |
| CCL Buffer 整段复用可能因 scratchMultiplier 更大导致 maxCountPerLoop 减小、loop 次数增加 | 与 2 级 aicpu 一致使用整段 CCL（非半段），每级数据量递减（level2 只处理 1/(rankSizeLevel0×rankSizeLevel1) 的量），带宽收益远大于 loop overhead |
| 3 级串行延迟叠加 | 每级数据量递减，实际延迟影响远小于带宽收益；流水线模式留作后续优化 |
| TopoForLayer2 的 rank 筛选逻辑可能有不对称 pod 场景的边界问题 | 参照 TopoForLayer1 的 GCD 处理模式，对不对称场景做 GCD 计算 |
| Selector 3 层分支最小修改，仅在 2 级 parallel 的 else 分支内新增 `topoLevelNums >= 3` 判断 | 修改范围极小：只改 line 288-290 的 else 分支，不影响 Level1Hd、小数据、Level1Nhr、64-bit/PROD 等任何其他分支 |
| 3 层拓扑统一走 3 级 sequence，小数据量场景 loop overhead 可能不划算 | 3 级 executor loop 分片机制对小数据量仅单轮 loop，开销可接受；后续可根据实测数据调整阈值 |

---

## Documentation / Operational Notes

- 新增算法名 `InsReduceScatterSequenceMesh1DNHRNHR` 需在 HCCL 算法配置文档中记录
- Selector 条件变更需在算法选择策略文档中记录 3 层拓扑场景的选择逻辑

---

## Sources & References

- **Origin document:** [reduce-scatter-3-level-sequence-requirements.md](files/three_sequence/reduce-scatter-3-level-sequence-requirements.md)
- Related code: `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_aicpu.cc` (2 级 aicpu executor — 整段 CCL 复用参考)
- Related code: `src/ops/op_common/topo/topo_match_multilevel.cc` (TopoMatch 扩展目标)
- Related code: `src/ops/op_common/topo/topo.cc` (旧 A3 拓扑参考)
- Related code: `src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.h` (注册宏)