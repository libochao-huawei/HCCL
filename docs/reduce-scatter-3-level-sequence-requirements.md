---
date: 2026-05-30
topic: reduce-scatter-3-level-sequence
---

# Reduce Scatter 3 级 Sequence Executor (Mesh + NHR + NHR)

## Summary

为 reduce_scatter 操作新增 3 级 sequence executor，支持 Mesh(level0) → NHR(level1) → NHR(level2) 的线性串行数据流，使 3 层拓扑场景下跨层级链路带宽得到充分利用，避免合并后两层为大 NHR 时每步受限于最慢链路。

---

## Problem Frame

当前 reduce_scatter 的 sequence executor 仅支持 2 级（Mesh + NHR），在 3 层拓扑（8P 框内 fullmesh → 64P pod 内 → 跨 pod）场景下，将 level1 与 level2 合并为一个大的 NHR 执行。合并后的 NHR 每一步通信都受限于最慢的链路（跨 pod 仅 4 端口收敛），无法充分利用框内和 pod 内的高带宽链路。level0 有 7+8 端口、level1 有 8 端口的带宽资源被浪费。

---

## Actors

- A1. **Selector**: 根据拓扑和数据量选择算法，决定何时使用 3 级 sequence executor
- A2. **TopoMatchMultilevel**: 拓扑匹配层，负责生成 3 层 algHierarchyInfo
- A3. **SequenceExecutor3Level**: 3 级序列执行器，编排 Mesh→NHR→NHR 的数据流
- A4. **AlgTemplate**: 各级算法模板（InsTempReduceScatterMesh1DZAxisDetour / InsTempReduceScatterNHR），执行实际通信

---

## Key Flows

- F1. **3 级 Reduce Scatter 执行流**
  - **Trigger:** Selector 检测到 3 层拓扑 + 大数据量场景，选择 3 级 sequence algorithm
  - **Actors:** A1, A2, A3, A4
  - **Steps:**
    1. TopoMatchMultilevel 扩展为 3 层，生成 `infos.size() == 3` 的 algHierarchyInfo
    2. CalcRes 计算 3 级资源需求（channels[0/1/2]、slaveThreadNum、notify）
    3. OrchestrateLoop 循环分片执行：step0 Mesh(level0) → step1 NHR(level1) → step2 NHR(level2)
    4. 每级完成后数据传递至下级的 CCL Buffer 输入区
  - **Outcome:** 每个 rank 获得 reduce_scatter 后属于自己的数据切片
  - **Covered by:** R1, R2, R3, R4

---

## Requirements

**Executor 框架**

- R1. 新建 `InsV2ReduceScatterSequenceExecutor3Level` 类（通用 3 级编排层，不绑定 aicpu，后续 CCU 等引擎可复用），接受 3 个算法模板参数 `<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>`，`SEQUENCE_EXECUTOR_LEVEL_NUM = 3`
- R2. 数据流为线性串行 3 步：INPUT → Mesh(level0) → CCL Buffer → NHR(level1) → CCL Buffer → NHR(level2) → OUTPUT，每步串行执行、级间数据通过 CCL Buffer 传递
- R3. 支持 loop 分片循环处理大数据量，每轮循环依次执行 3 级算法，与现有 2 级的 loop 机制一致
- R4. 注册宏需要支持 3 个模板参数（新增 `REGISTER_EXECUTOR_BY_THREE_TEMPS` 或等效机制），注册名称为 `InsReduceScatterSequenceMesh1DNHRNHR`

**资源计算**

- R5. CalcRes 为 3 级分别计算 AlgResourceRequest，合并 slaveThreadNum 为 max(res0, res1, res2)，notifyNumPerThread 为各级 max，channels 拆为 3 层 `channels[0/1/2]`
- R6. CCL Buffer scratchMultiple 计算：`templateScratchMultiplier = multiplier0 * multiplier1 * multiplier2`，每轮 maxCountPerLoop 基于此计算

**模板参数生成**

- R7. GenIntraTemplateParams (level0 Mesh)：`repeatNum = rankSizeLevel1 * rankSizeLevel2`，表示框内需要为每个更高层级的框重复执行
- R8. GenInterTemplateParams (level1 NHR)：`repeatNum = rankSizeLevel2`，表示 pod 内需要为每个跨 pod 组重复执行
- R9. GenInterTemplateParams (level2 NHR)：`repeatNum = 1`，跨 pod 不需要重复

**拓扑匹配**

- R10. 扩展 TopoMatchMultilevel 使其支持生成 `algHierarchyInfo.infos.size() == 3`，新增 level2 的子通信域划分逻辑

**算法选择**

- R11. 在 ReduceScatterAutoSelector::SelectAicpuAlgo 中新增分支：3 层拓扑 + 大数据量（超过 `RS_AICPU_SEQUENCE_SIZE_THRESHOLD`）时选择 `InsReduceScatterSequenceMesh1DNHRNHR`

---

## Acceptance Examples

- AE1. **Covers R1, R2, R7, R8, R9.** Given 8P×8Pod×2Cluster（128卡 3 层拓扑）, 当执行 reduce_scatter 且数据量 > 1GB 时, executor 在 loop 循环内依次执行: Mesh(level0, repeatNum=16) → NHR(level1, repeatNum=2) → NHR(level2, repeatNum=1), 最终每个 rank 的 outputPtr 包含正确规约后的数据切片。
- AE2. **Covers R5.** Given 3 级资源请求 resReq0(slaveThreadNum=2), resReq1(slaveThreadNum=3), resReq2(slaveThreadNum=1), 合并后 resourceRequest.slaveThreadNum = 3, channels 包含 3 层各自独立的 channel 映射。
- AE3. **Covers R11.** Given topoLevelNums==3 且 dataSize * userRankSize > RS_AICPU_SEQUENCE_SIZE_THRESHOLD, SelectAicpuAlgo 返回 "InsReduceScatterSequenceMesh1DNHRNHR"。

---

## Success Criteria

- 3 级 sequence executor 在 3 层拓扑场景下正确完成 reduce_scatter，输出数据与数学规约结果一致
- 跨 pod 链路上传输的数据量仅为 level1 规约后的子集，而非全量数据
- Selector 能正确识别 3 层拓扑 + 大数据量场景并选择 3 级算法
- 实现者可从本需求文档直接进入 ce-plan，无需补充产品行为或范围边界

---

## Scope Boundaries

- 不实现流水线（pipeline）模式——3 级流水线同步和 buffer 管理复杂度过高，作为后续优化方向
- 不修改现有 2 级 sequence executor 的行为——3 级是新增，不影响已有场景
- 不改变 NHR 或 Mesh 算法模板本身的内部实现——只新增编排层
- 不支持 4 级及以上拓扑——当前只解决 3 级场景

---

## Key Decisions

- **线性串行 3 步 vs 流水线**: 选择线性串行，因为实现复杂度低、与现有 2 级模式一致、且每级数据量递减使得延迟叠加的实际影响远小于带宽收益
- **CCL Buffer 管理**: 采用与现有非 aicpu executor 一致的双段分区模式（ccl_in + ccl_out），level0→level1 使用 ccl_in→ccl_out 传递，level1→level2 重新复用分区
- **扩展 TopoMatchMultilevel vs 新建**: 选择扩展现有 TopoMatchMultilevel，避免引入新的拓扑匹配类

---

## Dependencies / Assumptions

- TopoMatchMultilevel 的扩展必须在 executor 开发之前完成，executor 依赖 `infos.size() == 3` 的 algHierarchyInfo
- 现有 InsTempReduceScatterNHR 和 InsTempReduceScatterMesh1DZAxisDetour 算法模板可复用，无需修改内部实现
- `REGISTER_EXECUTOR_BY_THREE_TEMPS` 注册宏需要先实现或确认现有宏可扩展
- 假设 CCL Buffer 足够大以支持双段分区 + 3 级 scratchMultiple——若 CCL Buffer 不足，loop 次数会增加

---

## Outstanding Questions

### Resolve Before Planning

- [Affects R4][技术] `REGISTER_EXECUTOR_BY_THREE_TEMPS` 宏是否已存在于代码库中？还是需要新增？需要确认现有注册宏机制能否支持 3 个模板参数

### Deferred to Planning

- [Affects R2][技术] CCL Buffer 双段分区的具体偏移和大小计算——ccl_in 和 ccl_out 的分区比例需要根据各级 scratchMultiplier 动态计算，具体算法留给 planning
- [Affects R6][技术] 3 级 scratchMultiple 的精确计算公式——是 multiplier0 * multiplier1 * multiplier2 还是其他组合，需结合实际算法模板的 CalcScratchMultiple 返回值验证
- [Affects R7, R8, R9][技术] GenIntraTemplateParams / GenInterTemplateParams 中 repeatNum、stride 的精确偏移计算——每级的数据偏移量需要根据实际拓扑规模和 dataTypeSize 计算
- [Affects R10][技术] TopoMatchMultilevel 扩展为 3 层的具体 MatchTopo 逻辑——level2 的子通信域如何划分、rank 映射规则