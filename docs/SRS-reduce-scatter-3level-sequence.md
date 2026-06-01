# SRS: ReduceScatter 三级序列执行器 (3-Level Sequence Executor)

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | HCCL ReduceScatter 三级序列执行器 |
| 文档版本 | V1.0 |
| 对应 PR | MR #1158 |
| 编写日期 | 2026-06-01 |

## 2. 概述

### 2.1 背景

HCCL (Huawei Collective Communication Library) 在 ReduceScatter 算子中已支持二级序列执行器（2-Level Sequence Executor），用于在框内（Level0）+ 跨框（Level1）的两层拓扑场景下，通过 Mesh→NHR 的串行编排实现分层规约。

随着集群规模扩大，出现了三级拓扑结构（框内 Level0 → Pod 内 Level1 → 跨 Pod Level2），例如 8P×8Pod×2Cluster 的 128 卡部署。在现有实现中，三级拓扑被降级为二级处理，Level1 和 Level2 合并为一个大的 NHR 执行，导致跨 Pod 的低带宽链路（仅 4 端口收敛）成为瓶颈，框内和 Pod 内的高带宽资源被浪费。

### 2.2 目标

新增三级序列执行器 `InsV2ReduceScatterSequenceExecutor3Level`，在三级拓扑场景下实现 Mesh(Level0) → NHR(Level1) → NHR(Level2) 的分层串行规约，充分利用各级链路带宽，减少跨 Pod 数据传输量。

### 2.3 术语定义

| 术语 | 定义 |
|------|------|
| Level0 | 框内拓扑层，使用 Mesh1D 算法模板 |
| Level1 | Pod 内跨框拓扑层，使用 NHR 算法模板 |
| Level2 | 跨 Pod 拓扑层，使用 NHR 算法模板 |
| Sequence Executor | 序列执行器，按层级串行编排多级算法模板 |
| algHierarchyInfo | 算法层级信息，描述各级子通信域的 rank 划分 |
| scratchMultiple | CCL Buffer 的空间放大系数 |
| CCL Buffer | 集合通信库内部缓冲区，用于级间数据传递 |

## 3. 功能需求

### 3.1 FR-01: 三级序列执行器类

**需求描述**: 新建 `InsV2ReduceScatterSequenceExecutor3Level` 类，作为通用三级编排层，接受 3 个算法模板参数 `<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>`，不绑定特定引擎（AICPU/CCU 均可复用）。

**输入**:
- OpParam: 算子参数（输入输出指针、数据类型、数据量等）
- AlgResourceCtxSerializable: 序列化资源上下文

**输出**:
- 每个 rank 的 outputPtr 包含正确规约后的数据切片

**约束**:
- `SEQUENCE_EXECUTOR_LEVEL_NUM = 3`
- 继承自 `InsCollAlgBase`

### 3.2 FR-02: 线性串行三级数据流

**需求描述**: 数据流为线性串行三步：INPUT → Mesh(Level0) → CCL Buffer → NHR(Level1) → CCL Buffer → NHR(Level2) → OUTPUT，每步串行执行，级间数据通过 CCL Buffer 传递。

**数据流路径**:
```
INPUT → [Mesh1D Level0] → CCL Buffer → [NHR Level1] → CCL Buffer → [NHR Level2] → OUTPUT
```

**约束**:
- 不实现流水线（pipeline）模式
- 每级完成后数据才传递至下级

### 3.3 FR-03: Loop 分片循环

**需求描述**: 支持循环分片处理大数据量，每轮循环依次执行三级算法，与现有二级的循环机制一致。当 CCL Buffer 不足以容纳全部数据时，通过循环分批处理。

**计算规则**:
- `templateScratchMultiplier = multiplier0 × multiplier1 × multiplier2`
- `maxCountPerLoop = (cclMem.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN) × HCCL_MIN_SLICE_ALIGN / dataTypeSize_`
- `loopTimes = dataCount_ / maxCountPerLoop + (dataCount_ % maxCountPerLoop != 0 ? 1 : 0)`

### 3.4 FR-04: 注册宏支持三级模板

**需求描述**: 新增 `REGISTER_EXECUTOR_BY_THREE_TEMPS` 宏，支持 3 个算法模板参数的执行器注册，注册名称为 `InsReduceScatterSequenceMesh1DNHRNHR`。

**注册调用**:
```
REGISTER_EXECUTOR_BY_THREE_TEMPS(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
    InsReduceScatterSequenceMesh1DNHRNHR,
    InsV2ReduceScatterSequenceExecutor3Level,
    TopoMatchMultilevel,
    InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR,
    InsTempReduceScatterNHR)
```

### 3.5 FR-05: 三级资源计算

**需求描述**: CalcRes 为三级分别计算 AlgResourceRequest，合并结果：

| 资源项 | 合并策略 |
|--------|---------|
| slaveThreadNum | max(res0, res1, res2) |
| notifyNumPerThread | 各级对应线程取 max |
| notifyNumOnMainThread | max(res0, res1, res2) |
| channels | 拆为 3 层 channels[0/1/2]，分别取各级 channels[0] |

### 3.6 FR-06: Level0 模板参数生成 (GenIntraTemplateParams)

**需求描述**: 为 Level0 Mesh 算法模板生成参数。

| 参数 | 值 |
|------|-----|
| inBuffType | INPUT |
| outBuffType | HCCL_BUFFER |
| hcclBuffType | HCCL_BUFFER |
| repeatNum | rankSizeLevel1 × rankSizeLevel2 |
| inputSliceStride | dataSize_ |
| outputSliceStride | currDataCount × dataTypeSize_ |
| inputRepeatStride | rankSizeLevel0 × dataSize_ |
| outputRepeatStride | rankSizeLevel0 × currDataCount × dataTypeSize_ |

**语义**: 框内需要为每个更高层级的框重复执行，repeatNum = rankSizeLevel1 × rankSizeLevel2。

### 3.7 FR-07: Level1 模板参数生成 (GenInterTemplateParams1)

**需求描述**: 为 Level1 NHR 算法模板生成参数。

| 参数 | 值 |
|------|-----|
| inBuffType | HCCL_BUFFER |
| outBuffType | HCCL_BUFFER |
| hcclBuffType | HCCL_BUFFER |
| repeatNum | rankSizeLevel2 |
| inBuffBaseOff | rankIdxLevel0 × currDataCount × dataTypeSize_ |
| inputSliceStride | rankSizeLevel0 × currDataCount × dataTypeSize_ |
| outputSliceStride | 0 |
| inputRepeatStride | rankSizeLevel0 × rankSizeLevel1 × currDataCount × dataTypeSize_ |
| outputRepeatStride | 0 |

**语义**: Pod 内需要为每个跨 Pod 组重复执行，repeatNum = rankSizeLevel2。

### 3.8 FR-08: Level2 模板参数生成 (GenInterTemplateParams2)

**需求描述**: 为 Level2 NHR 算法模板生成参数。

| 参数 | 值 |
|------|-----|
| inBuffType | HCCL_BUFFER |
| outBuffType | OUTPUT |
| hcclBuffType | HCCL_BUFFER |
| repeatNum | 1 |
| inBuffBaseOff | rankIdxLevel0 × currDataCount × dataTypeSize_ |
| outBuffBaseOff | processedDataCount × dataTypeSize_ |
| inputSliceStride | rankSizeLevel0 × currDataCount × dataTypeSize_ |
| outputSliceStride | 0 |

**语义**: 跨 Pod 不需要重复，repeatNum = 1。

### 3.9 FR-09: 拓扑匹配扩展

**需求描述**: 扩展 `TopoMatchMultilevel` 支持三级拓扑，新增 `TopoForLayer2` 方法，生成 `algHierarchyInfo.infos.size() == 3` 的层级信息。

**Layer2 划分逻辑**:
1. 获取指定网络层的拓扑实例
2. 验证 topoInstNum == 1
3. 获取该拓扑实例下的所有 rank
4. 筛选与当前 rank 在 Level0 和 Level1 具有相同索引的 rank（即 `rankId % (layer0Size × layer1Size) == myRank % (layer0Size × layer1Size)`）
5. 验证 rank 间存在链路连接
6. 将筛选结果写入 `algHierarchyInfo.infos[2]`

**约束**:
- `COMM_LAYER_SIZE_3` 支持最大 3 层拓扑
- `topoLevelNums > COMM_LAYER_SIZE_3` 时返回错误
- 2 层拓扑场景下行为不变

### 3.10 FR-10: 算法选择器更新

**需求描述**: 在 `ReduceScatterAutoSelector::SelectAicpuAlgo` 中新增分支：当 `topoInfo->topoLevelNums >= 3` 时，选择 `InsReduceScatterSequenceMesh1DNHRNHR` 算法。

**选择逻辑**:
```
if (localNetInsSizeOfLayer[0] > 1 && level0Topo == MESH_1D) {
    if (topoLevelNums >= 3) {
        → "InsReduceScatterSequenceMesh1DNHRNHR"   // 新增
    } else if (dataSize > RS_AICPU_1D_MIN_DATA_SIZE) {
        → "InsReduceScatterSequenceMesh1DNhr" / "InsReduceScatterParallelMesh1DNHR"
    } else {
        → "InsReduceScatterParallelMesh1DNHR"
    }
}
```

**约束**:
- 三级拓扑场景统一走三级算法，不区分数据量
- 不影响现有二级拓扑的选择逻辑

## 4. 非功能需求

### 4.1 NFR-01: 正确性

三级序列执行器在三级拓扑场景下正确完成 ReduceScatter，输出数据与数学规约结果一致。

### 4.2 NFR-02: 带宽优化

跨 Pod 链路上传输的数据量仅为 Level1 规约后的子集，而非全量数据。相比二级合并方案，减少跨 Pod 数据传输量。

### 4.3 NFR-03: 兼容性

- 不修改现有二级序列执行器的行为
- 不改变 NHR 或 Mesh 算法模板本身的内部实现
- 不支持四级及以上拓扑

### 4.4 NFR-04: 可复用性

`InsV2ReduceScatterSequenceExecutor3Level` 为通用三级编排层，不绑定 AICPU 引擎，后续 CCU 等引擎可复用。

## 5. 接口需求

### 5.1 外部接口

| 接口 | 类型 | 描述 |
|------|------|------|
| `Orchestrate(param, resCtx)` | 覆写 | 执行三级序列编排 |
| `CalcRes(comm, param, topoInfo, algHierarchyInfo, resourceRequest)` | 覆写 | 计算三级资源需求 |
| `CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo)` | 覆写 | 计算三级算法层级信息 |

### 5.2 内部接口

| 接口 | 类型 | 描述 |
|------|------|------|
| `OrchestrateLoop(param, resCtx)` | protected | 循环分片执行三级算法 |
| `InitCommInfo(param, topoInfo, algHierarchyInfo)` | protected | 初始化通信信息 |
| `GenIntraTemplateParams(...)` | protected | 生成 Level0 模板参数 |
| `GenInterTemplateParams1(...)` | protected | 生成 Level1 模板参数 |
| `GenInterTemplateParams2(...)` | protected | 生成 Level2 模板参数 |
| `GenTempResource(...)` | protected | 生成模板资源 |

## 6. 验收标准

### 6.1 AC-01: 功能正确性

**Given** 8P×8Pod×2Cluster（128 卡三级拓扑），**When** 执行 ReduceScatter 且数据量 > 1GB，**Then** executor 在循环内依次执行 Mesh(Level0, repeatNum=16) → NHR(Level1, repeatNum=2) → NHR(Level2, repeatNum=1)，最终每个 rank 的 outputPtr 包含正确规约后的数据切片。

### 6.2 AC-02: 资源合并

**Given** 三级资源请求 resReq0(slaveThreadNum=2), resReq1(slaveThreadNum=3), resReq2(slaveThreadNum=1)，**Then** 合并后 resourceRequest.slaveThreadNum = 3，channels 包含 3 层各自独立的 channel 映射。

### 6.3 AC-03: 算法选择

**Given** topoLevelNums >= 3 且 level0Topo == MESH_1D，**Then** SelectAicpuAlgo 返回 "InsReduceScatterSequenceMesh1DNHRNHR"。

### 6.4 AC-04: 向后兼容

**Given** 二级拓扑场景，**Then** ReduceScatter 行为与变更前完全一致。

## 7. 范围边界

| 包含 | 不包含 |
|------|--------|
| 三级序列执行器实现 | 流水线（pipeline）模式 |
| 拓扑匹配扩展至三级 | 修改现有二级序列执行器 |
| 算法选择器三级分支 | 修改 NHR/Mesh 算法模板内部实现 |
| 注册宏三级支持 | 四级及以上拓扑支持 |
| CMakeLists 构建配置 | |

## 8. 依赖与假设

### 8.1 依赖

- TopoMatchMultilevel 的扩展必须在 executor 开发之前完成
- 现有 `InsTempReduceScatterNHR` 和 `InsTempReduceScatterMesh1DZAxisDetour` 算法模板可复用
- `REGISTER_EXECUTOR_BY_THREE_TEMPS` 注册宏需先实现

### 8.2 假设

- CCL Buffer 足够大以支持三级 scratchMultiple——若 CCL Buffer 不足，循环次数会增加
- 三级拓扑下 Level0 为 Mesh1D 拓扑
- 三级拓扑下 Level1 和 Level2 均为 NHR 拓扑
