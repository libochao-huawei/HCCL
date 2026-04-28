# Ring Pipeline AllGather 算法说明

## 算法名称

`InsAllGatherRingPipeline` (AICPU) / `CcuAllGatherRingPipeline` (CCU) — 环形流水线全收集算法

## 算法思想

将传统的 **一步全互联广播（GroupBroadcast）** 改为 **环形多步流水线通信**，以步数换取链路带宽均衡。

### 核心思路

1. **数据切分**：将 AllGather 的总数据量按 rank 数切分为 N 个等大的 chunk
2. **环形拓扑**：N 个 rank 组成一个逻辑环，每个 rank 只有两个邻居——前驱（prev）和后继（next）
3. **流水线转发**：执行 N-1 步，每一步每个 rank 同时做两件事：
   - 将自己持有的 chunk（或上一步转发的 chunk）发给后继
   - 从前驱接收一个 chunk
   - 将接收到的 chunk 写入 scratch buffer 的对应位置
4. **完成**：N-1 步后，每个 rank 都收到了所有其他 rank 的 chunk，从 scratch 拷贝到 output

### 以 4 rank 为例

```
初始状态（每个 rank 持有自己的 chunk）:
  R0: [chunk0]   R1: [chunk1]   R2: [chunk2]   R3: [chunk3]

Step 1: 每个 rank 把自己的 chunk 发给后继
  R0→R1: chunk0    R1→R2: chunk1    R2→R3: chunk2    R3→R0: chunk3
  结果:
  R0: [chunk0, chunk3]   R1: [chunk0, chunk1]   R2: [chunk1, chunk2]   R3: [chunk2, chunk3]

Step 2: 每个 rank 把刚收到的 chunk 转发给后继
  R0→R1: chunk3    R1→R2: chunk0    R2→R3: chunk1    R3→R0: chunk2
  结果:
  R0: [chunk0, chunk3, chunk2]   R1: [chunk0, chunk1, chunk3]   ...

Step 3: 继续转发
  R0→R1: chunk2    R1→R2: chunk3    R2→R3: chunk0    R3→R0: chunk1
  结果:
  R0: [chunk0, chunk3, chunk2, chunk1] ✅   ...
```

## 与现有算法对比

| 维度 | Mesh1D (现有) | NHR (现有) | Ring Pipeline (新) |
|------|--------------|-----------|-------------------|
| 通信模式 | 全互联，同时发给所有 rank | 非对称环减半 | 环形流水线 |
| 线程数 | rankSize-1 | 1 | 1 |
| 步数 | 1 | ceil(log2(N)) | N-1 |
| 每步链路数 | rankSize-1 | 1 | 1 |
| 每步传输量 | 全量数据 | 递增 | 固定 1 份 chunk |
| 大数据性能 | 线程开销大 | 链路负载不均 | **链路均衡，内存友好** |
| 适用场景 | 小数据量 | 通用 | **大数据量（>8MB）+ 4~8P** |

## 选择器匹配条件

### AICPU RingPipeline（SelectAicpuAlgo）
```cpp
// MESH_1D 拓扑 + 大数据量(>8MB) + 多 rank(≥4P) → 选择 Ring Pipeline
if (dataSize > 8MB && topoInfo->userRankSize >= 4) {
    selectAlgName = "InsAllGatherRingPipeline";
    return SelectorStatus::MATCH;
}
```

### CCU RingPipeline（SelectCcuScheduleLevel0Algo）
```cpp
// MESH_1D 拓扑 + 大数据量(>4MB) + 多 rank(≥8P) → 选择 Ring Pipeline
if (dataSize > 4MB && topoInfo->userRankSize >= 8) {
    selectAlgName = "CcuAllGatherRingPipeline";
    return SelectorStatus::MATCH;
}
```

## 代码文件

| 文件 | 职责 |
|------|------|
| `template/aicpu/ins_temp_all_gather_ring_pipeline.h` | AICPU 模板头文件 |
| `template/aicpu/ins_temp_all_gather_ring_pipeline.cc` | AICPU 模板实现 |
| `executor/ins_v2_all_gather_sole_executor.cc` | 注册执行器（REGISTER_EXEC_V2） |
| `selector/all_gather_auto_selector.cc` | 选择器匹配逻辑（SelectAicpuAlgo + SelectCcuScheduleLevel0Algo） |
| `RING_PIPELINE_ALGORITHM.md` | 算法说明文档 |

## AICPU 实现要点

### 通信模式
使用 `SendRecvWrite`（或 `SendRecvRead`，PCIE 链路场景）进行 rank 间数据传输，单线程串行完成 N-1 步通信。

### 数据流
```
input buffer → LocalDataCopy → scratch buffer → RingPipeline(N-1步) → scratch buffer → PostLocalCopy → output buffer
```

### 关键函数
- `RunRingPipelineAllGather()`: 核心通信循环，N-1 步 SendRecv
- `LocalDataCopy()`: input → scratch 初始拷贝
- `PostLocalCopy()`: scratch → output 最终拷贝

## 设计原则

1. **遵循 HCCL 四层架构**：Selector → Executor → Template → 通信原语
2. **复用基类基础设施**：继承 `InsAlgTemplateBase`，复用通道管理、LocalCopy、SendRecv 等
3. **最小化新增接口**：仅新增 `InsTempAllGatherRingPipeline` 一个类
4. **与现有算法互补**：小数据走 Mesh1D，大数据走 RingPipeline
