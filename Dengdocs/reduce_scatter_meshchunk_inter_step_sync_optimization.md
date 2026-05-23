# ReduceScatter MeshChunk 步间同步优化分析

## 1 算法背景

MeshChunk 是 HCCL 中 ReduceScatter 的大数据场景算法（`src/ops/reduce_scatter/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc`），采用 1D Mesh + MeshChunk 模式：

- 共 N-1 步（N = rankSize）
- 每步 N-1 个子操作，每个子操作与不同 peer 做 `SendRecvBatchWriteReduce`（inline reduce）
- 线程按 frontRank（peer rank）分配，每线程固定与一个 peer 通信
- 步间插入全量 barrier（`PreSyncInterThreads` + `PostSyncInterThreads`）

### 四阶段流程

1. **PreCopy**：将本 rank 对应的输入切片拷贝到 scratch buffer
2. **DoMeshChunk**：核心算法，N-1 步 × N-1 子操作，每子操作 SendRecvReduce
3. **步间同步**：每步结束后全量 barrier（共 N-2 次）
4. **PostCopy**：将 scratch buffer 中归约完成的最终结果拷贝到用户 output buffer

### 同步开销量化（N 个 rank）

- N-2 次全量 barrier
- 每次 barrier = 4 notify round-trip（main→sub record, sub→main wait, sub→main record, main→sub wait）
- Scratch 倍数 = N-1

---

## 2 步间 Barrier 的根因分析

MeshChunk 每步对所有 peer 执行 `SendRecvBatchWriteReduce`，**接收端做 inline reduce**——将远程数据 reduce 到本地 scratch buffer 的**同一组 recvOffset**。因此：

- 步 k 中线程 T1 在 `hcclBuff[recvOffset_j]` 上做了 reduce
- 步 k+1 中线程 T2 要在同一 `hcclBuff[recvOffset_j]` 上再做 reduce
- **T2 必须等 T1 完成**，否则 reduce 读到不完整中间值

本质原因：**hcclBuff 的 recvOffset 跨步复用，且跨线程共享**。

---

## 3 消除步间 Barrier 的思路与冲突

### 3.1 思路：Thread-by-Offset（按 offset 分线程）

如果每个 recvOffset 始终由同一线程处理，则同一 stream 上的 reduce 操作自然 FIFO 有序，步 k+1 的 reduce 在步 k 的 reduce 之后自动执行，无需跨线程 barrier。

关键验证：recvOffset 在各步间是否固定？——是的。每步开始 `sliceSendOffset_=0, sliceRecvOffset_=sliceRecvBaseOffset`，内层循环对同一 i 的 offset 累加值每步完全相同，与 stepIdx 无关。变化的只是 frontRank（通信对端）。

### 3.2 Channel Notify 错位问题

Thread-by-offset 导致同一 channel 在不同步被不同线程使用，channel notify 操作来自不同 stream，FIFO 顺序被打破。

`SendRecvBatchWriteReduce` 的 4 阶段握手使用 channel notify：
- Phase 1: ACK record on recv channel（信号从本 rank 到远端）
- Phase 2: ACK wait on send channel（等待远端信号）
- Phase 3: 数据传输 + reduce
- Phase 4: DATA_SIGNAL record + DATA_SIGNAL wait

每个方向（A→B, B→A）各有 NOTIFY_IDX_ACK=0 和 NOTIFY_IDX_DATA_SIGNAL=1。当不同线程在不同步使用同一 channel 时，notify 操作来自不同 stream，可交错执行。

### 3.3 死锁推演（N=3 最简场景）

步 0 与步 1 的线程-channel 映射完全翻转：

| channel | 步 0 使用线程 | 步 1 使用线程 |
|---|---|---|
| 0↔1 | rank0 thread1 ↔ rank1 thread0 | rank0 thread0 ↔ rank1 thread1 |
| 0↔2 | rank0 thread0 ↔ rank2 thread1 | rank0 thread1 ↔ rank2 thread0 |
| 1↔2 | rank1 thread1 ↔ rank2 thread0 | rank1 thread0 ↔ rank2 thread1 |

**每个 channel 在相邻两步间都切换了线程！**

Flag-based notify 死锁场景（以 channel 0↔1 的 A→B 方向为例）：

```
时刻1: rank0 thread1(步0) ACK_record → flag = 1
时刻2: rank0 thread0(步1) ACK_record → flag = 1（已为1，信号丢失！）
时刻3: rank1 thread0(步0) ACK_wait  → flag=1→0，消耗了一个信号 ✓
时刻4: rank1 thread1(步1) ACK_wait  → flag=0，永远阻塞 ← DEADLOCK
```

Counter-based notify 不死锁（counts 总匹配），但 DATA_SIGNAL 匹配错位导致**数据正确性问题**：步 k+1 的 DATA_SIGNAL wait 可能消费步 k 的 DATA_SIGNAL record，导致步 k+1 在远端数据写入未完成时执行 reduce，读到不完整数据。

### 3.4 根本冲突

- 消除 intra-rank barrier 需要：线程按 offset 分（同 offset 同线程 → stream 有序）
- 避免 cross-rank notify 错位需要：线程按 frontRank 分（同 channel 同线程 → notify FIFO 有序）

这两个目标在 MeshChunk 的 frontRank 跨步变化模式下**互斥**。

---

## 4 2x Thread 方案评估

### 4.1 方案描述

将线程数翻倍，分为 TX 组和 RX 组，期望同一 channel 不会被不同线程使用。

### 4.2 三种分配策略分析

#### 策略 A：TX/RX 都按 frontRank 分

Channel notify 各方向各 index 的线程归属跨步固定，notify FIFO 保序 ✓。但 offset 依赖仍在：步 k 中 `RX_thread(B)` 在 `recvOffset_i` 做 reduce，步 k+1 中 `RX_thread(C)` 在同一 `recvOffset_i` 做 reduce。**仍需步间 barrier** ✗。

#### 策略 B：TX/RX 都按 offset(i) 分

Offset 依赖消除 ✓。但步 k 中 `RX_thread(i_B(k))` 在 channel A↔B 做 ACK record，步 k+1 中 `RX_thread(i_B(k+1))`（不同线程）在同一 channel 做 ACK record。**channel notify 错位问题重现** ✗。

#### 策略 C：TX 按 frontRank + RX 按 offset（混合）

TX_thread(B) 固定 channel → notify FIFO 保序 ✓。RX_thread(j) 固定 offset → stream 保序 ✓。RX 不直接操作 channel notify，由 TX_thread 代劳，RX 只通过 intra-rank thread notify 与 TX 协调。

协议设计：

```
RX_thread(j) @ 步k:
  1. thread_notify → TX_thread(frontRank(k, j)): "recvOffset_j 已就绪"
  2. thread_notify_wait ← TX_thread(frontRank(k, j)): "远端数据已写入完成"
  3. LocalReduce(recvOffset_j)

TX_thread(B) @ 步k:
  1. thread_notify_wait ← RX_thread(i_B(k)): "本地 buffer 就绪"
  2. channel ACK_record (Dir A→B)
  3. channel ACK_wait   (Dir B→A)
  4. TX: Write to remote scratch
  5. channel DATA_SIGNAL_record (Dir A→B)
  6. channel DATA_SIGNAL_wait   (Dir B→A)
  7. thread_notify → RX_thread(i_B(k)): "远端数据可用"
```

死锁检查：RX 先发后收，TX 先收后发，无循环等待 ✓。跨 rank ACK 对称握手 ✓。

开销（N=8）：14 线程，0 步间 barrier，98 次 intra-rank thread notify（per sub-op）。98 次 stream 上的轻量 device 操作仍偏高，**方案整体代价过大**。

---

## 5 备选方案

### 方案 A：Per-step Channel Notify Index——最干净的 MeshChunk 优化

**核心思路**：保留 thread-by-offset（消除 offset 依赖），给每步分配独立的 channel notify index，避免步间 notify 错位。

```
步0: NOTIFY_IDX_ACK=0,  NOTIFY_IDX_DATA_SIGNAL=1
步1: NOTIFY_IDX_ACK=2,  NOTIFY_IDX_DATA_SIGNAL=3
步k: NOTIFY_IDX_ACK=2k, NOTIFY_IDX_DATA_SIGNAL=2k+1
```

每个 notify index 在每个方向上只被 record 1 次 + wait 1 次，天然无错位。线程按 offset 分配，stream 保序，**零 barrier + 零 thread notify**。

代价：每 channel 需 2(N-1) 个 notify index。

| N | notify index/channel |
|---|---|
| 4 | 6 |
| 8 | 14 |
| 16 | 32 |

需确认 Ascend NPU 的 channel notify index 硬件上限。若支持数十个 index，此方案最优。

### 方案 B：小 N 单线程——零成本零 Barrier

N≤4 时线程数极少，并行收益有限，barrier 开销占比高。改为单线程执行：

- 所有子操作在同一 stream 上 FIFO 排队
- 步间自然有序，无需任何同步
- 实现：`threadNum=1, slaveThreadNum=0`，循环提交 SendRecvReduce

N=3 对比：
- 多线程：2步 × 2子操作/步 × 1 barrier ≈ 2×(并行时间 + barrier)
- 单线程：2步 × 2子操作/步 × 0 barrier ≈ 4×(单操作时间)

N≤4 时并行度仅 2~3，barrier 开销可能抵消并行收益，单线程反而更快。

### 方案 C：Barrier 开销压缩——不改算法结构，渐进优化

#### C1. Notify 提交与数据传输重叠

当前：内层循环完 → 提交 PreSync → 提交 PostSync → 下一步

优化：内层循环中最后一个子操作之后，在同一 stream 上紧接提交 notify record。Stream FIFO 保证 notify 在最后一个数据操作完成后才生效。每次 barrier 约省 1 个子操作时间的感知延迟。

#### C2. 减少 Notify Round-trip

当前 PreSync + PostSync = 4 次 notify。可尝试合并为 2 次：所有 slave 线程向 main 的同一 notify index record（位域合并），main 向所有 slave 广播一次 record。需硬件支持位域 notify 或集体 notify。

#### C3. 只同步必要线程

分析跨步依赖图，只对共享同一 recvOffset 的线程对做定向 notify，而非全量 barrier。N=8 时全量 barrier 每次等 7 个线程；定向 notify 每次只等 1~2 个。仍需解决 channel notify 问题（需方案 A 配合）。

### 方案 D：选择器层面调整——最省力

MeshChunk 当前仅在 `dataSize * ratio > 16MB` 时被选中。NHR 在同等场景下零 barrier，只是 scratch 倍数更大（N vs N-1）。

调整策略：
- 放宽 NHR 适用范围：scratch 充足时优先走 NHR
- NHR + MeshChunk 混合：先 NHR reduce-scatter（log2N 步零 barrier），再处理余量

已有代码参数调优，不需要新算法。

### 方案 E：Ring-pipeline 重设计——零 Barrier + 可控带宽

将 MeshChunk 的全连接模式改为固定 peer 环形流水线：

```
每个 rank 有固定左/右邻居
步k: 从左邻居收 → reduce → 向右邻居发
同 offset 同 peer → thread-by-offset = thread-by-peer
零 barrier, 零 notify 错位
```

带宽利用率低于 MeshChunk（每步只用 2 条链路 vs N-1 条），可通过 chunk 并行弥补：数据切成 M 个 chunk，M 个线程各跑一个 chunk 的 ring pipeline，M 条链路同时活跃。延迟 ≈ (N-1)/M × 单 chunk 时间。

本质是 NHR 的另一种形态，需完整新算法实现。

---

## 6 推荐优先级

| 优先级 | 方案 | 理由 |
|---|---|---|
| P0 | B（小N单线程） | 立刻可做，N≤4 场景直接受益，实现极简 |
| P1 | A（per-step notify index） | 若硬件支持，是 MeshChunk 的终极优化，零 barrier 零 thread notify |
| P2 | D（选择器调优） | 低风险增量优化，已有 NHR 实现可复用 |
| P3 | C（barrier 压缩） | 渐进改善，不依赖硬件变更，可与任何方案叠加 |
| P4 | E（ring-pipeline） | 长期方向，需完整新算法 |

**最关键的一步**：确认 Ascend NPU channel notify index 的硬件上限。若 ≥16（覆盖 N=8），方案 A 可直接落地，MeshChunk 的步间同步问题彻底消除。

---

## 7 代码中已有的替代算法同步模式对比

| 算法 | 步间同步 | 原理 | 代价 |
|---|---|---|---|
| Mesh 1D（diagonal） | 0 | 各 rank 并行写入所有 peer 的不同 offset，reduce 延后到 PostCopy | scratch 倍数 = N |
| NHR | 0 | log2N 步，每步操作不相交 slice 组，天然无依赖 | scratch 倍数 = N |
| OmniPipe NHR | 0 | 单线程 + executor 控制跨级同步 | 需 OmniPipe 框架 |
| CCU Mesh1D Mem2Mem | 0 | 全量并行 ReadNb + 本地 reduce | 仅 CCU 硬件可用 |
| MeshChunk | N-2 次全量 barrier | inline reduce 共享 recvOffset 跨步复用 | scratch 倍数 = N-1 |

关键洞察：只要每步操作的 scratch offset 不重叠，或者 reduce 不 inline 而延后，就不需要步间同步。MeshChunk 的 per-step sync 是 inline reduce + offset 复用的代价。