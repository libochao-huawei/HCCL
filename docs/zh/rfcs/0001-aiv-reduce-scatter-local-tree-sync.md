# AIV ReduceScatter LocalTree 同步优化设计

## 背景

`reduce_scatter` AIV local-tree 路径原实现依赖多处全局同步来保证 CCL buffer 生命周期安全：publish 后同步、fetch 后同步、reduce 轮间同步、store 后跨 rank `BarrierAll()`。这些同步点可以保证正确性，但会让每次 collective 都在尾部等待所有 rank，影响连续小包或多 slice 场景的流水效率。

A3/91093 deterministic `reduce_scatter` 的同步模型更细：publish 完成后用 ACK 类 flag 暴露数据可读，fetch 完成后用 `DataSignal` 类 ack 告诉 peer 已经读完对方 publish 区，local reduce 轮间再用 per-buffer ready flag 管理依赖。因此 local reduce/store 阶段不需要继续依赖跨 rank `BarrierAll()`。

本设计在 A5/HCCL AIV 框架内做最小对齐，不引入 op 间 data buffer 四区 ping-pong，优先解决 local-tree 路径中可删同步和 publish buffer 生命周期问题。

## 目标

1. 保持现有 `2 * rankSize` CCL scratch 布局，继续使用 `[publish][stage/reduce]` 两段数据区。
2. 保留 GM_OUT flag ping-pong，用完整 `curTag_` 作为 flag value，避免只依赖 `tag % 2`。
3. 新增 fetch-done ack，证明 peer 已经读完本 rank publish 区。
4. 将 local reduce 轮间 `SyncAll<true>()` 替换为 per-offset reduce-ready flag。
5. 删除 local-tree 和 local-tree-corectrl 路径中可由细粒度同步替代的 `SyncAll<true>()` 和尾部 `BarrierAll()`。
6. 保留 first-op flag 清理，确保普通 flag 区和第二套 ping-pong flag 区首次使用前均清零。

## 非目标

1. 不修改 host scratch multiple，`CalcScratchMultiple()` 仍为 `2 * tempRankSize_`。
2. 不实现 op 间 data buffer 四区 ping-pong。如果后续必须支持，需要扩展为 `[publish0][stage0][publish1][stage1]`，并重新评估 buffer 容量。
3. 不改变 bigdata 分支同步语义。当前 bigdata 路径仍保留 `BarrierAll()`。
4. 不新增对外 API、环境变量或 host tiling 参数。

## Buffer 布局

每个 rank 的 CCL scratch 仍按 rank shard 划分为两段：

```text
GM_IN[rank]

[publish 区][stage/reduce 区]
 rankSize     rankSize
```

publish 区用于存放本 rank 输入中面向各目标 rank 的 shard，stage/reduce 区用于存放从各 peer publish 区 fetch 到本 rank 的 shard，并在后续 local-tree reduce 中就地归约。

对应 offset：

```text
LocalPublishOffset(targetRank) = targetRank * lenPerRank * sizeof(T)
LocalStageOffset(peerRank)    = (rankSize + peerRank) * lenPerRank * sizeof(T)
```

## Flag 布局

local-tree 新增三类普通 flag offset：

```text
[0, rankSize)              PublishReady flags
[rankSize, 2 * rankSize)   FetchDone flags
[2 * rankSize, 3 * rankSize) ReduceReady flags
```

这些 offset 通过 `Record()` 和 `WaitFlag()` 转换为 `flag_offset * FLAG_SIZE` 访问 `GM_OUT[targetRank]`。以 `MAX_RANK_SIZE = 128` 估算，新增 flag 占用低地址普通 flag 区前 `3 * 128 * FLAG_SIZE`，远小于 `BASE_FLAG_OFFSET`，不与 `BarrierAll()`、`BarrierForFirstOP()` 使用的高地址 barrier 区冲突。

`GM_OUT` 仍按 `tag_ % 2` 选择普通 flag 区或 `GM_OUT_PINGPONG_OFFSET` 后的第二套 flag 区，但 wait/record value 使用完整 generation：

```text
publish-ready value = curTag_
fetch-done value    = curTag_
reduce-ready value  = curTag_ + round + 1
```

## LocalTree 流程

enough-core 路径 `AivReduceScatterV2LocalTree` 的流程调整为：

```text
PublishLocalShard()
FetchPeerShardToLocalStage()
SyncAll<true>()

LocalTreeReduce()
StoreResult()
WaitAllFetchDone()
```

关键变化：

1. publish 后不再执行 rank 内 `SyncAll<true>()`。每个 publish block 在 `CpGM2GM` 和 `pipe_barrier` 后 `Record(targetRank, rank_, curTag_)`，peer fetch 前通过 `WaitFlag(rank_, peerRank, curTag_)` 精确等待对应 publish 完成。
2. fetch 完成后向 peer 写 fetch-done ack：`Record(peerRank, FetchDoneFlagOffset(rank_), curTag_)`。
3. fetch 后仍保留一次 `SyncAll<true>()`，确保本 rank 所有 stage shard 写入完成后再进入 local reduce。
4. local reduce 轮间不再使用全 block `SyncAll<true>()`，而是下一轮读取某个 `offset` 或 `backIdx` 前等待上一轮对应 `ReduceReadyFlagOffset()`。
5. `StoreResult()` 后执行 `WaitAllFetchDone()`，确保 kernel 退出前所有 peer 都已经读完本 rank 本次 publish 区，从而替代原尾部跨 rank `BarrierAll()` 对 publish 区生命周期的保护。

## CoreCtrl 流程

low-core 路径 `AivReduceScatterV2LocalTreeCoreCtrl` 使用同样协议，只是 publish、fetch、fetch-done wait 都按 block 负责的逻辑 rank range 分片执行：

```text
PublishLocalShardRange()
FetchPeerShardRange()
SyncAll<true>()

LocalTreeReduceCoreCtrl()
StoreResult()
WaitAllFetchDone()
```

`SplitLogicalRange()` 保证 `blockNum < rankSize` 时每个 logical rank 都恰好由一个 block 负责 publish/fetch/wait。所有 block 覆盖的 fetch-done wait 集合合并后等价于等待所有 peer。

## ReduceReady 轮间依赖

local-tree reduce 每轮把 `backIdx` 累加到 `offset`。第 `round` 轮完成后，负责 `offset` 的 block 写：

```text
Record(rank_, ReduceReadyFlagOffset(offset), curTag_ + round + 1)
```

下一轮如果需要读取上一轮产生的 `offset` 或 `backIdx`，先等待：

```text
WaitFlag(rank_, ReduceReadyFlagOffset(offset),  curTag_ + round)
WaitFlag(rank_, ReduceReadyFlagOffset(backIdx), curTag_ + round)
```

对于非 2 次幂 rankSize，某些 offset 在某轮没有 paired `backIdx`，仍会 record ready。这保证下一轮读取这些保留下来的 shard 时不会等待缺失 flag。

## First-Op 清理

当前 local-tree kernel entry 使用的普通 `AivCommBase::Init(..., KERNEL_CLASS_INIT, ...)` 路径先 `GetTag(buffIn)` 再 `InitBuffArray(buffIn)`，确保 `GM_OUT` 能按当前 tag 选择正确 ping-pong flag 区。superkernel overload 不在本次 local-tree 同步优化范围内。

`IsFirstOP(sliceId)` 仍以 `sliceId == 1 && tag_ == 1` 判定。首次使用时执行 `BarrierForFirstOP()`，内部 `ClearGM()` 清理两套 flag 区：

```text
GM_OUT[rank_] + blockOffset
GM_OUT[rank_] + GM_OUT_PINGPONG_OFFSET + blockOffset
```

因此新增的 publish-ready、fetch-done、reduce-ready flag 在两套 ping-pong region 中都能被首次清零。

## 正确性约束

1. peer fetch 只能在对应 publish-ready flag 到达 `curTag_` 后开始。
2. 本 rank kernel 退出前必须等待所有 peer 对本 rank publish 区的 fetch-done ack。
3. local reduce 进入下一轮前，必须等待本轮要读取的 `offset` 和 `backIdx` ready。
4. fetch 后到 reduce 前保留一次 rank 内 `SyncAll<true>()`，保证所有 stage shard 对本 rank reduce 可见。
5. `curTag_` 使用完整 generation 和 slice id 组合值，避免相邻 op 或 slice 复用 flag 时误判旧值。
6. `rankSize` 不能超过 `MAX_RANK_SIZE`，新增 flag 区按该上限评估。

## 风险与边界

1. 当前未修改 bigdata 分支，`AivReduceScatterV2Mesh1DBigData` 仍保留 `BarrierAll()`。同一入口不同分支 completion 语义不同，但都保证 kernel 退出前 publish 生命周期安全。
2. 未增加 op-level done flag。当前协议只证明 publish 区已不再被 peer 读取，local output store 完成由 kernel 内 block0 store 和 kernel completion 保证。
3. 未引入 data buffer ping-pong，因此连续 op 的 buffer 复用依赖 kernel completion 和 fetch-done wait，而不是双数据页轮转。
4. 当前本地环境未执行 NPU 编译、UT、ST 或上板验证，需要在配套 CANN/NPU 环境补充验证。

## 验证建议

功能覆盖：

1. `rankSize = 1`。
2. 2 次幂和非 2 次幂 rankSize。
3. `blockNum >= rankSize` 的 local-tree 路径。
4. `blockNum < rankSize` 的 local-tree-corectrl 路径。
5. 连续多次 reduce_scatter 和多 slice 场景。
6. `sum`、`max`、`min` 等 reduce op。
7. `fp16`、`fp32`、`int64` 等典型 dtype。

竞态覆盖：

1. 制造 rank 间执行偏斜，验证慢 rank fetch 不会读到被复用的 publish 区。
2. 连续 op1/op2/op3，验证 tag ping-pong 和完整 `curTag_` 不误判旧 flag。
3. 覆盖 tag reset 后首轮，验证两套 flag 区清理有效。

性能对比：

1. 对比原始 `BarrierAll()` 版本。
2. 对比 fetch-done ack + reduce-ready flag 版本。
3. 分别统计 enough-core、low-core、连续小包和多 slice 场景。
