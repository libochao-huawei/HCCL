# ReduceScatter AIV LocalTree 设计

## 1. 目标

在 `hccl/src/ops/reduce_scatter/template/aiv/kernel/` 新增 AIV kernel：
`aiv_reduce_scatter_local_tree.h`。

该 kernel 用于调试阶段直接替代现有 reduce_scatter AIV 分发中的：
- `AivReduceScatterV2Mesh1D`
- `AivReduceScatterV2Mesh1DCoreCtrl`
- `AivReduceScatterV2Mesh1DBigData`

第一版目标：
- 调试阶段让 `aiv_reduce_scatter_##type` 入口统一走新 kernel。
- 仅支持足够核数场景，要求 `numBlocks >= rankSize`。
- 暂不引入控核版本。
- 不保留旧 `91093` 命名或专用设备语义。

核心思路：
- 每个 rank 先将发往各目标 rank 的输入分片发布到本地 CCL 区。
- 当前 rank 从每个 peer 的 CCL 区拉取自己需要的分片到本地 staging 区。
- 在本地 staging 区做二分树形 reduce。
- 将归约结果写回 output。

## 2. 背景

现有 `hccl` AIV reduce_scatter 主要实现包括：
- `aiv_reduce_scatter_mesh_1d.h`
- `aiv_reduce_scatter_mesh_1d_corectrl.h`
- `aiv_reduce_scatter_mesh_1d_bigdata.h`

其中 `mesh_1d` / `corectrl` 采用顺序累加，不包含本地树形 reduce。

`hcomm` 中的 `aiv_reduce_scatter_91093_deter.h` 提供了“本地聚合 + 本地树形 reduce”的思路，但依赖：
- `AivCrossNode91093Base`
- cross-node 定制同步
- `buffIn0/buffOut0/buffOut1` 参数
- 双半区 buffer 布局

因此 `hccl` 侧不能直接搬运该实现，只抽取算法流程并适配 `AivCommBase`。

## 3. 适配原则

### 3.1 保持 kernel ABI 不变

新 kernel 继续使用：
- `EXTERN_KERNEL_ARGS_DEF_V2`
- `KERNEL_CLASS_INIT`
- `AivCommBase`
- `buffIn -> GM_IN[] / GM_OUT[]` 地址初始化方式

Host 侧 selector / executor / template 类名暂不调整，调试阶段只在 device kernel 入口切换分发。

### 3.2 使用 hccl 现有通信 buffer

新 kernel 不引入 `hcomm` 的 `buffIn0/buffOut0/buffOut1` 接口。

每张卡的 `GM_IN[rank_]` 被划分为两个连续逻辑区域：
- publish 区：`[0, rankSize * len)`，保存本 rank 发往所有目标 rank 的原始输入分片。
- stage/reduce 区：`[rankSize * len, 2 * rankSize * len)`，保存当前 rank 从所有 peer 拉回的待归约分片，并在该区域内完成树形 reduce。

### 3.3 scratch multiple

原 `CalcScratchMultiple()` 返回 `tempRankSize_`，只够现有 `mesh_1d` 单区发布语义。

LocalTree 如果只复用单区，会出现 peer 间互相覆盖风险：当前 rank 将 peer `p` 的数据拉到本地 slot `p` 时，会覆盖本 rank 发布给 peer `p`、且 peer `p` 可能尚未读取的数据。这个 pairwise swap hazard 不能仅靠本地同步消除。

因此第一版将 `CalcScratchMultiple()` 调整为：
- `2 * tempRankSize_`

这使 publish 区和 stage/reduce 区分离，保证远端读取与本地归约互不覆盖。

## 4. 同步能力

### 4.1 直接复用

新 kernel 直接复用：
- `SyncAll<true>()`
- `Record(...)`
- `WaitFlag(...)`
- `BarrierAll()`
- `CpGM2GM(...)`

### 4.2 发布同步

每个 producer block 发布一个目标 rank 的输入分片后执行：
- `Record(targetRank, rank_, curTag_)`

对应目标 rank 拉取该 peer 分片前执行：
- `WaitFlag(rank_, peerRank, curTag_)`

语义是：peer 在当前 rank 的 flag 区通知“peer 发给当前 rank 的分片已发布完成”。

### 4.3 本地 reduce 同步

本地树形 reduce 轮次之间使用 `SyncAll<true>()`，保证上一轮对 stage 区的写入在下一轮读取前完成。

第一版每轮最多使用 `rankSize` 个 block 参与本地 reduce，因此只覆盖 `numBlocks >= rankSize` 的足够核数场景。

## 5. 算法流程

### Step1 本端 input -> 本端 publish 区

对 `targetRank = blockIdx`：
- src: `input + targetRank * inputSliceStride`
- dst: `GM_IN[rank_] + targetRank * len * sizeof(T)`
- count: `len`

完成后通知目标 rank。

### Step2 peer publish 区 -> 本端 stage 区

对 `peerRank = blockIdx`：
- 等待 peer 对本 rank 的发布 flag。
- src: `GM_IN[peerRank] + rank * len * sizeof(T)`
- dst: `GM_IN[rank_] + (rankSize + peerRank) * len * sizeof(T)`
- count: `len`

### Step3 本地树形 reduce

在本地 stage 区上执行二分树形 reduce。

每轮：
- `powerOf2 = largest_power_of_2_less_than(curBlocks)`
- 将 `[powerOf2, curBlocks)` 中存在的后半元素 reduce 到 `[0, powerOf2)` 对应前半元素。
- 轮次结束后 `curBlocks = powerOf2`。

最终结果位于：
- `GM_IN[rank_] + rankSize * len * sizeof(T)`

### Step4 本端 stage 区 -> output

由 block 0 将最终结果拷贝到 output：
- src: `GM_IN[rank_] + rankSize * len * sizeof(T)`
- dst: `output`
- count: `len`

## 6. 代码改动范围

已修改：
- `src/ops/reduce_scatter/template/aiv/kernel/aiv_reduce_scatter_local_tree.h`
- `src/ops/reduce_scatter/template/aiv/kernel/aiv_communication_v2.h`
- `src/ops/reduce_scatter/template/aiv/aiv_temp_reduce_scatter_mesh_1D.cc`
- `demo/tmp/aiv_reduce_scatter_local_tree_design.md`

暂不修改：
- selector
- executor
- template 类命名
- 算法注册名

## 7. 限制与风险

第一版限制：
- 要求 `numBlocks >= rankSize`。
- 少核/控核版本暂未实现。
- BigData 原分支在调试阶段也被统一替换为 LocalTree。
- 未对数据量超出 `2 * rankSize * len` scratch 空间之外的场景做额外流水分块。

需要重点验证：
- AIV CCE 编译是否接受新增模板和 `SyncAll<true>()` 使用位置。
- `2 * tempRankSize_` scratch 是否与资源申请和 buffer 地址初始化完全匹配。
- `sum/max/min` 以及 `int64_t` 路径下 `CpGM2GM(..., reduceOp_)` 行为是否符合预期。
- 多 slice 场景下 `curTag_` 与已有 flag offset 是否冲突。
- 小数据、大数据、rankSize 非 2 幂场景正确性。

## 8. 后续演进

后续计划：
- 补充少核/控核版本。
- 根据验证结果决定 BigData 是否恢复独立分支或迁移到 LocalTree 分块流水。
- 若需要进一步降低 scratch，占用可设计更细粒度 handshake 或 ping-pong staging，但第一版不做该复杂化。
