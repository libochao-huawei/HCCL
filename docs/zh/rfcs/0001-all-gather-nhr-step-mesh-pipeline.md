# AllGather NHR Step Mesh Pipeline

## 背景

现有 `InsAllGatherParallelMesh1DNHR` 将 AllGather 数据拆成两段：

- `part0 = 1 - multipleDimensionSplitRatio`：先执行 Server 内 `mesh`，再执行 Server 间 `nhr`。
- `part1 = multipleDimensionSplitRatio`：先执行 Server 间 `nhr`，再执行 Server 内 `mesh`。

原方案以阶段为同步边界，`part1` 必须等待完整 NHR 完成后才进入 Server 内 mesh。新方案在 `part1` 的 NHR 每完成一个 step 后，立即将本 step 新收到的 node 数据送入 Server 内 mesh，以减少 Server 内链路等待时间。

数据比例仍由 `param.opConfig.multipleDimensionSplitRatio` 配置。代码和文档不固化特定比例，实际最优比例由测试决定。

## 整体流程

新增 AICPU/Ins executor：

```cpp
InsAllGatherPipelinedMesh1DNHR
```

该 executor 只注册 AICPU/Ins 模板，不接入 CCU fast launch。CCU 路径继续使用现有 executor。

单 loop 内的逻辑顺序如下：

```cpp
PreSyncIntraAndInter();

// Inter lane: part1 NHR does not wait for part0 mesh.
PrepareInterNHRPart1(dataOffset1, currCountPart1);

// Intra lane: these two mesh tasks are queued on the intra template.
RunIntraMeshPart0(dataOffset0, currCountPart0);
RunIntraMeshPart1LocalNode(dataOffset1, currCountPart1);

for (u32 step = 0; step < nSteps; ++step) {
    AicpuNHRStepInfo stepInfo;
    RunInterNHRStepPart1(step, stepInfo);

    for (u32 nodeIdx : stepInfo.rxSliceIdxs) {
        WaitPreviousIntraIfNeeded();
        RunIntraMeshPart1FromInterScratch(nodeIdx, dataOffset1, currCountPart1);
    }
}
WaitPreviousIntraIfNeeded();
FinalizeInterNHRPart1(copyToOutput = false);

RunInterNHRPart0(dataOffset0, currCountPart0);
```

### 执行流水示意

下面的图强调两条链路的依赖关系。`part1` 的 NHR step 每收到一批 node 数据后，会触发对应 node 的 Server 内 mesh；实现中允许保留一个 pending 的 Server 内 mesh，在下一次需要复用 intra template 前再等待，使上一批 streamed mesh 与下一次 NHR step 有机会重叠。

```mermaid
sequenceDiagram
    autonumber
    participant Main as control thread
    participant Intra as intra mesh template
    participant Inter as inter NHR template
    participant CCL as CCL buffer
    participant Out as output

    par inter lane
        Main->>Inter: PrepareStepRun(part1)
        Inter->>CCL: copy local part1 into inter scratch
    and intra lane
        Main->>Intra: queue part0 local mesh
        Main->>Intra: queue part1 local-node mesh
        Intra->>Out: write part0 / local part1 node blocks
    end

    loop each NHR step
        Main->>Inter: RunNHRStep(step)
        Inter->>CCL: receive rxSliceIdx node blocks
        Main->>Intra: wait previous intra work if template is still pending
        Main->>Intra: mesh rxSliceIdx blocks from inter scratch
        Intra->>Out: write streamed part1 node blocks
        Note over Intra,Inter: previous mesh may overlap next NHR step until intra template is reused
    end

    Main->>Inter: FinalizeStepRun(copyToOutput=false)
    Main->>Inter: part0 inter NHR
    Inter->>Out: complete part0 remote node blocks
```

从 lane 的角度看，真实流水线不是所有任务从同一个起点并排展开，而是：

- `part1 NHR` 不等待 `part0 mesh`，两条 lane 在第一阶段同时启动。
- `part0 mesh` 和 `part1 local mesh` 共享 intra template，二者在 intra lane 上顺序执行。
- `part1 NHR step0` 完成后，`step0 rx` 对应的 streamed mesh 等 intra lane 可复用后启动。
- `part1 NHR step1` 可以和 `step0 rx mesh` 重叠。
- `part1 NHR step2` 可以和 `step1 rx mesh` 重叠。
- 每次需要复用 intra template 前，等待上一批 pending mesh 完成。
- 所有 `part1` streamed mesh 完成后，才进入 `part0 NHR`。

```mermaid
gantt
    title AllGather Pipelined Mesh/NHR Schedule (single loop)
    dateFormat YYYY-MM-DD HH:mm
    axisFormat %H:%M

    section Intra mesh lane
    part0 local mesh                         :i0, 2026-01-01 00:00, 2m
    part1 local-node mesh                    :i1, 2026-01-01 00:02, 1m
    mesh rxSliceIdxs(step0)                  :m0, 2026-01-01 00:03, 1m
    mesh rxSliceIdxs(step1)                  :m1, 2026-01-01 00:04, 1m
    mesh rxSliceIdxs(step2) drain            :m2, 2026-01-01 00:05, 1m

    section Inter NHR lane
    PrepareStepRun(part1)                    :p0, 2026-01-01 00:00, 1m
    part1 NHR step0                          :n0, 2026-01-01 00:01, 1m
    part1 NHR step1                          :n1, 2026-01-01 00:02, 2m
    part1 NHR step2                          :n2, 2026-01-01 00:04, 1m
    FinalizeStepRun(copyToOutput=false)      :f0, 2026-01-01 00:06, 1m
    part0 NHR                                :n3, 2026-01-01 00:07, 2m

    section Data readiness
    rxSliceIdxs(step0) ready in inter scratch :milestone, r0, 2026-01-01 00:02, 0m
    rxSliceIdxs(step1) ready in inter scratch :milestone, r1, 2026-01-01 00:04, 0m
    rxSliceIdxs(step2) ready in inter scratch :milestone, r2, 2026-01-01 00:05, 0m
    part1 streamed mesh drained               :milestone, d0, 2026-01-01 00:06, 0m
```

图中以 3 个 NHR step 为例，横轴的 `00:00` 到 `00:09` 仅对应逻辑时间 `T0` 到 `T9`，用于表达依赖和重叠窗口，不代表固定耗时比例。真实流水线是：

- `part1 PrepareStepRun` 和 `part1 NHR step` 不依赖 `part0 mesh`，可以与 `part0 mesh`、`part1 local mesh` 同时在 inter lane 上推进。
- `part0 mesh` 和 `part1 local mesh` 仍在 intra lane 上顺序执行，因为它们共享同一个 intra template。
- `NHR step0` 结束时，`rxSliceIdxs(step0)` 已经落在 inter scratch；对应的 `mesh rx0` 起点是 `max(NHR step0 end, part0 mesh + part1 local mesh end)`。
- `mesh rx0` 与 `NHR step1` 重叠，`mesh rx1` 与 `NHR step2` 重叠。
- 每次要复用 intra template 启动下一批 streamed mesh 前，先等待上一批 pending mesh 完成；如果 mesh 比下一次 NHR step 更慢，流水线会在该边界处回压。
- `mesh rx2` 是最后一批，没有后续 NHR step 可重叠，因此 drain 完成后才执行 `FinalizeStepRun(copyToOutput=false)` 和 `part0 NHR`。

## NHR Step 接口

`InsTempAllGatherNHR` 增加 step 级接口：

```cpp
HcclResult PrepareStepRun(const OpParam &param,
                          const TemplateDataParams &tempAlgParams,
                          TemplateResource &templateResource);

HcclResult RunNHRStep(const std::vector<ThreadHandle> &threads,
                      const std::map<u32, std::vector<ChannelInfo>> &channels,
                      const u32 channelIdx,
                      const u32 step,
                      AicpuNHRStepInfo &stepInfo);

HcclResult FinalizeStepRun(const std::vector<ThreadHandle> &threads,
                           const u32 channelIdx,
                           bool copyToOutput);
```

旧 `KernelRun` 保持可用，旧 executor 不依赖新调度。兼容路径仍等价于：

```cpp
PrepareStepRun(...);
for (u32 channelIdx = 0; channelIdx < channelsPerRank_; ++channelIdx) {
    for (u32 step = 0; step < GetNHRStepNum(templateRankSize_); ++step) {
        RunNHRStep(..., channelIdx, step, stepInfo);
    }
    FinalizeStepRun(..., channelIdx, true);
}
```

## CCL Buffer 排布

CCL/HCCL 临时 buffer 的总体分区沿用 parallel executor：

```cpp
intraScratchOffset = 0;
interScratchOffset = scratchMultipleIntra * scratchMemBlockSize;
```

总体布局：

```mermaid
flowchart LR
    subgraph CCL["resCtx.cclMem / CCL buffer"]
        direction LR
        I0["intra scratch region<br/>offset = 0<br/>size = scratchMultipleIntra * scratchMemBlockSize"]
        I1["inter scratch region<br/>offset = interScratchOffset<br/>size = scratchMultipleInter * scratchMemBlockSize"]
    end
```

`part1` 的 inter scratch 由 NHR 使用，按 node 切 slot。NHR 每个 step 收到的数据保留在对应 `rxSliceIdx` slot：

```mermaid
flowchart LR
    subgraph Inter["inter scratch for part1 NHR"]
        direction LR
        N0["node 0 slot<br/>interScratchOffset + 0 * part1SliceSize"]
        N1["node 1 slot<br/>interScratchOffset + 1 * part1SliceSize"]
        N2["node 2 slot<br/>interScratchOffset + 2 * part1SliceSize"]
        NX["..."]
        Nk["node k slot<br/>interScratchOffset + k * part1SliceSize"]
    end

    Step["RunNHRStep<br/>rxSliceIdxs"] --> N1
    Step --> N2
    N1 --> Mesh["streamed intra mesh"]
    N2 --> Mesh
```

`part1` 的 intra scratch 由 Mesh 使用，每个 streamed node 使用独立 mesh scratch slot。slot stride 使用 Mesh 模板现有规则：

```cpp
nodeIntraScratchBase =
    intraScratchOffset + nodeIdx * currCountPart1 * dataTypeSize * rankSizeLevel0;
```

对应布局：

```mermaid
flowchart LR
    subgraph Intra["intra scratch for streamed mesh"]
        direction LR
        M0["node 0 mesh slot<br/>0 * part1SliceSize * rankSizeLevel0"]
        M1["node 1 mesh slot<br/>1 * part1SliceSize * rankSizeLevel0"]
        M2["node 2 mesh slot<br/>2 * part1SliceSize * rankSizeLevel0"]
        MK["node k mesh slot<br/>k * part1SliceSize * rankSizeLevel0"]
    end

    S0["inter node k slot"] --> P["GenTemplateAlgParamsIntra1FromInterScratch"]
    P --> MK
    MK --> O["output node k ranks<br/>nodeIdx * rankSizeLevel0 * dataSize + dataOffset1"]
```

不使用 `scratchMemBlockSize` 作为 streamed mesh 的 node stride，避免与 `InsTempAllGatherMesh1D` 的 `scratchRepeatStride = sliceSize * templateRankSize` 不一致。

`part1` 的 NHR 不再执行全量 post-copy 到 output；收到的数据由后续 streamed mesh 从 inter scratch 读取，并写入最终 output。

## 回退方式

现有 `InsAllGatherParallelMesh1DNHR` 保留注册和实现。若需要回退，可在 selector 中将 `InsAllGatherPipelinedMesh1DNHR` 改回原注册名，或通过显式算法选择使用旧 executor。

## 验证清单

- 新 executor 只接入 AICPU/Ins，不影响 CCU。
- CCL/HCCL temp buffer 总分区沿用 parallel executor，无全局重排。
- `part1` inter scratch 中的 `rxSliceIdx` 数据能被 streamed mesh 正确读取。
- streamed mesh 的 intra scratch slot 按 `sliceSize * rankSizeLevel0` 排布。
- `part1` 不依赖 NHR 全量 post-copy。
- `part1` 本节点数据也执行节点内 mesh。
- `part0` 仍按 `mesh -> nhr` 执行。
- `multipleDimensionSplitRatio` 语义保持不变：表示 `part1` 比例。
- selector 只替换目标 AICPU 分支。
- 不硬编码任何测试推导比例。
