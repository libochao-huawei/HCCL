# SD: ReduceScatter 三级序列执行器 (3-Level Sequence Executor) 软件设计文档

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | HCCL ReduceScatter 三级序列执行器 |
| 文档版本 | V1.0 |
| 对应 PR | MR #1158 |
| 编写日期 | 2026-06-01 |

## 2. 设计概述

### 2.1 设计目标

在 HCCL ReduceScatter 算子中实现三级序列执行器，支持框内(Mesh1D) → Pod内(NHR) → 跨Pod(NHR) 的分层串行规约，充分利用各级链路带宽。

### 2.2 设计原则

- **扩展而非修改**: 扩展现有 TopoMatchMultilevel 和注册宏，不修改已有二级执行器
- **通用编排层**: 三级执行器不绑定特定引擎，AICPU/CCU 均可复用
- **与二级模式一致**: Loop 分片、资源计算、模板参数生成等模式与二级序列执行器保持一致

## 3. 系统架构

### 3.1 模块关系

```
┌─────────────────────────────────────────────────────────┐
│                  ReduceScatterAutoSelector               │
│  (算法选择: topoLevelNums >= 3 → InsReduceScatter...NHRNHR) │
└───────────────────────┬─────────────────────────────────┘
                        │ 选择
                        ▼
┌─────────────────────────────────────────────────────────┐
│       InsV2ReduceScatterSequenceExecutor3Level          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Level0     │  │  Level1     │  │  Level2     │    │
│  │  Mesh1D     │→ │  NHR        │→ │  NHR        │    │
│  │  Template   │  │  Template   │  │  Template   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└───────────────────────┬─────────────────────────────────┘
                        │ 依赖
                        ▼
┌─────────────────────────────────────────────────────────┐
│              TopoMatchMultilevel (扩展)                  │
│  MatchTopo → TopoForLayer0 + TopoForLayer1 + TopoForLayer2 │
│  输出: algHierarchyInfo.infos.size() == 3               │
└─────────────────────────────────────────────────────────┘
                        │ 依赖
                        ▼
┌─────────────────────────────────────────────────────────┐
│         REGISTER_EXECUTOR_BY_THREE_TEMPS (新增宏)        │
│  注册: InsReduceScatterSequenceMesh1DNHRNHR             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
INPUT Buffer
    │
    ▼
┌──────────────────────┐
│ Level0: Mesh1D       │  repeatNum = rankSizeLevel1 × rankSizeLevel2
│ INPUT → CCL Buffer   │  inputSliceStride = dataSize_
└──────────┬───────────┘  outputSliceStride = currDataCount × dataTypeSize_
           │
           ▼
      CCL Buffer (框内规约结果)
           │
           ▼
┌──────────────────────┐
│ Level1: NHR          │  repeatNum = rankSizeLevel2
│ CCL Buffer → CCL Buffer │ inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_
└──────────┬───────────┘  inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_
           │
           ▼
      CCL Buffer (Pod内规约结果)
           │
           ▼
┌──────────────────────┐
│ Level2: NHR          │  repeatNum = 1
│ CCL Buffer → OUTPUT  │  inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_
└──────────┬───────────┘  outBuffBaseOff = processedDataCount × dataTypeSize_
           │
           ▼
      OUTPUT Buffer (最终规约结果)
```

## 4. 详细设计

### 4.1 类设计: InsV2ReduceScatterSequenceExecutor3Level

#### 4.1.1 类定义

```cpp
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
class InsV2ReduceScatterSequenceExecutor3Level : public InsCollAlgBase {
public:
    explicit InsV2ReduceScatterSequenceExecutor3Level();
    ~InsV2ReduceScatterSequenceExecutor3Level() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    void GenIntraTemplateParams(TemplateDataParams &tempAlgParamsIntra, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    void GenInterTemplateParams1(TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    void GenInterTemplateParams2(TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempReousrce) const;

    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};
    uint32_t rankSizeLevel2_{0};

    uint32_t rankIdxLevel0_{0};
    uint32_t rankIdxLevel1_{0};
    uint32_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
};
```

#### 4.1.2 模板参数说明

| 模板参数 | 实际类型 | 用途 |
|----------|---------|------|
| AlgTopoMatch | TopoMatchMultilevel | 拓扑匹配策略 |
| InsAlgTemplate0 | InsTempReduceScatterMesh1DZAxisDetour | Level0 Mesh1D 算法模板 |
| InsAlgTemplate1 | InsTempReduceScatterNHR | Level1 NHR 算法模板 |
| InsAlgTemplate2 | InsTempReduceScatterNHR | Level2 NHR 算法模板 |

#### 4.1.3 成员变量说明

| 成员变量 | 类型 | 用途 |
|----------|------|------|
| rankSizeLevel0_ | uint32_t | Level0 (框内) rank 数量 |
| rankSizeLevel1_ | uint32_t | Level1 (Pod内) rank 数量 |
| rankSizeLevel2_ | uint32_t | Level2 (跨Pod) rank 数量 |
| rankIdxLevel0_ | uint32_t | 当前 rank 在 Level0 中的索引 |
| rankIdxLevel1_ | uint32_t | 当前 rank 在 Level1 中的索引 |
| rankIdxLevel2_ | uint32_t | 当前 rank 在 Level2 中的索引 |
| algHierarchyInfo_ | AlgHierarchyInfoForAllLevel | 三级算法层级信息 |
| remoteRankToChannelInfo_ | vector<map<u32, vector<ChannelInfo>>> | 各级 channel 映射 |
| threads_ | vector<ThreadHandle> | 线程句柄 |

### 4.2 核心方法设计

#### 4.2.1 CalcAlgHierarchyInfo

```
CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo)
├── myRank_ = topoInfo->userRank
├── rankSize_ = topoInfo->userRankSize
└── topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo)
    ├── TopoForLayer0(...) → algHierarchyInfo.infos[0]
    ├── TopoForLayer1(...) → algHierarchyInfo.infos[1]
    └── TopoForLayer2(...) → algHierarchyInfo.infos[2]  // 新增
```

**关键逻辑**: 调用 `TopoMatchMultilevel::MatchTopo` 生成三级 algHierarchyInfo，验证 `infos.size() == 3`。

#### 4.2.2 CalcRes

```
CalcRes(comm, param, topoInfo, algHierarchyInfo, resourceRequest)
├── InitCommInfo(param, topoInfo, algHierarchyInfo)
├── 验证 algHierarchyInfo.infos.size() == 3
├── 创建三级临时算法模板
│   ├── tempAlgLevel0 = make_shared<InsAlgTemplate0>(param, myRank_, infos[0])
│   ├── tempAlgLevel1 = make_shared<InsAlgTemplate1>(param, myRank_, infos[1])
│   └── tempAlgLevel2 = make_shared<InsAlgTemplate2>(param, myRank_, infos[2])
├── 分别计算三级资源请求
│   ├── tempAlgLevel0->CalcRes → resReq0
│   ├── tempAlgLevel1->CalcRes → resReq1
│   └── tempAlgLevel2->CalcRes → resReq2
├── 合并资源请求
│   ├── slaveThreadNum = max(resReq0, resReq1, resReq2)
│   ├── notifyNumPerThread = 逐线程取 max
│   ├── notifyNumOnMainThread = max(resReq0, resReq1, resReq2)
│   └── channels[0/1/2] = 各级 channels[0]
└── 计算 CCL Buffer 需求
```

#### 4.2.3 Orchestrate

```
Orchestrate(param, resCtx)
├── InitCommInfo(param, topoInfo, algHierarchyInfo)
│   ├── 初始化 myRank_, rankSize_, reduceOp_, dataType_, dataCount_, dataTypeSize_
│   ├── 计算 rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size()
│   ├── 计算 rankIdxLevel1_ = (myRank_ / rankSizeLevel0_) % rankSizeLevel1_
│   └── 计算 rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ × rankSizeLevel1_)
├── RestoreChannelMap(resCtx, remoteRankToChannelInfo_)
└── OrchestrateLoop(param, resCtx)
```

#### 4.2.4 OrchestrateLoop

```
OrchestrateLoop(param, resCtx)
├── 初始化三级 TemplateDataParams
│   ├── tempAlgParamsLevel0: INPUT → HCCL_BUFFER
│   ├── tempAlgParamsLevel1: HCCL_BUFFER → HCCL_BUFFER
│   └── tempAlgParamsLevel2: HCCL_BUFFER → OUTPUT
├── 创建三级算法模板实例
│   ├── algTemplateLevel0(InsAlgTemplate0)
│   ├── algTemplateLevel1(InsAlgTemplate1)
│   └── algTemplateLevel2(InsAlgTemplate2)
├── 计算 scratchMultiple
│   └── templateScratchMultiplier = m0 × m1 × m2
├── 生成三级 TemplateResource
│   ├── templateResource0 (channelLevelIdx=0)
│   ├── templateResource1 (channelLevelIdx=1)
│   └── templateResource2 (channelLevelIdx=2)
├── 计算 loopTimes
│   └── maxCountPerLoop = cclMem.size / templateScratchMultiplier / align × align / dataTypeSize_
└── for loop = 0 to loopTimes-1
    ├── currDataCount = (最后一轮) ? dataCount_ - processedDataCount : maxCountPerLoop
    ├── GenIntraTemplateParams(level0, processedDataCount, currDataCount, loop)
    ├── algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0)
    ├── GenInterTemplateParams1(level1, processedDataCount, currDataCount, loop)
    ├── algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1)
    ├── GenInterTemplateParams2(level2, processedDataCount, currDataCount, loop)
    ├── algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2)
    └── processedDataCount += currDataCount
```

### 4.3 模板参数生成设计

#### 4.3.1 GenIntraTemplateParams (Level0)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Level0: Mesh1D 框内规约                                              │
│                                                                      │
│ 输入: INPUT Buffer (全量数据)                                        │
│ 输出: CCL Buffer (框内规约后数据)                                    │
│                                                                      │
│ 参数:                                                                │
│   count = currDataCount                                              │
│   inBuffBaseOff = processedDataCount × dataTypeSize_                 │
│   outBuffBaseOff = 0                                                 │
│   hcclBuffBaseOff = 0                                                │
│   sliceSize = currDataCount × dataTypeSize_                          │
│   inputSliceStride = dataSize_                                       │
│   outputSliceStride = currDataCount × dataTypeSize_                  │
│   repeatNum = rankSizeLevel1 × rankSizeLevel2                        │
│   inputRepeatStride = rankSizeLevel0 × dataSize_                     │
│   outputRepeatStride = rankSizeLevel0 × currDataCount × dataTypeSize_│
│                                                                      │
│ 说明: 框内每个 rank 需要为所有跨框和跨Pod组重复执行规约               │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.3.2 GenInterTemplateParams1 (Level1)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Level1: NHR Pod内规约                                                │
│                                                                      │
│ 输入: CCL Buffer (框内规约后数据)                                    │
│ 输出: CCL Buffer (Pod内规约后数据)                                   │
│                                                                      │
│ 参数:                                                                │
│   count = currDataCount                                              │
│   inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_      │
│   outBuffBaseOff = 0                                                 │
│   hcclBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_    │
│   sliceSize = currDataCount × dataTypeSize_                          │
│   inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_  │
│   outputSliceStride = 0                                              │
│   repeatNum = rankSizeLevel2                                         │
│   inputRepeatStride = rankSizeLevel0 × rankSizeLevel1                │
│                       × currDataCount × dataTypeSize_                │
│   outputRepeatStride = 0                                             │
│                                                                      │
│ 说明: Pod内每个 rank 需要为每个跨Pod组重复执行规约                   │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.3.3 GenInterTemplateParams2 (Level2)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Level2: NHR 跨Pod规约                                                │
│                                                                      │
│ 输入: CCL Buffer (Pod内规约后数据)                                   │
│ 输出: OUTPUT Buffer (最终规约结果)                                   │
│                                                                      │
│ 参数:                                                                │
│   count = currDataCount                                              │
│   inBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_      │
│   outBuffBaseOff = processedDataCount × dataTypeSize_                │
│   hcclBuffBaseOff = rankIdxLevel0 × currDataCount × dataTypeSize_    │
│   sliceSize = currDataCount × dataTypeSize_                          │
│   inputSliceStride = rankSizeLevel0 × currDataCount × dataTypeSize_  │
│   outputSliceStride = 0                                              │
│   repeatNum = 1                                                      │
│                                                                      │
│ 说明: 跨Pod规约不需要重复，直接输出到最终位置                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.4 拓扑匹配扩展设计

#### 4.4.1 TopoForLayer2 方法

```
TopoForLayer2(comm, netLayer, layer0Size, layer1Size, myRank, algHierarchyInfo)
├── 获取指定网络层的拓扑实例
│   └── HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum)
├── 验证 topoInstNum == 1
├── 获取该拓扑实例下的所有 rank
│   └── HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInsts[0], &ranks, &rankNum)
└── 筛选同组 rank
    ├── myRank → 直接加入
    └── 其他 rank:
        ├── 条件1: rankId % (layer0Size × layer1Size) == myRank % (layer0Size × layer1Size)
        │   (即 Level0 和 Level1 索引相同)
        ├── 条件2: HcclRankGraphGetLinks(comm, netLayer, myRank, rankId, ...) linkNum > 0
        │   (即存在链路连接)
        └── 满足两个条件 → 加入 rankVecLayer2WithSameIdx
```

#### 4.4.2 MatchTopo 扩展

```
MatchTopo(comm, topoInfo, algHierarchyInfo)
├── 验证 topoLevelNums ∈ [1, COMM_LAYER_SIZE_3]   // 从 COMM_LAYER_SIZE_2 扩展
├── ... (Level0 逻辑不变)
├── TopoForLayer1(comm, netLayer, layer0Size, myRank, algHierarchyInfo)
├── layer1Size = algHierarchyInfo.infos[1][0].size()
└── if (topoLevelNums >= COMM_LAYER_SIZE_3)
    └── TopoForLayer2(comm, netLayer2, layer0Size, layer1Size, myRank, algHierarchyInfo)
```

**关键变更**:
- `COMM_LAYER_SIZE_2` → `COMM_LAYER_SIZE_3` 作为最大层级校验
- `algHierarchyInfo.infos.resize()` 根据 `topoLevelNums` 动态决定为 2 或 3
- 新增 Level2 拓扑计算分支

### 4.5 注册宏设计

#### 4.5.1 REGISTER_EXECUTOR_BY_THREE_TEMPS

```cpp
#define REGISTER_EXECUTOR_BY_THREE_TEMPS_HELPER(ctr, type, name, insCollAlgBase, \
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2)              \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register( \
        type, std::string(#name), \
        DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch, InsAlgTemplate0, \
            InsAlgTemplate1, InsAlgTemplate2>>)

#define REGISTER_EXECUTOR_BY_THREE_TEMPS_HELPER_1(ctr, type, name, insCollAlgBase, \
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2)               \
    REGISTER_EXECUTOR_BY_THREE_TEMPS_HELPER(ctr, type, name, insCollAlgBase, \
        AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2)

#define REGISTER_EXECUTOR_BY_THREE_TEMPS(type, name, insCollAlgBase, AlgTopoMatch, \
    InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2)                             \
    REGISTER_EXECUTOR_BY_THREE_TEMPS_HELPER_1(__COUNTER__, type, name, insCollAlgBase, \
        AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2)
```

**设计说明**: 与现有 `REGISTER_EXECUTOR_BY_TWO_TEMPS` 模式一致，通过 `__COUNTER__` 解决重复注册问题，使用三层宏展开避免宏参数展开问题。

### 4.6 算法选择器设计

#### 4.6.1 SelectAicpuAlgo 修改

```
SelectAicpuAlgo(topoInfo, dataSize)
├── ...
├── if (localNetInsSizeOfLayer[0] > 1 && level0Topo == MESH_1D)
│   ├── if (topoLevelNums >= 3)                          // 新增分支
│   │   └── return "InsReduceScatterSequenceMesh1DNHRNHR"
│   ├── else if (dataSize > RS_AICPU_1D_MIN_DATA_SIZE)
│   │   └── return "InsReduceScatterSequenceMesh1DNhr" / "InsReduceScatterParallelMesh1DNHR"
│   └── else
│       └── return "InsReduceScatterParallelMesh1DNHR"
└── ...
```

**设计说明**: 三级拓扑优先判断，在数据量判断之前，确保三级拓扑场景统一走三级算法。

## 5. 文件变更清单

| 文件路径 | 变化类型 | 变更内容 |
|----------|---------|---------|
| `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_3level.h` | 新增 | 三级序列执行器头文件，类声明 |
| `src/ops/reduce_scatter/executor/ins_v2_reduce_scatter_sequence_executor_3level.cc` | 新增 | 三级序列执行器实现，含注册调用 |
| `src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.h` | 修改 | 新增 `REGISTER_EXECUTOR_BY_THREE_TEMPS` 宏 |
| `src/ops/op_common/topo/topo_match_multilevel.h` | 修改 | 新增 `TopoForLayer2` 方法声明 |
| `src/ops/op_common/topo/topo_match_multilevel.cc` | 修改 | 实现 `TopoForLayer2`，扩展 `MatchTopo` 支持三级 |
| `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc` | 修改 | 新增三级拓扑算法选择分支 |
| `src/ops/reduce_scatter/executor/CMakeLists.txt` | 修改 | 新增 3level.cc 到编译目标 |
| `src/CMakeLists.txt` | 修改 | 新增 3level.cc 到编译目标 |
| `test/st/algorithm/utils/src/aicpu/CMakeLists.txt` | 修改 | 新增 3level.cc 到测试编译目标 |

## 6. 关键设计决策

### 6.1 线性串行 vs 流水线

**决策**: 选择线性串行三步执行

**理由**:
1. 实现复杂度低，与现有二级模式一致
2. 每级数据量递减（Level0 全量 → Level1 子集 → Level2 更小子集），延迟叠加的实际影响远小于带宽收益
3. 三级流水线同步和 Buffer 管理复杂度过高，作为后续优化方向

### 6.2 扩展 TopoMatchMultilevel vs 新建拓扑类

**决策**: 扩展现有 TopoMatchMultilevel

**理由**:
1. Level0/Level1 的逻辑完全复用
2. 只需新增 `TopoForLayer2` 方法
3. 避免引入新的拓扑匹配类，减少维护成本

### 6.3 CCL Buffer 管理

**决策**: 采用与现有非 AICPU executor 一致的模式

**说明**:
- Level0: INPUT → HCCL_BUFFER
- Level1: HCCL_BUFFER → HCCL_BUFFER
- Level2: HCCL_BUFFER → OUTPUT
- scratchMultiple = multiplier0 × multiplier1 × multiplier2

### 6.4 三级拓扑算法选择策略

**决策**: 三级拓扑统一走三级算法，不区分数据量

**理由**:
1. 三级拓扑场景下，二级合并方案浪费框内和 Pod 内带宽
2. 即使数据量较小，分层规约也能减少跨 Pod 数据传输
3. 简化选择逻辑，避免引入额外的数据量阈值判断

## 7. currDataCount 语义说明

`currDataCount` 在三级执行器中表示每轮循环中每 rank 最终输出的数据量，所有三级共用同一 `currDataCount`。各级间每 rank 数据量递减靠 `repeatNum` 体现而非 `currDataCount` 变化：

| 级别 | 输入数据量/rank | 输出数据量/rank | repeatNum |
|------|----------------|----------------|-----------|
| Level0 | dataSize_ (全量) | dataSize_ / rankSizeLevel0 | rankSizeLevel1 × rankSizeLevel2 |
| Level1 | dataSize_ / rankSizeLevel0 | dataSize_ / (rankSizeLevel0 × rankSizeLevel1) | rankSizeLevel2 |
| Level2 | dataSize_ / (rankSizeLevel0 × rankSizeLevel1) | dataSize_ / (rankSizeLevel0 × rankSizeLevel1 × rankSizeLevel2) = currDataCount | 1 |

## 8. 错误处理

| 错误场景 | 处理方式 |
|----------|---------|
| algHierarchyInfo.infos.size() != 3 | 返回 HCCL_E_INTERNAL，打印错误日志 |
| topoLevelNums > COMM_LAYER_SIZE_3 | 返回 HCCL_E_INTERNAL，打印错误日志 |
| TopoForLayer2 topoInstNum != 1 | 返回 HCCL_E_PARA，打印错误日志 |
| channelLevelIdx >= remoteRankToChannelInfo_.size() | 返回 HCCL_E_INTERNAL，打印错误日志 |
| KernelRun 返回非 HCCL_SUCCESS | 打印错误日志，向上传播错误码 |

## 9. 与二级执行器的对比

| 维度 | 二级序列执行器 | 三级序列执行器 |
|------|--------------|--------------|
| 模板参数 | 2 个 (Template0, Template1) | 3 个 (Template0, Template1, Template2) |
| LEVEL_NUM | 2 | 3 |
| 数据流 | INPUT → Mesh → CCL → NHR → OUTPUT | INPUT → Mesh → CCL → NHR → CCL → NHR → OUTPUT |
| scratchMultiple | m0 × m1 | m0 × m1 × m2 |
| channels | channels[0/1] | channels[0/1/2] |
| GenXxxParams | 2 个 | 3 个 |
| 注册宏 | REGISTER_EXECUTOR_BY_TWO_TEMPS | REGISTER_EXECUTOR_BY_THREE_TEMPS |
| 拓扑层级 | infos.size() == 2 | infos.size() == 3 |
