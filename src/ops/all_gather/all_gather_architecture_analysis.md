# All_Gather 算子架构深度分析报告

## 一、目录结构与组织方式

### 1.1 整体目录结构

```
src/ops/all_gather/
├── all_gather_op.h/cc           # 顶层操作入口
├── selector/                     # 算法选择器模块
│   └── all_gather_auto_selector.h/cc
├── executor/                     # 执行器模块
│   ├── ins_v2_all_gather_parallel_executor.h/cc    # 并行执行器
│   ├── ins_v2_all_gather_concurrent_executor.h/cc  # 并发执行器
│   ├── ins_v2_all_gather_sequence_executor.h/cc    # 序列执行器
│   └── ins_v2_all_gather_sole_executor.h/cc        # 单一执行器
└── template/                     # 算法模板模块
    ├── aicpu/                    # AICPU 模板
    │   ├── ins_temp_all_gather_mesh_1D.h/cc
    │   ├── ins_temp_all_gather_nhr.h/cc
    │   └── ins_temp_all_gather_nhr_dpu.h/cc
    ├── aiv/                      # AIV 模板
    │   ├── aiv_temp_all_gather_mesh_1D.h/cc
    │   └── kernel/
    │       └── aiv_all_gather_mesh_1d.h
    └── ccu/                      # CCU 模板
        ├── ccu_temp_all_gather_mesh_1D.h/cc
        ├── ccu_temp_all_gather_mesh_1D_mem2mem.h/cc
        ├── ccu_temp_all_gather_nhr_1D_mem2mem.h/cc
        ├── ccu_temp_all_gather_2dies_mesh_1D.h/cc
        └── kernel/               # CCU Kernel 实现
            ├── ccu_kernel_all_gather_mesh1d.h/cc
            ├── ccu_kernel_all_gather_mesh1d_mem2mem.h/cc
            └── ...
```

### 1.2 模块职责划分

| 模块 | 职责 | 关键文件 |
|------|------|----------|
| **selector** | 根据拓扑和参数自动选择最优算法 | all_gather_auto_selector.h/cc |
| **executor** | 编排算法执行流程,管理资源 | ins_v2_all_gather_*_executor.h/cc |
| **template** | 实现具体算法逻辑 | ccu/, aiv/, aicpu/ 子目录 |

---

## 二、Selector 模块分析

### 2.1 核心类: AllGatherAutoSelector

**文件位置:** `selector/all_gather_auto_selector.h:18-41`

**职责:** 根据拓扑信息和参数自动选择最优算法

### 2.2 主要选择方法

#### 2.2.1 SelectCcuScheduleAlgo

**位置:** `selector/all_gather_auto_selector.cc:141-174`

**功能:** 选择 CCU Schedule 模式算法

**选择逻辑:**
```cpp
// 伪代码
if (topoLevelNums > 1) {
    // 多层级拓扑
    if (level0Topo == TopoType::TOPO_MESH_1D) {
        if (dataSize > SMALL_COUNT_512KB) {
            return "InsAllGatherParallelMesh1DNHR";
        } else {
            return "InsAllGatherNHR";
        }
    }
} else {
    // 单层级拓扑
    return SelectCcuScheduleLevel0Algo(...);
}
```

#### 2.2.2 SelectAicpuAlgo

**位置:** `selector/all_gather_auto_selector.cc:176-232`

**功能:** 选择 AICPU 模式算法

**支持拓扑:**
- MESH_1D: 单服务器内通信
- MESH_1D_CLOS (UBX): 跨服务器通信
- CLOS: 纯 Clos 拓扑

**UBX 特殊处理逻辑:**
```cpp
// selector/all_gather_auto_selector.cc:204-222
if (dataSize > SMALL_COUNT_512KB) {
    if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= 4)) {
        selectAlgName = "InsAllGatherConcurrentMesh1DNHR";
    } else if (isClosNumMultipleOfMeshNum) {
        selectAlgName = "InsAllGatherParallelMesh1DNHRUBX";
    } else {
        selectAlgName = "InsAllGatherNHRUBX";
    }
} else {
    selectAlgName = "InsAllGatherMesh1DUBX";
}
```

#### 2.2.3 SelectAivAlgo

**位置:** `selector/all_gather_auto_selector.cc:234-245`

**功能:** 选择 AIV 模式算法

**当前支持:** 仅支持 `AivAllGatherMesh1D`

#### 2.2.4 SelectDPUAlgo

**位置:** `selector/all_gather_auto_selector.cc:247-262`

**功能:** 选择 DPU 模式算法(特殊硬件场景)

### 2.3 选择逻辑关键点

| 判断条件 | 说明 |
|---------|------|
| **topoLevelNums** | 拓扑层级数,区分单层级和多层级 |
| **level0Topo** | 第0层拓扑形状(MESH_1D, MESH_1D_CLOS, CLOS等) |
| **dataSize** | 数据量大小,与 SMALL_COUNT_512KB 比较选择不同策略 |
| **isMeshNumEqualToClosNum** | Mesh 数量是否等于 Clos 数量(UBX 特殊处理) |
| **isClosNumMultipleOfMeshNum** | Clos 数量是否是 Mesh 数量的倍数 |

### 2.4 注册机制

**位置:** `selector/all_gather_auto_selector.cc:264`

```cpp
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER, 18, AllGatherAutoSelector);
```

---

## 三、Executor 模块分析

### 3.1 执行器类型对比

| 执行器类型 | 用途 | 特点 | 关键文件 |
|-----------|------|------|---------|
| **SoleExecutor** | 单模板执行 | 最简单,直接调用一个模板 | ins_v2_all_gather_sole_executor.cc |
| **ParallelExecutor** | 并行执行 | 两个模板并行执行,数据分割 | ins_v2_all_gather_parallel_executor.cc |
| **ConcurrentExecutor** | 并发执行 | 优化小规模场景(≤4P) | ins_v2_all_gather_concurrent_executor.cc |
| **SequenceExecutor** | 序列执行 | 两个模板顺序执行 | ins_v2_all_gather_sequence_executor.cc |

### 3.2 InsV2AllGatherSoleExecutor 详细分析

**文件:** `executor/ins_v2_all_gather_sole_executor.cc`

#### 3.2.1 核心流程

```
CalcAlgHierarchyInfo → CalcRes → Orchestrate → OrchestrateLoop
```

#### 3.2.2 CalcAlgHierarchyInfo

**位置:** `ins_v2_all_gather_sole_executor.cc:32-39`

**功能:** 使用拓扑匹配器计算层级信息

```cpp
HcclResult InsV2AllGatherSoleExecutor::CalcAlgHierarchyInfo(...)
{
    TopoMatch1D topoMatch;
    CHK_RET(topoMatch.MatchTopo(param, algHierarchyInfo));
    return HcclResult::HCCL_SUCCESS;
}
```

#### 3.2.3 CalcRes

**位置:** `ins_v2_all_gather_sole_executor.cc:42-61`

**功能:** 构建算法模板实例,计算所需资源

```cpp
HcclResult InsV2AllGatherSoleExecutor::CalcRes(...)
{
    // 创建模板实例
    InsAlgTemplate algTemplate(param, ...);
    
    // 计算所需 scratch 内存
    u32 templateScratchMultiplier = algTemplate.CalcScratchMultiple(...);
    
    // 计算所需线程、通知、通道
    CHK_RET(algTemplate.CalcRes(...));
    
    return HcclResult::HCCL_SUCCESS;
}
```

#### 3.2.4 Orchestrate

**位置:** `ins_v2_all_gather_sole_executor.cc:64-88`

**功能:** 准备执行环境,调用 OrchestrateLoop

#### 3.2.5 OrchestrateLoop

**位置:** `ins_v2_all_gather_sole_executor.cc:91-172`

**核心逻辑:**

```cpp
// 数据分片处理
u64 sliceCount = opCount_ / maxCountPerSlice;
u64 tailCount = opCount_ % maxCountPerSlice;
u64 loopTimes = (tailCount == 0) ? sliceCount : sliceCount + 1;

// 循环执行
for (u64 loop = 0; loop < loopTimes; loop++) {
    // 计算当前数据量
    u64 currDataCount = (loop < sliceCount) ? maxCountPerSlice : tailCount;
    
    // 准备参数
    tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
    
    // 执行 kernel
    CHK_RET(algTemplate.KernelRun(param, tempAlgParams, templateAlgRes));
}
```

#### 3.2.6 注册示例

**位置:** `ins_v2_all_gather_sole_executor.cc:174-212`

```cpp
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, InsAllGatherMesh1D, 
                 InsV2AllGatherSoleExecutor, TopoMatch1D, InsTempAllGatherMesh1D);
```

### 3.3 InsV2AllGatherParallelExecutor 详细分析

**文件:** `executor/ins_v2_all_gather_parallel_executor.cc`

#### 3.3.1 核心特点

- 使用两个模板并行执行
- 数据按比例分割(默认 0.5:0.5)
- 两阶段执行: Intra0+Inter1 → Inter0+Intra1

#### 3.3.2 OrchestrateLoop 核心逻辑

**位置:** `ins_v2_all_gather_parallel_executor.cc:355-448`

```cpp
// 阶段1: 数据0的intra + 数据1的inter
GenTemplateAlgParamsIntra0(...);
tempAlgIntra.KernelRun(...);
GenTemplateAlgParamsInter1(...);
tempAlgInter.KernelRun(...);

// 阶段2: 数据0的inter + 数据1的intra
GenTemplateAlgParamsInter0(...);
tempAlgInter.KernelRun(...);
GenTemplateAlgParamsIntra1(...);
tempAlgIntra.KernelRun(...);
```

---

## 四、Template 模块分析

### 4.1 CCU Template 层

#### 4.1.1 CcuTempAllGatherMesh1D

**文件:** `template/ccu/ccu_temp_all_gather_mesh_1D.cc`

##### CalcRes

**位置:** `ccu_temp_all_gather_mesh_1D.cc:36-80`

**功能:** 创建 CCU Kernel 信息,计算 Mesh1D 通道请求

**关键代码:**

```cpp
// 第50-52行: Kernel 创建器
kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
    return std::make_unique<CcuKernelAllGatherMesh1D>(arg);
};
```

##### KernelRun

**位置:** `ccu_temp_all_gather_mesh_1D.cc:82-114`

**功能:** 准备输入输出地址,创建任务参数,启动 CCU Kernel

**关键代码:**

```cpp
// 第111行: Kernel 启动
CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], 
                             templateResource.ccuKernels[0], taskArgPtr));
```

#### 4.1.2 CcuTempAllGatherMesh1DMem2Mem

**文件:** `template/ccu/ccu_temp_all_gather_mesh_1D_mem2mem.cc`

##### 与普通版本的区别

| 特性 | 普通版本 | Mem2Mem 版本 |
|------|---------|-------------|
| 输入输出地址 | 不同 | 可相等(原地操作) |
| 执行次数 | 单次 | 可重复 |
| 步长设置 | 无 | 支持步长 |
| 尾块处理 | 无 | 支持 |

##### 关键参数

**位置:** `ccu_temp_all_gather_mesh_1D_mem2mem.cc:91-98`

```cpp
uint64_t inputSliceStride   = templateDataParams.inputSliceStride;   // 输入步长
uint64_t outputSliceStride  = templateDataParams.outputSliceStride;  // 输出步长
uint32_t repeatNum          = templateDataParams.repeatNum;          // 重复次数
uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;  // 输入重复步长
uint64_t outputRepeatStride = templateDataParams.outputRepeatStride; // 输出重复步长
uint64_t normalSliceSize    = templateDataParams.sliceSize;          // 正常块大小
uint64_t lastSliceSize      = templateDataParams.tailSize;           // 尾块大小
uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;     // 是否原地操作
```

##### 尾块处理

**位置:** `ccu_temp_all_gather_mesh_1D_mem2mem.cc:99-101`

```cpp
if (templateDataParams.tailSize != 0 && mySubCommRank_ == templateRankSize_ - 1) {
    normalSliceSize = templateDataParams.tailSize;
}
```

最后一个 rank 使用尾块大小,其他 rank 使用正常块大小。

#### 4.1.3 CcuTempAllGather2DiesMesh1D

**文件:** `template/ccu/ccu_temp_all_gather_2dies_mesh_1D.cc`

##### 双Die特殊处理

**特点:**
- 创建两个 Kernel
- 按 Die ID 分组通道
- 双流并行执行

##### 按 Die 分组通道

**位置:** `ccu_temp_all_gather_2dies_mesh_1D.cc:61-70`

```cpp
for (u32 j = 0; j < channelDescs.size(); j++) {
    CHK_RET(GetChannelDieId(comm, rankId, channelDescs[j], tmpDieId));
    if (tmpDieId == 0) {
        kernelInfo0.channels.push_back(channelDescs[j]);
    } else {
        kernelInfo1.channels.push_back(channelDescs[j]);
    }
}
```

##### 双流并行执行

**位置:** `ccu_temp_all_gather_2dies_mesh_1D.cc:120-126`

```cpp
// 创建两个 Kernel
ccuKernelNum.push_back(ALL_GATHER_DIE_NUM);  // 第41行

// 双流并行
CHK_RET(HcclCcuKernelLaunch(..., threads[0], ccuKernels[0], ...));
CHK_RET(HcclCcuKernelLaunch(..., threads[1], ccuKernels[1], ...));
```

### 4.2 CCU Kernel 层

#### 4.2.1 CcuKernelAllGatherMesh1D

**文件:** `template/ccu/kernel/ccu_kernel_all_gather_mesh1d.cc`

##### 核心算法流程

**位置:** `ccu_kernel_all_gather_mesh1d.cc:127-144`

```cpp
HcclResult CcuKernelAllGatherMesh1D::Algorithm()
{
    CHK_RET(InitResource());    // 初始化资源
    LoadArgs();                 // 加载参数
    PreSync();                  // 前同步
    DoAllGather();              // 执行 AllGather
    PostSync();                 // 后同步
    return HcclResult::HCCL_SUCCESS;
}
```

##### DoAllGather 实现

**位置:** `ccu_kernel_all_gather_mesh1d.cc:107-125`

**功能:** 使用 GroupBroadcast 原语执行 AllGather

```cpp
HcclResult CcuKernelAllGatherMesh1D::DoAllGather()
{
    // 设置源地址和目标地址
    CcuRep::Variable srcAddr = input_[myRank_];
    CcuRep::Variable dstAddr = output_[myRank_];
    
    // 通过通道进行广播
    CHK_RET(GroupBroadcast(srcAddr, dstAddr, sliceSize_, channels_));
    
    return HcclResult::HCCL_SUCCESS;
}
```

##### 关键数据结构

```cpp
std::vector<CcuRep::Variable> input_;   // 输入地址数组
std::vector<CcuRep::Variable> output_;  // 输出地址数组
std::vector<CcuRep::Variable> token_;   // 同步令牌
CcuRep::Variable offset_;               // 偏移量
CcuRep::Variable sliceSize_;            // 切片大小
```

### 4.3 AICPU Template 层

#### 4.3.1 InsTempAllGatherMesh1D

**文件:** `template/aicpu/ins_temp_all_gather_mesh_1D.cc`

##### 核心流程

**位置:** `ins_temp_all_gather_mesh_1D.cc:57-92`

```cpp
HcclResult InsTempAllGatherMesh1D::KernelRun(...)
{
    CHK_RET(LocalDataCopy(threads));           // 本地拷贝
    CHK_RET(RunAllGatherMesh(threads, channels)); // Mesh AllGather
    if (opMode_ == OpMode::OPBASE) {
        CHK_RET(PostLocalCopy(threads));       // 后拷贝
    }
    return HcclResult::HCCL_SUCCESS;
}
```

##### RunAllGatherMesh 实现

**位置:** `ins_temp_all_gather_mesh_1D.cc:94-195`

**核心逻辑:**

```cpp
// 遍历所有连接的 rank
for (u32 step = 0; step < connectedRankSize; step++) {
    u32 remoteRank = (myRank_ + step) % rankSize;
    
    // 计算数据大小
    u64 curCount = (remoteRank == rankSize - 1) ? tailCount : normalCount;
    
    // 传输数据
    if (useDmaRead) {
        CHK_RET(SendRecvRead(...));  // DMA Read 模式
    } else {
        CHK_RET(SendRecvWrite(...)); // Send/Write 模式
    }
}
```

##### 关键优化

**DMA Read 模式:** 当输入在 HCCL_BUFFER 时直接读取到输出

**位置:** `ins_temp_all_gather_mesh_1D.cc:104, 184`

```cpp
bool useDmaRead = (inputMemType == HcclMemType::HCCL_BUFFER);
if (useDmaRead) {
    CHK_RET(SendRecvRead(...));
}
```

**尾块处理:** 最后一个 rank 处理尾块大小

**位置:** `ins_temp_all_gather_mesh_1D.cc:106-108`

```cpp
u64 curCount = (remoteRank == rankSize - 1) ? tailCount : normalCount;
```

#### 4.3.2 InsTempAllGatherNHR

**文件:** `template/aicpu/ins_temp_all_gather_nhr.h`

##### 特点

- 用于非均匀拓扑
- 使用环形通信模式
- 支持多步骤执行

### 4.4 AIV Template 层

#### 4.4.1 AivTempAllGatherMesh1D

**文件:** `template/aiv/aiv_temp_all_gather_mesh_1D.cc`

##### KernelRun 实现

**位置:** `aiv_temp_all_gather_mesh_1D.cc:53-107`

**功能:** 准备 AIV 参数结构,调用 ExecuteKernelLaunch 启动设备端 kernel

**关键参数设置:**

```cpp
aivAllGatherArgs.cmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
aivAllGatherArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + ...;
aivAllGatherArgs.output = tempAlgParams.buffInfo.outBuffBaseOff + ...;
aivAllGatherArgs.rank = u32(myRank_);
aivAllGatherArgs.rankSize = tempRankSize_;
```

#### 4.4.2 AivAllGatherMesh1D (设备端 Kernel)

**文件:** `template/aiv/kernel/aiv_all_gather_mesh_1d.h`

##### 核心算法

**位置:** `aiv_all_gather_mesh_1d.h:45-75`

```cpp
__aicore__ inline void Run(uint64_t len, uint64_t stride)
{
    // 步骤1: LocalCopy - 拷贝 input 到 GM_IN
    CpGM2GM(gmIn, input, curCount);
    
    // 步骤2: 设置完成标志
    Record(rank_, GetBlockIdx() * FLAG_SIZE + flagOffset, curTag_);
    
    // 步骤3: 循环读取每个 rank 的数据到 output
    for (uint32_t rank = 0; rank < rankSize_; ++rank) {
        WaitFlag(rank, GetBlockIdx() * FLAG_SIZE + flagOffset, curTag_);
        CpGM2GM(output, gmOthers, curCount);
    }
}
```

##### 核数优化

**位置:** `aiv_all_gather_mesh_1d.h:80-88`

```cpp
// 当 numBlocks_ >= rankSize_ 时,每个核处理一部分数据
// 当 numBlocks_ < rankSize_ 时,使用 RunCtrlCore 模式
if (numBlocks_ >= rankSize_) {
    Run(count, sliceId, stride);
} else {
    RunCtrlCore(count, sliceId, stride);
}
```

---

## 五、不同拓扑结构实现差异

### 5.1 MESH_1D (一维网格)

#### 特点

- 所有 rank 按线性排列
- 每个 rank 与相邻 rank 通信
- 适用于单服务器内通信

#### 实现

| 模式 | 实现类 |
|------|--------|
| AICPU | InsTempAllGatherMesh1D |
| CCU | CcuTempAllGatherMesh1D / CcuTempAllGatherMesh1DMem2Mem |
| AIV | AivTempAllGatherMesh1D |

### 5.2 MESH_1D_CLOS (UBX 拓扑)

#### 特点

- Mesh 与 Clos 混合拓扑
- 支持跨服务器通信
- 需要特殊通道选择策略

#### 选择逻辑

**位置:** `selector/all_gather_auto_selector.cc:204-222`

```cpp
if (dataSize > SMALL_COUNT_512KB) {
    if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= 4)) {
        selectAlgName = "InsAllGatherConcurrentMesh1DNHR";
    } else if (isClosNumMultipleOfMeshNum) {
        selectAlgName = "InsAllGatherParallelMesh1DNHRUBX";
    } else {
        selectAlgName = "InsAllGatherNHRUBX";
    }
} else {
    selectAlgName = "InsAllGatherMesh1DUBX";
}
```

### 5.3 2DIES_MESH_1D (双Die网格)

#### 特点

- 两个 Die 的特殊硬件架构
- 每个 Die 有独立的通道
- 需要双 Kernel 并行执行

#### 实现

**类:** CcuTempAllGather2DiesMesh1D

**关键处理:**

1. 按 Die ID 分组通道 (`ccu_temp_all_gather_2dies_mesh_1D.cc:61-70`)
2. 创建两个 Kernel (`ccu_temp_all_gather_2dies_mesh_1D.cc:41`)
3. 双流并行执行 (`ccu_temp_all_gather_2dies_mesh_1D.cc:120-126`)

### 5.4 NHR (非均匀环形)

#### 特点

- 非均匀拓扑结构
- 使用环形通信模式
- 支持跨服务器通信

#### 实现

| 模式 | 实现类 |
|------|--------|
| AICPU | InsTempAllGatherNHR |
| CCU | CcuTempAllGatherNHR1DMem2Mem |

---

## 六、Mem2Mem 变体特殊处理

### 6.1 与普通版本的区别

| 特性 | 普通版本 | Mem2Mem 版本 |
|------|---------|-------------|
| 输入输出地址 | 不同 | 可相等(原地操作) |
| 执行次数 | 单次 | 可重复 |
| 步长设置 | 无 | 支持步长 |
| 尾块处理 | 无 | 支持 |

### 6.2 关键参数

**位置:** `template/ccu/ccu_temp_all_gather_mesh_1D_mem2mem.cc:91-98`

```cpp
uint64_t inputSliceStride   = templateDataParams.inputSliceStride;   // 输入步长
uint64_t outputSliceStride  = templateDataParams.outputSliceStride;  // 输出步长
uint32_t repeatNum          = templateDataParams.repeatNum;          // 重复次数
uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;  // 输入重复步长
uint64_t outputRepeatStride = templateDataParams.outputRepeatStride; // 输出重复步长
uint64_t normalSliceSize    = templateDataParams.sliceSize;          // 正常块大小
uint64_t lastSliceSize      = templateDataParams.tailSize;           // 尾块大小
uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;     // 是否原地操作
```

### 6.3 尾块处理

**位置:** `template/ccu/ccu_temp_all_gather_mesh_1D_mem2mem.cc:99-101`

```cpp
if (templateDataParams.tailSize != 0 && mySubCommRank_ == templateRankSize_ - 1) {
    normalSliceSize = templateDataParams.tailSize;
}
```

**说明:** 最后一个 rank 使用尾块大小,其他 rank 使用正常块大小。

---

## 七、调用链路追踪

### 7.1 完整调用链

```
用户调用 HcclAllGather
    ↓
AllGatherOutPlace (all_gather_op.cc:213)
    ↓
AllGatherOutPlaceCommon (all_gather_op.cc:150)
    ↓
Selector (all_gather_op.cc:189) → 选择算法名称
    ↓
HcclExecOp (all_gather_op.cc:198)
    ↓
Executor::Orchestrate
    ↓
Executor::OrchestrateLoop
    ↓
Template::KernelRun
    ↓
[分支1: AICPU] SendRecvWrite/SendRecvRead
[分支2: CCU] HcclCcuKernelLaunch → CcuKernel::Algorithm
[分支3: AIV] ExecuteKernelLaunch → AivKernel::Process
```

### 7.2 Selector 选择流程

**文件:** `selector/all_gather_auto_selector.cc`

#### CCU Schedule 选择

**位置:** `all_gather_auto_selector.cc:141-174`

```
SelectCcuScheduleAlgo
    ├─ 判断 topoLevelNums > 1 (多层级)
    │   ├─ 是: 选择 Parallel 或 NHR 算法
    │   └─ 否: 调用 SelectCcuScheduleLevel0Algo
    │       ├─ MESH_1D: 选择 Mesh1D 或 2Dies
    │       └─ MESH_1D_CLOS: 调用 SelectCcuScheduleUBXAlgo
    └─ 返回算法名称
```

#### AICPU 选择

**位置:** `all_gather_auto_selector.cc:176-232`

```
SelectAicpuAlgo
    ├─ 判断 topoLevelNums > 1
    │   ├─ Level1Nhr: InsAllGatherNHR
    │   ├─ Level0Nhr: InsAllGatherNHR
    │   └─ MESH_1D: InsAllGatherParallelMesh1DNHR
    └─ 单层级
        ├─ MESH_1D: InsAllGatherMesh1D
        ├─ MESH_1D_CLOS: UBX 特殊处理
        └─ CLOS: InsAllGatherNHR
```

### 7.3 Template 实例化流程

**SoleExecutor 示例:**

**位置:** `executor/ins_v2_all_gather_sole_executor.cc:126`

```cpp
// 创建模板实例
InsAlgTemplate algTemplate(param, resCtx.topoInfo.userRank, 
                           resCtx.algHierarchyInfo.infos[0]);

// 计算所需 scratch 内存
u32 templateScratchMultiplier = 
    algTemplate.CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType, 
                                    tempAlgParams.buffInfo.outBuffType);

// 循环执行
for (u64 loop = 0; loop < loopTimes; loop++) {
    // 准备参数
    tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
    
    // 执行 kernel
    CHK_RET(algTemplate.KernelRun(param, tempAlgParams, templateAlgRes));
}
```

### 7.4 Kernel 执行流程

#### CCU Kernel 示例

**位置:** `template/ccu/kernel/ccu_kernel_all_gather_mesh1d.cc:127-144`

```
Algorithm()
    ├─ InitResource()      // 创建变量,初始化通道
    ├─ LoadArgs()          // 加载任务参数
    ├─ PreSync()           // 前同步: NotifyRecord + NotifyWait
    ├─ DoAllGather()       // GroupBroadcast
    └─ PostSync()          // 后同步: NotifyRecord + NotifyWait
```

#### AIV Kernel 示例

**位置:** `template/aiv/kernel/aiv_all_gather_mesh_1d.h:77-88`

```
Process(count, sliceId, stride)
    ├─ 设置 tag
    ├─ 判断核数与 rankSize 关系
    │   ├─ numBlocks_ >= rankSize_: Run (每个核处理部分数据)
    │   └─ numBlocks_ < rankSize_: RunCtrlCore (每个核处理多个 rank)
    └─ BarrierAll()  // 全同步
```

---

## 八、关键类职责总结

| 类名 | 职责 | 关键方法 | 文件位置 |
|------|------|----------|---------|
| **AllGatherAutoSelector** | 算法自动选择 | SelectCcuScheduleAlgo, SelectAicpuAlgo, SelectAivAlgo | selector/all_gather_auto_selector.h |
| **InsV2AllGatherSoleExecutor** | 单模板执行编排 | CalcRes, Orchestrate, OrchestrateLoop | executor/ins_v2_all_gather_sole_executor.cc |
| **InsV2AllGatherParallelExecutor** | 并行执行编排 | OrchestrateLoop (两阶段并行) | executor/ins_v2_all_gather_parallel_executor.cc |
| **CcuTempAllGatherMesh1D** | CCU Mesh1D 模板 | CalcRes, KernelRun | template/ccu/ccu_temp_all_gather_mesh_1D.cc |
| **CcuKernelAllGatherMesh1D** | CCU Mesh1D Kernel | Algorithm, DoAllGather | template/ccu/kernel/ccu_kernel_all_gather_mesh1d.cc |
| **InsTempAllGatherMesh1D** | AICPU Mesh1D 模板 | RunAllGatherMesh, LocalDataCopy | template/aicpu/ins_temp_all_gather_mesh_1D.cc |
| **AivTempAllGatherMesh1D** | AIV Mesh1D 模板 | KernelRun, CalNumBlocks | template/aiv/aiv_temp_all_gather_mesh_1D.cc |
| **AivAllGatherMesh1D** | AIV 设备端 Kernel | Run, Process | template/aiv/kernel/aiv_all_gather_mesh_1d.h |

---

## 九、关键代码段行号引用

### 9.1 Selector 选择逻辑

| 功能 | 文件位置 |
|------|---------|
| CCU Schedule 选择 | selector/all_gather_auto_selector.cc:141-174 |
| AICPU 选择 | selector/all_gather_auto_selector.cc:176-232 |
| AIV 选择 | selector/all_gather_auto_selector.cc:234-245 |
| UBX 特殊处理 | selector/all_gather_auto_selector.cc:76-99 |

### 9.2 Executor 编排逻辑

| 功能 | 文件位置 |
|------|---------|
| Sole 执行循环 | executor/ins_v2_all_gather_sole_executor.cc:147-169 |
| Parallel 两阶段执行 | executor/ins_v2_all_gather_parallel_executor.cc:414-446 |
| 资源计算 | executor/ins_v2_all_gather_sole_executor.cc:42-61 |

### 9.3 Template 执行逻辑

| 功能 | 文件位置 |
|------|---------|
| CCU Mesh1D KernelRun | template/ccu/ccu_temp_all_gather_mesh_1D.cc:82-114 |
| CCU Kernel Algorithm | template/ccu/kernel/ccu_kernel_all_gather_mesh1d.cc:127-144 |
| AICPU Mesh1D RunAllGatherMesh | template/aicpu/ins_temp_all_gather_mesh_1D.cc:94-195 |
| AIV KernelRun | template/aiv/aiv_temp_all_gather_mesh_1D.cc:53-107 |

### 9.4 特殊处理

| 功能 | 文件位置 |
|------|---------|
| 双Die分组 | template/ccu/ccu_temp_all_gather_2dies_mesh_1D.cc:61-70 |
| Mem2Mem 参数 | template/ccu/ccu_temp_all_gather_mesh_1D_mem2mem.cc:91-98 |
| 尾块处理 | template/ccu/ccu_temp_all_gather_mesh_1D_mem2mem.cc:99-101 |

---

## 十、总结

### 10.1 架构设计特点

All_Gather 算子采用了**分层架构设计**:

1. **Selector 层**: 根据拓扑、数据量、硬件特性自动选择最优算法
2. **Executor 层**: 编排执行流程,管理资源分配
3. **Template 层**: 实现具体算法逻辑,支持多种执行模式(AICPU/CCU/AIV)

### 10.2 扩展性与可维护性

- **良好的模块化**: Selector、Executor、Template 职责清晰
- **灵活的拓扑支持**: 支持 MESH_1D、MESH_1D_CLOS、2DIES_MESH_1D、NHR 等多种拓扑
- **多样的执行模式**: AICPU、CCU、AIV 三种模式适应不同硬件场景
- **优化的变体实现**: Mem2Mem 变体支持原地操作和尾块处理

### 10.3 性能优化策略

- **数据量自适应**: 根据 512KB 阈值选择不同策略
- **并行执行**: ParallelExecutor 支持两阶段并行
- **DMA 优化**: AICPU 支持 DMA Read 模式
- **核数优化**: AIV 根据核数与 rankSize 关系选择不同执行模式
- **双Die并行**: 2DIES_MESH_1D 支持双 Kernel 并行执行

---

**文档生成时间:** 2026-04-13  
**分析范围:** src/ops/all_gather 目录下所有源文件  
**文档版本:** v1.0
