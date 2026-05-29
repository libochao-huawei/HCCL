# 算法选择表（TableBasedAlgoSelector）使用说明

## 1. 模块概述

`algo_selection_table` 模块提供了一个基于规则表的算法选择器，用于根据运行时上下文（操作类型、拓扑结构、数据大小等条件）自动选择最合适的集合通信算法。

该模块由两个文件组成：

| 文件 | 说明 |
|------|------|
| `algo_selection_table.h` | 头文件，定义枚举类型、运行时上下文结构体及选择器类接口 |
| `algo_selection_table.cc` | 实现文件，包含规则匹配逻辑和默认规则表的初始化 |

**核心设计思想：** 将所有匹配条件平铺在一个 `RuleMap`（`std::map<std::string, std::string>`）中，规则按顺序匹配，先匹配的先返回对应算法名。

---

## 2. 核心数据结构

### 2.1 枚举类型

#### OpExecuteConfig — 执行配置

| 枚举值 | 说明 |
|--------|------|
| `CCU_MS` | CCU MS 模式 |
| `CCU_SCHED` | CCU 调度模式 |
| `CCU_FAIL` | CCU 失败回退 |
| `AICPU_TS` | AICPU 时间片模式（默认） |
| `AIV` | AIV 模式 |
| `AIV_ONLY` | 仅 AIV 模式 |
| `HOSTCPU` | 主机 CPU 模式 |
| `HOSTCPU_TS` | 主机 CPU 时间片模式 |

#### HcclCMDType — 集合通信操作类型

| 枚举值 | 说明 |
|--------|------|
| `HCCL_CMD_ALLREDUCE` | AllReduce |
| `HCCL_CMD_ALLGATHER` | AllGather |
| `HCCL_CMD_ALLGATHER_V` | AllGatherV |
| `HCCL_CMD_REDUCE_SCATTER` | ReduceScatter |
| `HCCL_CMD_REDUCE_SCATTER_V` | ReduceScatterV |
| `HCCL_CMD_BROADCAST` | Broadcast |
| `HCCL_CMD_REDUCE` | Reduce |
| `HCCL_CMD_SCATTER` | Scatter |
| `HCCL_CMD_ALLTOALL` | AlltoAll |
| `HCCL_CMD_ALLTOALLV` | AlltoAllV |
| `HCCL_CMD_ALLTOALLVC` | AlltoAllVC |
| `HCCL_CMD_SEND` | Send |
| `HCCL_CMD_RECV` | Recv |
| `HCCL_CMD_BATCH_SEND_RECV` | BatchSendRecv |

#### HcclDataType — 数据类型

`INT8`、`INT32`、`FP16`、`BF16`、`FP32`、`INT64`、`UINT64`、`FP64`

#### HcclReduceOp — 归约操作类型

`SUM`（求和）、`PROD`（乘积）、`MAX`（最大值）、`MIN`（最小值）

#### Level0Shape — Level0 拓扑形状

| 枚举值 | 说明 |
|--------|------|
| `MESH_1D` | 一维 Mesh |
| `MESH_1D_CLOS` | 一维 Mesh + CLOS 混合 |
| `CLOS` | CLOS 拓扑 |

#### Level0MeshType — Level0 Mesh 类型

| 枚举值 | 说明 |
|--------|------|
| `SINGLE_DIE` | 单 Die |
| `TWO_DIE_REGULAR` | 双 Die 规则排列 |
| `TWO_DIE_NOT_REGULAR` | 双 Die 非规则排列 |

### 2.2 运行时上下文 AlgoSelectContext

在调用算法选择之前，需要填充此结构体以描述当前的运行时环境：

```cpp
struct AlgoSelectContext {
    OpExecuteConfig execConfig;     // 执行配置（必填）
    HcclCMDType opType;             // 操作类型（必填）
    HcclDataType dataType;          // 数据类型
    HcclReduceOp reduceOp;          // 归约操作类型
    u64 dataSize;                   // 数据大小（字节）
    u32 topoLevelNums;              // 拓扑层级数（1=单层, >1=多层）
    Level0Shape level0Topo;         // Level0 拓扑形状
    Level0MeshType level0MeshType;  // Level0 Mesh 类型
    bool level0PcieMix;             // Level0 是否 PCIe 混合
    bool is2DieFullMesh;            // 是否为双 Die 全 Mesh
    bool level1Nhr;                 // Level1 是否 NHR
    bool level0Nhr;                 // Level0 是否 NHR
    bool meshNumEqualToClosNum;     // Mesh 数量是否等于 CLOS 数量
    bool closNumMultipleOfMeshNum;  // CLOS 数量是否为 Mesh 数量的倍数
    u32 localNetInsSizeOfLayer0;    // Level0 本地网络实例数
    u32 userRankSize;               // 用户 Rank 数量
    bool isInputOutputOverlap;      // 输入输出是否重叠
    bool isHostToDevice;            // 是否 Host 到 Device 传输
    bool isDeviceToHost;            // 是否 Device 到 Host 传输
};
```

---

## 3. 规则条件一览

每条规则是一个 `RuleMap`（`std::map<std::string, std::string>`），下表列出所有支持的条件 key：

### 3.1 必填条件

| Key | 说明 | 取值示例 |
|-----|------|----------|
| `_execConfig` | 执行配置 | `"CCU_MS"`, `"CCU_SCHED"`, `"AICPU_TS"`, `"HOSTCPU"` |
| `_opType` | 操作类型 | `"AllReduce"`, `"Broadcast"`, `"AllGather"` |

### 3.2 可选条件

| Key | 类型 | 说明 | 取值示例 |
|-----|------|------|----------|
| `_topoLevel` | 数值/字符串 | 拓扑层级，`"multi"` 表示 >1 层 | `"1"`, `"2"`, `"multi"` |
| `_level0Topo` | 枚举字符串 | Level0 拓扑形状 | `"MESH_1D"`, `"CLOS"`, `"MESH_1D_CLOS"` |
| `_level0MeshType` | 枚举字符串 | Level0 Mesh 类型 | `"SINGLE_DIE"`, `"TWO_DIE_REGULAR"` |
| `_level0PcieMix` | 布尔 | Level0 是否 PCIe 混合 | `"true"`, `"false"` |
| `_is2DieFullMesh` | 布尔 | 是否双 Die 全 Mesh | `"true"`, `"false"` |
| `_level1Nhr` | 布尔 | Level1 是否 NHR | `"true"`, `"false"` |
| `_level0Nhr` | 布尔 | Level0 是否 NHR | `"true"`, `"false"` |
| `_meshEqClos` | 布尔 | Mesh 数量是否等于 CLOS 数量 | `"true"`, `"false"` |
| `_closMulMesh` | 布尔 | CLOS 数量是否为 Mesh 数量的倍数 | `"true"`, `"false"` |
| `_localNetIns` | 数值 | Level0 本地网络实例数（精确匹配） | `"4"` |
| `_rankSizeMin` | 数值 | Rank 数量最小值（含），`userRankSize >= 此值` | `"1"`, `"8"` |
| `_rankSizeMax` | 数值 | Rank 数量最大值（含），`userRankSize <= 此值` | `"7"`, `"64"` |
| `_dataSizeMin` | 数值(字节) | 数据大小最小值（含），`dataSize >= 此值` | `"524288"`（512KB） |
| `_dataSizeMax` | 数值(字节) | 数据大小最大值（不含），`dataSize < 此值` | `"1048576"`（1MB） |
| `_dataTypes` | 逗号分隔列表 | 数据类型白名单，匹配任一即可 | `"INT64,UINT64,FP64"` |
| `_reduceOp` | 枚举字符串 | 归约操作类型 | `"SUM"`, `"PROD"`, `"MAX"`, `"MIN"` |
| `_overlap` | 布尔 | 输入输出是否重叠 | `"true"`, `"false"` |
| `_hostToDevice` | 布尔 | 是否 Host → Device 传输 | `"true"`, `"false"` |
| `_deviceToHost` | 布尔 | 是否 Device → Host 传输 | `"true"`, `"false"` |

### 3.3 结果字段

| Key | 说明 |
|-----|------|
| `_algo` | 匹配成功后返回的算法名称字符串 |

---

## 4. 匹配规则说明

1. **顺序匹配**：规则按加入顺序逐条匹配，第一条完全匹配的规则返回其 `_algo` 值。
2. **条件与逻辑**：一条规则中的所有条件必须全部满足（AND 关系）。
3. **可选条件缺省**：若规则中未指定某可选条件，则该条件不参与匹配（视为通配）。
4. **范围匹配**：
   - `_rankSizeMin` / `_rankSizeMax`：`userRankSize` 的闭区间 `[min, max]`
   - `_dataSizeMin` / `_dataSizeMax`：`dataSize` 的半开区间 `[min, max)`
5. **多值匹配**：`_dataTypes` 支持逗号分隔的多个值，上下文的数据类型匹配其中任意一个即通过。

---

## 5. API 使用指南

### 5.1 类接口

```cpp
class TableBasedAlgoSelector {
public:
    // 初始化默认规则表
    void Initialize();

    // 根据上下文选择算法，匹配成功返回算法名，否则返回 std::nullopt
    std::optional<std::string> SelectAlgo(const AlgoSelectContext& ctx) const;

    // 获取规则表引用（可用于遍历或修改）
    std::vector<RuleMap>& GetRules();
    const std::vector<RuleMap>& GetRules() const;

    // 手动添加一条规则
    void AddRule(const RuleMap& rule);

    // 打印规则表（调试用）
    void DumpTable() const;
};
```

### 5.2 基本使用流程

```cpp
#include "algo_selection_table.h"

// 1. 创建选择器并初始化默认规则
ops_hccl::TableBasedAlgoSelector selector;
selector.Initialize();

// 2. 构造运行时上下文
ops_hccl::AlgoSelectContext ctx;
ctx.execConfig = ops_hccl::OpExecuteConfig::AICPU_TS;
ctx.opType = ops_hccl::HcclCMDType::HCCL_CMD_ALLREDUCE;
ctx.dataType = ops_hccl::HcclDataType::HCCL_DATA_TYPE_FP32;
ctx.reduceOp = ops_hccl::HcclReduceOp::HCCL_REDUCE_SUM;
ctx.dataSize = 256 * 1024;       // 256KB
ctx.topoLevelNums = 1;
ctx.level0Topo = ops_hccl::Level0Shape::MESH_1D;
ctx.userRankSize = 8;

// 3. 调用选择算法
auto result = selector.SelectAlgo(ctx);
if (result.has_value()) {
    std::cout << "Selected algorithm: " << result.value() << std::endl;
} else {
    std::cout << "No matching algorithm found." << std::endl;
}
```

### 5.3 自定义规则

```cpp
// 添加一条自定义规则：CCU_MS 模式下，CLOS 拓扑的 AllGather 使用特定算法
ops_hccl::RuleMap customRule = {
    {"_execConfig", "CCU_MS"},
    {"_opType", "AllGather"},
    {"_level0Topo", "CLOS"},
    {"_dataSizeMin", "0"},
    {"_dataSizeMax", std::to_string(1024 * 1024)},  // 1MB
    {"_algo", "my_custom_allgather_clos"}
};
selector.AddRule(customRule);
```

> **注意：** 自定义规则添加后位于规则表末尾。如果需要优先匹配，应在 `Initialize()` 之前或清理规则表后添加，或者通过 `GetRules()` 获取引用后手动调整顺序。

### 5.4 清空并重建规则表

```cpp
selector.GetRules().clear();  // 清空所有规则

// 手动逐条添加
selector.AddRule({...});
selector.AddRule({...});
```

### 5.5 调试：打印规则表

```cpp
selector.DumpTable();
// 输出示例：
// === Algorithm Selection Table (25 rules) ===
// [0] CCU_MS AllReduce -> ccu_ms_ar_small_rank
// [1] CCU_MS AllReduce -> ccu_ms_ar_large_rank
// ...
```

---

## 6. 内置默认规则表

`Initialize()` 方法会加载以下默认规则（按优先级从高到低排列）：

### 6.1 CCU_MS 模式 — AllReduce

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | topoLevel=1, level0Topo=MESH_1D, rankSize∈[1,7], dataSize<512KB | `ccu_ms_ar_small_rank` |
| 2 | topoLevel=1, level0Topo=MESH_1D, rankSize≥8, dataSize<512KB | `ccu_ms_ar_large_rank` |
| 3 | topoLevel=1, level0Topo=MESH_1D, level0PcieMix=false, dataSize∈[512KB, 8MB) | `ccu_ms_ar_medium` |
| 4 | topoLevel=1, level0Topo=MESH_1D, dataSize≥8MB | `ccu_ms_ar_large` |
| 5 | level0Topo=CLOS | `ccu_ms_ar_clos` |

### 6.2 CCU_SCHED 模式 — AllReduce

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | level0Topo=MESH_1D, is2DieFullMesh=true | `ccu_sched_ar_2die` |
| 2 | level0Topo=MESH_1D | `ccu_sched_ar_default` |

### 6.3 AICPU_TS 模式 — AllReduce

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | dataTypes=INT64/UINT64/FP64 | `aicpu_ar_64bit` |
| 2 | reduceOp=PROD | `aicpu_ar_prod` |
| 3 | dataSize<512KB | `aicpu_ar_small` |
| 4 | （无额外条件） | `aicpu_ar_default` |
| 5 | topoLevel=multi, level1Nhr=true | `aicpu_ar_l1nhr` |
| 6 | topoLevel=multi | `aicpu_ar_multilvl` |

### 6.4 AICPU_TS 模式 — Broadcast

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | dataSize<1MB | `broadcast_oneshot` |
| 2 | （无额外条件） | `broadcast_twoshot` |

### 6.5 AICPU_TS 模式 — AllGather

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | dataSize<256KB | `allgather_small` |
| 2 | （无额外条件） | `allgather_default` |

### 6.6 AICPU_TS 模式 — ReduceScatter

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | dataSize<256KB | `reducescatter_small` |
| 2 | （无额外条件） | `reducescatter_default` |

### 6.7 AICPU_TS 模式 — AlltoAll

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | level0PcieMix=true | `alltoall_pcie` |
| 2 | （无额外条件） | `alltoall_default` |

### 6.8 HOSTCPU 模式 — Send

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | hostToDevice=true | `send_host_dpu` |
| 2 | （无额外条件） | `send_device_dpu` |

### 6.9 HOSTCPU 模式 — Recv

| # | 条件 | 算法名 |
|---|------|--------|
| 1 | deviceToHost=true | `recv_host_dpu` |
| 2 | （无额外条件） | `recv_device_dpu` |

---

## 7. 扩展指南

### 7.1 新增一种条件

1. 在 `AlgoSelectContext` 结构体中添加新字段。
2. 在 `MatchRule()` 方法中添加对应的匹配逻辑。
3. 在规则 Map 中约定新的 key 名称（建议以 `_` 开头保持一致）。
4. 在 `Initialize()` 或外部通过 `AddRule()` 添加使用新条件的规则。

### 7.2 新增一种操作类型

1. 在 `HcclCMDType` 枚举中添加新值。
2. 在 `CmdTypeToStr()` 中添加对应的字符串映射。
3. 在 `Initialize()` 中添加该操作类型的默认规则。

### 7.3 新增一种执行配置

1. 在 `OpExecuteConfig` 枚举中添加新值。
2. 在 `ExecConfigToStr()` 中添加对应的字符串映射。
3. 在 `Initialize()` 中添加该执行配置下的默认规则。

---

## 8. 注意事项

- **规则顺序决定优先级**：先添加的规则先匹配，请确保更具体的规则排在更通用的规则之前。
- **Initialize() 会清空现有规则**：调用 `Initialize()` 会先执行 `rules_.clear()`，自定义规则需在 `Initialize()` 之后添加。
- **SelectAlgo 返回 std::optional**：若无规则匹配，返回 `std::nullopt`，调用方需处理此情况。
- **数值精度**：`_dataSizeMin` 和 `_dataSizeMax` 使用 `u64`（uint64_t），避免大数值溢出。
- **_dataSizeMax 是半开区间**：匹配条件为 `dataSize < _dataSizeMax`（不含等号），而 `_dataSizeMin` 为闭区间 `dataSize >= _dataSizeMin`。