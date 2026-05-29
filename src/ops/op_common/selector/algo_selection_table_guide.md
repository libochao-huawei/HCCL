# 算法选择表（TableBasedAlgoSelector）使用说明

## 1. 模块概述

`algo_selection_table` 模块提供了一个基于规则表的算法选择器，用于根据运行时上下文（操作类型、拓扑结构、数据大小等条件）自动选择最合适的集合通信算法。

该模块由两个文件组成：

| 文件 | 说明 |
|------|------|
| `algo_selection_table.h` | 头文件，定义枚举类型、运行时上下文结构体及选择器类接口 |
| `algo_selection_table.cc` | 实现文件，包含规则匹配逻辑、默认规则表初始化及外部配置文件加载 |
| `algo_selection_config_example.txt` | 外部配置文件示例，包含 8 条典型规则 |

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

    // 初始化默认规则表 + 加载外部配置文件（外部规则优先）
    // configFilePath: 外部配置文件路径，为空字符串时仅加载默认规则
    void InitializeWithConfig(const std::string& configFilePath);

    // 从外部 txt 文件加载规则，返回成功加载的规则数量，失败返回 -1
    int LoadRulesFromFile(const std::string& filePath);

    // 从字符串内容解析规则（用于测试或嵌入式场景）
    int LoadRulesFromString(const std::string& content);

    // 根据上下文选择算法，匹配成功返回算法名，否则返回 std::nullopt
    std::optional<std::string> SelectAlgo(const AlgoSelectContext& ctx) const;

    // 获取规则表引用（可用于遍历或修改）
    std::vector<RuleMap>& GetRules();
    const std::vector<RuleMap>& GetRules() const;

    // 手动添加一条规则（追加到末尾）
    void AddRule(const RuleMap& rule);

    // 将规则插入到规则表头部（优先级高于默认规则）
    void PrependRule(const RuleMap& rule);

    // 打印规则表（调试用）
    void DumpTable() const;
};
```

### 5.2 基本使用流程（仅默认规则）

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

### 5.3 使用外部配置文件（推荐）

```cpp
#include "algo_selection_table.h"

// 一行完成：加载默认规则 + 外部配置文件
ops_hccl::TableBasedAlgoSelector selector;
selector.InitializeWithConfig("/path/to/algo_selection_config.txt");

// 外部规则会自动插入规则表头部，优先于默认规则匹配
ops_hccl::AlgoSelectContext ctx;
// ... 填充 ctx ...
auto result = selector.SelectAlgo(ctx);
```

### 5.4 仅从文件加载规则（追加到已有规则表）

```cpp
ops_hccl::TableBasedAlgoSelector selector;
selector.Initialize();  // 先加载默认规则

// 后续手动加载外部文件（外部规则插入头部，优先级高于默认规则）
int count = selector.LoadRulesFromFile("/path/to/custom_rules.txt");
if (count > 0) {
    std::cout << "Loaded " << count << " external rules." << std::endl;
}
```

### 5.5 从字符串加载规则（测试/嵌入场景）

```cpp
ops_hccl::TableBasedAlgoSelector selector;
selector.Initialize();

std::string configContent = R"(
[rule]
_execConfig = AICPU_TS
_opType = AllReduce
_dataSizeMax = 65536
_algo = aicpu_ar_ultra_small

[rule]
_execConfig = CCU_MS
_opType = Broadcast
_algo = ccu_ms_broadcast_fast
)";

int count = selector.LoadRulesFromString(configContent);
```

### 5.6 自定义规则（代码方式）

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

### 5.7 清空并重建规则表

```cpp
selector.GetRules().clear();  // 清空所有规则

// 手动逐条添加
selector.AddRule({...});
selector.AddRule({...});
```

### 5.8 调试：打印规则表

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

## 7. 外部配置文件详解

### 7.1 文件格式

配置文件为纯文本 `.txt` 文件，每条规则以 `[rule]` 标记开始，内部条件以 `key = value` 格式书写：

```
# 注释行（以 # 开头）
[rule]
_execConfig = AICPU_TS
_opType = AllReduce
_rankSizeMin = 1
_rankSizeMax = 4
_dataSizeMax = 131072
_algo = aicpu_ar_tiny_optimized

[rule]
_execConfig = AICPU_TS
_opType = Broadcast
_level0Topo = CLOS
_algo = broadcast_clos_custom
```

### 7.2 格式规则

| 规则 | 说明 |
|------|------|
| `[rule]` | 每条规则的起始标记，必须独占一行 |
| `key = value` | 条件键值对，等号两侧允许空格 |
| `# ...` | 注释行，会被忽略 |
| 空行 | 会被忽略 |
| `_execConfig` / `_opType` | 每条规则必填 |
| `_algo` | 每条规则必填，指定匹配成功后的算法名 |
| 其他 key | 可选，不写则不参与匹配 |

### 7.3 优先级机制

```
规则表顺序（从高到低）：
┌──────────────────────────────────────┐
│  外部配置规则 1  （文件中先写的）     │ ← 最高优先级
│  外部配置规则 2                       │
│  ...                                 │
│  外部配置规则 N  （文件中后写的）     │
├──────────────────────────────────────┤
│  内置默认规则 1                       │
│  内置默认规则 2                       │
│  ...                                 │
│  内置默认规则 25                      │ ← 最低优先级
└──────────────────────────────────────┘
```

- 外部规则通过 `PrependRule()` 插入规则表头部，**整体优先于**内置默认规则
- 外部规则之间，**文件中先写的规则优先级更高**
- 当外部规则与默认规则条件冲突时，外部规则优先生效

### 7.4 覆盖默认规则的策略

若希望外部规则精确覆盖某条默认规则，需确保外部规则的条件是默认规则的**子集或等价集**：

```
# 默认规则: AICPU_TS + AllReduce + dataSize<512KB → aicpu_ar_small
# 要覆盖它，外部规则需满足相同或更窄的条件范围：

[rule]
_execConfig = AICPU_TS
_opType = AllReduce
_dataSizeMax = 524288
_algo = my_custom_small_allreduce    # 替代 aicpu_ar_small
```

### 7.5 示例配置文件

参见项目中的 `algo_selection_config_example.txt`，其中包含 8 条覆盖不同场景的示例规则。

---

## 8. 扩展指南

### 8.1 新增一种条件

1. 在 `AlgoSelectContext` 结构体中添加新字段。
2. 在 `MatchRule()` 方法中添加对应的匹配逻辑。
3. 在规则 Map 中约定新的 key 名称（建议以 `_` 开头保持一致）。
4. 在 `Initialize()` 或外部通过 `AddRule()` 添加使用新条件的规则。

### 8.2 新增一种操作类型

1. 在 `HcclCMDType` 枚举中添加新值。
2. 在 `CmdTypeToStr()` 中添加对应的字符串映射。
3. 在 `Initialize()` 中添加该操作类型的默认规则。

### 8.3 新增一种执行配置

1. 在 `OpExecuteConfig` 枚举中添加新值。
2. 在 `ExecConfigToStr()` 中添加对应的字符串映射。
3. 在 `Initialize()` 中添加该执行配置下的默认规则。

---

## 9. FastAlgoSelector 高速算法选择器

### 9.1 设计动机

`TableBasedAlgoSelector` 采用 O(N) 全量遍历的方式逐条匹配规则，每条规则需要做多次字符串比较。当规则数量增长到数百甚至数千条时，线性扫描会成为性能瓶颈。`FastAlgoSelector` 通过**分桶 + 预解析 + 哈希查表**三层优化，将单次查询的时间复杂度从 O(N) 降低到接近 O(1)。

### 9.2 性能对比

| 优化维度 | TableBasedAlgoSelector | FastAlgoSelector |
|----------|----------------------|------------------|
| 查找范围 | O(N) 全量遍历所有规则 | O(1) 桶定位，候选集缩减到 2~6 条 |
| 条件匹配 | 每条规则做字符串转换 + 字符串比较 | 预解析为枚举/bool 类型化值，零字符串转换 |
| 离散条件匹配 | 顺序遍历每条规则 | O(1) 哈希精确查找（composite key） |
| 范围条件匹配 | 混合在全量遍历中 | 独立小集合顺序遍历（k≈2~4） |
| 外部规则优先级 | 插入链表头部 | 更低的 priority 值 + exactMap 覆盖 |

### 9.3 三层查找架构

```
查询 ctx (execConfig=AICPU_TS, opType=AllReduce, ...)
  │
  ▼  第 1 层: 桶查找 O(1)
buckets_[(execConfig, opType)]
  │    从 25+ 条规则缩减到 2~6 条候选
  │
  ▼  第 2 层: exactMap 哈希精确匹配 O(1)
exactMap[compositeKey]
  │    纯离散条件规则，类型化比较，零字符串转换
  │
  │  miss
  ▼  第 3 层: rangeRules 顺序遍历 O(k), k≈2~4
rangeRules 按优先级排序
       仅哈希 miss 时遍历小集合
```

**第 1 层 — 分桶（Bucket Index）：** 以 `(execConfig, opType)` 为 key 将所有规则分桶。查询时通过 `std::unordered_map` 直接定位到候选规则集，将搜索空间从全量 N 条规则缩减到单个桶内的 2~6 条。

**第 2 层 — 哈希精确匹配（Exact Map）：** 桶内纯离散条件（不含 `_rankSizeMin/Max`、`_dataSizeMin/Max`）的规则被编译为 `FastRule`，其所有离散条件按固定顺序拼接为一个 composite key 字符串。查询时根据规则的 key 子集构建相同的查询 key，通过 `std::unordered_map` 做 O(1) 精确查找。

**第 3 层 — 范围回退（Range Fallback）：** 桶内含范围条件的规则放入 `rangeRules` 向量，按优先级排序。仅在第 2 层未命中时顺序遍历（通常仅 2~4 条）。

### 9.4 核心数据结构

#### FastRule — 预解析规则

```cpp
struct FastRule {
    // --- 预解析的离散条件（std::optional 表示是否参与匹配） ---
    std::optional<int> topoLevel;            // -1 表示 "multi"
    std::optional<Level0Shape> level0Topo;
    std::optional<Level0MeshType> level0MeshType;
    std::optional<bool> level0PcieMix;
    std::optional<bool> is2DieFullMesh;
    std::optional<bool> level1Nhr;
    std::optional<bool> level0Nhr;
    std::optional<bool> meshEqClos;
    std::optional<bool> closMulMesh;
    std::optional<u32> localNetIns;
    std::optional<HcclReduceOp> reduceOp;
    std::optional<bool> overlap;
    std::optional<bool> hostToDevice;
    std::optional<bool> deviceToHost;
    std::vector<HcclDataType> dataTypes;     // 空 = 匹配所有

    // --- 范围条件 ---
    u32 rankSizeMin = 0;
    u32 rankSizeMax = UINT32_MAX;
    u64 dataSizeMin = 0;
    u64 dataSizeMax = UINT64_MAX;
    bool hasRange = false;

    // --- 结果 ---
    std::string algo;

    // --- composite key（仅离散部分，用于哈希查表） ---
    std::string compositeKey;

    // --- 优先级（越小越优先） ---
    int priority = 0;
};
```

**设计要点：**
- 所有离散条件在 `CompileRule()` 阶段从字符串预解析为 `std::optional<枚举/bool>` 类型化值，匹配时直接比较枚举/布尔值，无需任何字符串操作。
- `std::optional` 表示该条件是否参与匹配（无值 = 不参与 = 通配）。
- `compositeKey` 将所有已设置的离散条件按固定顺序拼接为规范化字符串，用于哈希精确查找。

#### AlgoBucket — 桶结构

```cpp
struct AlgoBucket {
    // 纯离散条件规则：compositeKey -> FastRule*（O(1) 哈希查表）
    std::unordered_map<std::string, const FastRule*> exactMap;

    // 含范围条件的规则：按优先级排序，顺序遍历
    std::vector<const FastRule*> rangeRules;
};
```

### 9.5 规则分类策略

`IndexRule()` 方法根据规则是否包含范围条件（`_rankSizeMin/Max`、`_dataSizeMin/Max`）自动将规则分类：

| 规则类型 | 存放位置 | 匹配方式 |
|----------|----------|----------|
| 纯离散条件 | `exactMap` | O(1) 哈希精确查找 |
| 含范围条件 | `rangeRules` | O(k) 顺序遍历 |

**外部规则优先级处理：** 外部配置规则通过 `PrependRule()` 插入，获得更低的 `priority` 值。对于 `exactMap` 中相同 `compositeKey` 的条目，外部规则会覆盖默认规则；对于 `rangeRules`，通过 priority 排序保证外部规则优先匹配。

### 9.6 API 使用

`FastAlgoSelector` 的 API 与 `TableBasedAlgoSelector` 完全兼容：

```cpp
class FastAlgoSelector {
public:
    void Initialize();
    void InitializeWithConfig(const std::string& configFilePath);
    int LoadRulesFromFile(const std::string& filePath);
    int LoadRulesFromString(const std::string& content);
    std::optional<std::string> SelectAlgo(const AlgoSelectContext& ctx) const;
    void AddRule(const RuleMap& rule);
    void PrependRule(const RuleMap& rule);
    const std::vector<RuleMap>& GetRules() const;
    void DumpTable() const;
};
```

### 9.7 使用示例

```cpp
#include "algo_selection_table.h"

// 一行完成：加载默认规则 + 外部配置文件
ops_hccl::FastAlgoSelector selector;
selector.InitializeWithConfig("/path/to/algo_selection_config.txt");

// 构造运行时上下文
ops_hccl::AlgoSelectContext ctx;
ctx.execConfig = ops_hccl::OpExecuteConfig::AICPU_TS;
ctx.opType = ops_hccl::HcclCMDType::HCCL_CMD_ALLREDUCE;
ctx.dataType = ops_hccl::HcclDataType::HCCL_DATA_TYPE_FP16;
ctx.reduceOp = ops_hccl::HcclReduceOp::HCCL_REDUCE_SUM;
ctx.dataSize = 4 * 1024 * 1024;    // 4MB
ctx.topoLevelNums = 1;
ctx.level0Topo = ops_hccl::Level0Shape::MESH_1D;
ctx.userRankSize = 32;

// 调用选择算法 — O(1) 查找
auto algo = selector.SelectAlgo(ctx);
if (algo.has_value()) {
    std::cout << "Selected: " << algo.value() << std::endl;
}
```

### 9.8 调试：打印桶结构

```cpp
selector.DumpTable();
// 输出示例：
// === FastAlgoSelector (25 rules, 8 buckets) ===
// Bucket [AICPU_TS | AllReduce]:
//   exact (3):
//     P4 key="_dataTypes=INT64,UINT64,FP64" -> aicpu_ar_64bit
//     P5 key="_reduceOp=PROD" -> aicpu_ar_prod
//     P7 key="" -> aicpu_ar_default
//   range (2):
//     P6 -> aicpu_ar_small
//     P8 -> aicpu_ar_l1nhr
// Bucket [CCU_MS | AllReduce]:
//   exact (1):
//     P10 key="_level0Topo=CLOS" -> ccu_ms_ar_clos
//   range (4):
//     P0 -> ccu_ms_ar_small_rank
//     P1 -> ccu_ms_ar_large_rank
//     ...
```

### 9.9 配置文件兼容性

`FastAlgoSelector` 与 `TableBasedAlgoSelector` 共享相同的配置文件格式。`algo_selection_config_example.txt` 中的 8 条示例规则无需任何修改即可被 `FastAlgoSelector` 正确加载和解析。

### 9.10 选择建议

| 场景 | 推荐选择 |
|------|----------|
| 规则数量较少（<50 条），对性能不敏感 | `TableBasedAlgoSelector`（实现简单，易于调试） |
| 规则数量较多（>100 条），热路径调用 | `FastAlgoSelector`（接近 O(1) 查询） |
| 需要动态修改规则表 | `TableBasedAlgoSelector`（直接操作 `vector`） |
| 外部配置覆盖 + 高性能查询 | `FastAlgoSelector`（自动分桶索引） |

---

## 10. 注意事项

- **规则顺序决定优先级**：先添加的规则先匹配，请确保更具体的规则排在更通用的规则之前。
- **Initialize() 会清空现有规则**：调用 `Initialize()` 会先执行 `rules_.clear()`，自定义规则需在 `Initialize()` 之后添加。
- **InitializeWithConfig() 是推荐入口**：它会先加载默认规则，再加载外部配置，外部规则自动获得最高优先级。
- **外部配置文件路径无效时不崩溃**：`LoadRulesFromFile()` 打开文件失败时会输出错误并返回 -1，不会抛出异常。
- **SelectAlgo 返回 std::optional**：若无规则匹配，返回 `std::nullopt`，调用方需处理此情况。
- **数值精度**：`_dataSizeMin` 和 `_dataSizeMax` 使用 `u64`（uint64_t），避免大数值溢出。
- **_dataSizeMax 是半开区间**：匹配条件为 `dataSize < _dataSizeMax`（不含等号），而 `_dataSizeMin` 为闭区间 `dataSize >= _dataSizeMin`。
- **FastAlgoSelector 的 Initialize() 会重建索引**：调用 `Initialize()` 会清空所有桶和 fastRules，然后从 `TableBasedAlgoSelector` 的默认规则重新构建索引。外部规则需在 `Initialize()` 之后通过 `AddRule()` 或 `PrependRule()` 添加。
- **FastAlgoSelector 的 PrependRule 语义**：外部规则获得更低的 priority 值，在 `exactMap` 中覆盖同 key 的默认规则，在 `rangeRules` 中按 priority 排序优先匹配。
