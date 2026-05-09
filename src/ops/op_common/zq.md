# CalcAGOmniPipeSliceInfo 函数详细解析

## 1. 函数概述

**函数签名**:
```cpp
OmniPipeSliceInfo CalcAGOmniPipeSliceInfo(OmniPipeSliceParam& omniPipeSliceParam);
```

**所在文件**:
- 声明: `src/ops/op_common/omnipipe_data_slice_calc.h:208`
- 实现: `src/ops/op_common/omnipipe_data_slice_calc.cc:837-1308`

**功能**: 计算 **3D AllGather (AG)** OmniPipe 通信的数据切片信息。在三维拓扑（x轴/y轴/z轴）下，根据各轴带宽比例，计算每步通信的数据量、偏移量，最终返回三轴各自的 `StepSliceInfo` 列表，供执行器按步执行数据搬运。

**调用方**:
| 调用位置 | 文件 | 行号 | 场景 |
|---------|------|------|------|
| AllGather OmniPipe 执行器 | `all_gather/executor/ins_v2_all_gather_omnipipe_executor.cc` | 318 | 对齐部分数据切片计算 |
| AllGather OmniPipe 执行器 | `all_gather/executor/ins_v2_all_gather_omnipipe_executor.cc` | 327 | 尾部数据切片计算 |
| AllReduce OmniPipe 执行器 | `all_reduce/executor/ins_v2_all_reduce_omnipipe_executor.cc` | 611 | AllReduce 中的 AG 阶段切片计算 |

---

## 2. 输入参数结构体 OmniPipeSliceParam

```cpp
struct OmniPipeSliceParam {
    std::vector<u64> levelRankSize;              // 三轴 rankSize: [x轴卡数, y轴卡数, z轴卡数]
    std::vector<EndpointAttrBwCoeff> endpointAttrBw; // 三轴平均带宽: [xB, yB, zB]
    std::vector<u64> dataSizePerLoop{0};         // 一次 loop 各 rank 的数据量大小
    u64 dataTypeSize{0};                         // 数据类型大小 (如 float=4, double=8)
    std::vector<u64> dataWholeSize{0};           // 总数据量 (各 rank)
    std::vector<u64> levelRankId;                // 本 rank 三轴坐标: [xAxis, yAxis, zAxis]
    std::vector<u64> levelAlgType;               // 三轴算法类型 (MESH=1, NHR=0)
    u64 root{0};                                 // root 节点 rankID
    OpMode opMode;                               // 操作模式 (OPBASE/OFFLOAD)
    CommEngine engine;                           // 通信引擎类型
};
```

**字段说明**:
| 字段 | 含义 | 示例 |
|------|------|------|
| `levelRankSize` | 三维拓扑各轴的 rank 数量 | `[2, 4, 8]` 表示 x=2, y=4, z=8 |
| `endpointAttrBw` | 三轴带宽系数 | `[10.0, 40.0, 20.0]` |
| `dataSizePerLoop` | 单次 loop 各 rank 搬运的数据量 | `[1024, 1024, ...]` |
| `dataTypeSize` | 单个数据元素的字节数 | `4` (float) |
| `dataWholeSize` | 各 rank 的总数据量 | `[4096, 4096, ...]` |
| `levelRankId` | 当前 rank 在三轴上的坐标 | `[1, 2, 3]` |
| `levelAlgType` | 各轴拓扑算法类型 | `[1, 1, 0]` |

---

## 3. 返回值结构体 OmniPipeSliceInfo

```cpp
struct OmniPipeSliceInfo {
    std::vector<StepSliceInfo> dataSliceLevel0;  // x轴每步数据偏移信息
    std::vector<StepSliceInfo> dataSliceLevel1;  // y轴每步数据偏移信息
    std::vector<StepSliceInfo> dataSliceLevel2;  // z轴每步数据偏移信息
    std::vector<std::vector<std::vector<u64>>> axlesReduceDstAddr;
};
```

其中 `StepSliceInfo` 定义:
```cpp
struct StepSliceInfo {
    BuffInfo buffInfo;                              // 缓冲区信息
    std::vector<std::vector<u64>> stepCount;        // 每步各 rank 的数据 count
    std::vector<std::vector<u64>> stepSliceSize;    // 每步各 rank 的数据 size (字节)
    std::vector<u64> stepInputSliceStride;          // 输入数据步间偏移
    std::vector<u64> stepOutputSliceStride;         // 输出数据步间偏移
    std::vector<std::vector<u64>> inputOmniPipeSliceStride;   // 输入 OmniPipe 切片偏移
    std::vector<std::vector<u64>> outputOmniPipeSliceStride;  // 输出 OmniPipe 切片偏移
};
```

---

## 4. 核心算法流程

### 4.1 整体流程图

```
CalcAGOmniPipeSliceInfo
│
├── 1. 提取拓扑参数 (x/y/z 轴的 rankSize, 带宽, 坐标)
│
├── 2. CalcBandwidth2D(xB, yB) → xyB  (计算 xy 平面 2D 等效带宽)
│
├── 3. OmniPipeSplitSliceInfoListAssign  (计算每 rank 数据切片分配)
│   ├── perLoop 版本 (dataSizePerLoop)
│   └── total 版本 (dataWholeSize)
│
├── 4. 计算通信步数和每步数据量 (分 xyB>zB 和 xyB<=zB 两种策略)
│   ├── 外层: CalAllgatherDataSize2D (z vs xy 平面)
│   ├── 内层: CalAllgatherDataSize2D (x vs y 平面)
│   └── CalAllgather2DOffset (z 轴偏移)
│
├── 5. 构建 z 轴 StepSliceInfo
│   ├── 同轴步 (zConnerStep 步)
│   └── 斜对角步 (outerStepNum - zConnerStep 步)
│
├── 6. CalAllgather2DOffset (x/y 轴偏移)
│
├── 7. 构建 x 轴 StepSliceInfo
│   ├── xyConnerStep 内: 同轴步 + 斜对角步
│   └── xyConnerStep 外: z 斜对角展开后的同轴步 + 斜对角步
│
├── 8. 构建 y 轴 StepSliceInfo (结构同 x 轴)
│
└── 9. 组装 OmniPipeSliceInfo 返回
```

### 4.2 阶段一：参数提取 (L839-865)

```cpp
int maxStepNum = MAX_STEP_NUM;  // = 5, 最大通信步数
u64 xRankSize = levelRankSize[OMNIPIPE_LEVEL0];  // x轴卡数 (机内 mesh)
u64 yRankSize = levelRankSize[OMNIPIPE_LEVEL1];  // y轴卡数 (机内 clos)
u64 zRankSize = levelRankSize[OMNIPIPE_LEVEL2];  // z轴卡数 (机间)
double xB = endpointAttrBw[OMNIPIPE_LEVEL0];     // x轴带宽
double yB = endpointAttrBw[OMNIPIPE_LEVEL1];     // y轴带宽
double zB = endpointAttrBw[OMNIPIPE_LEVEL2];     // z轴带宽
double xyB = CalcBandwidth2D(xB, yB, xRankSize, yRankSize, maxStepNum);  // xy平面2D等效带宽
u64 xAxis = levelRankId[OMNIPIPE_LEVEL0];  // 当前卡 x 坐标
u64 yAxis = levelRankId[OMNIPIPE_LEVEL1];  // 当前卡 y 坐标
u64 zAxis = levelRankId[OMNIPIPE_LEVEL2];  // 当前卡 z 坐标
u64 rankid = xAxis + yAxis * xRankSize + zAxis * xRankSize * yRankSize;  // 全局 rankID
u64 rankSize = xRankSize * yRankSize * zRankSize;  // 总 rank 数
```

**关键概念**:
- **x轴**: 机内 mesh 维度，通常带宽最低（慢轴）
- **y轴**: 机内 clos 维度，通常带宽最高（快轴）
- **z轴**: 机间维度，带宽居中
- **xyB**: 将 x/y 两轴合并为一个等效 2D 平面后的等效带宽

### 4.3 阶段二：数据切片分配 (L867-872)

```cpp
// 按 loop 数据量计算每 rank 的切片信息
std::vector<OmniPipeSplitSliceInfo> omniPipeSplitSliceInfoListPerLoop =
    OmniPipeSplitSliceInfoListAssign(dataSizePerLoop, rankSize, dataTypeSize);

// 按总数据量计算每 rank 的切片信息
std::vector<OmniPipeSplitSliceInfo> omniPipeSplitSliceInfoListTotal =
    OmniPipeSplitSliceInfoListAssign(dataSize, rankSize, dataTypeSize);
```

`OmniPipeSplitSliceInfoListAssign` 为每个 rank 计算其数据片的 `{offset, size, count}`:
- `offset`: 该 rank 数据在整个数据空间中的起始偏移
- `size`: 该 rank 的数据量 (字节)
- `count`: 该 rank 的数据元素个数 = size / dataTypeSize

### 4.4 阶段三：通信步数与数据量计算 (L874-940)

这是核心计算阶段，根据 **xy 平面等效带宽 (xyB)** 与 **z 轴带宽 (zB)** 的大小关系，分两种策略：

#### 策略 A: xyB > zB (xy 平面快，z 轴慢)

```
外层: CalAllgatherDataSize2D(zAGDataSize, xyAGDataSize, zB, xyB, zRankSize, xRankSize*yRankSize, ...)
  → z 为慢轴，xy 为快轴
内层: CalAllgatherDataSize2D(xAGDataSize, yAGDataSize, xB, yB, xRankSize, yRankSize, ...)
  → x 为慢轴，y 为快轴
```

#### 策略 B: xyB <= zB (z 轴快，xy 平面慢)

```
外层: CalAllgatherDataSize2D(xyAGDataSize, zAGDataSize, xyB, zB, xRankSize*yRankSize, zRankSize, ...)
  → xy 为慢轴，z 为快轴
内层: CalAllgatherDataSize2D(xAGDataSize, yAGDataSize, xB, yB, xRankSize, yRankSize, ...)
  → x 为慢轴，y 为快轴 (内层不变)
```

**关键变量**:
| 变量 | 含义 |
|------|------|
| `zAGDataSize[rs][osn]` | 第 rs 个 rank 在外层第 osn 步的 z 轴数据量 |
| `xyAGDataSize[rs][osn]` | 第 rs 个 rank 在外层第 osn 步的 xy 平面数据量 |
| `xAGDataSize[rs][osn][isn]` | 第 rs 个 rank 在外层第 osn 步、内层第 isn 步的 x 轴数据量 |
| `yAGDataSize[rs][osn][isn]` | 第 rs 个 rank 在外层第 osn 步、内层第 isn 步的 y 轴数据量 |
| `outerStepNum` | 外层 (z vs xy) 通信步数 |
| `innerStepNum` | 内层 (x vs y) 通信步数 |
| `zConnerStep` | z 轴同轴步数 (= outerStepNum - 1) |
| `xyConnerStep` | xy 平面同轴步数 (= outerStepNum - 1) |
| `xInCornerStep` | x 轴内层同轴步数 (= innerStepNum - 1) |

### 4.5 阶段四：z 轴 StepSliceInfo 构建 (L942-1003)

z 轴通信分为两部分：

#### 4.5.1 z 轴同轴步 (L944-970)

```
for osn = 0 to zConnerStep-1:
    对 z 轴上每个 rank (oneDid = 0..zRankSize-1):
        pieceId = oneDid * xRankSize * yRankSize + yAxis * xRankSize + xAxis
        sliceSize = zAGDataSize[pieceId][osn]
        inputOffset = zAGOffset[pieceId][osn]
        → 构建 StepSliceInfo (同轴数据搬运)
```

**同轴通信特点**: 数据在 z 轴方向上沿同一 xy 坐标传输，每个 rank 只搬运自己那片数据。`stepInputSliceStride` 和 `stepOutputSliceStride` 设为该 rank 数据片的 `offset`（表示从该 rank 数据起始位置开始）。

#### 4.5.2 z 轴斜对角步 (L972-1003)

```
for osn = zConnerStep to outerStepNum-1:
    对 z 轴上每个 rank (oneDid = 0..zRankSize-1):
        遍历 xy 平面所有 rank (connerDataSlice = 0..xRankSize*yRankSize-1):
            跳过自己 (connerDataSlice == yAxis*xRankSize+xAxis)
            pieceId = oneDid * xRankSize * yRankSize + connerDataSlice
            sliceSize = zAGDataSize[pieceId][osn]
            inputOffset = zAGOffset[pieceId][osn] + omniPipeSplitSliceInfoListTotal[pieceId].offset
        → 构建 StepSliceInfo (斜对角数据搬运)
```

**斜对角通信特点**: 向 z 轴上其他 rank 发送同 xy 平面的数据，需要遍历 xy 平面上除自己以外的所有数据片。偏移需要加上该 rank 数据片的 `offset`（跨 rank 数据定位）。`stepInputSliceStride` 和 `stepOutputSliceStride` 设为 0。

### 4.6 阶段五：x/y 轴 2D 偏移计算 (L1004-1010)

```cpp
for rs = 0 to rankSize-1:
    for osn = 0 to outerStepNum-1:
        CalAllgather2DOffset(xAGOffset[rs][osn], yAGOffset[rs][osn], innerStepNum,
                             xRankSize, yRankSize, xAGDataSize[rs][osn], yAGDataSize[rs][osn]);
```

为每个 rank、每个外层步计算内层 x/y 轴的 2D 偏移。

### 4.7 阶段六：x 轴 StepSliceInfo 构建 (L1012-1160)

x 轴通信分为两大阶段，每个阶段又分同轴步和斜对角步：

#### 4.7.1 xy 平面同轴阶段 (osn = 0..xyConnerStep-1)

**x 轴同轴步** (isn = 0..xInCornerStep-1):
```
对 x 轴上每个 rank (oneDid = 0..xRankSize-1):
    pieceId = zAxis * xRankSize * yRankSize + yAxis * xRankSize + oneDid
    sliceSize = xAGDataSize[pieceId][osn][isn]
    inputOffset = xyAGOffset[pieceId][osn] + xAGOffset[pieceId][osn][isn]
```
同轴通信：在 x 轴方向上沿同一 y、z 坐标传输。

**x 轴斜对角步** (isn = xInCornerStep..innerStepNum-1):
```
对 x 轴上每个 rank (oneDid = 0..xRankSize-1):
    遍历 y 轴所有 rank (connerDataSlice = 0..yRankSize-1):
        跳过自己 (connerDataSlice == yAxis)
        pieceId = zAxis * xRankSize * yRankSize + connerDataSlice * xRankSize + oneDid
        sliceSize = xAGDataSize[pieceId][osn][isn]
        inputOffset = xyAGOffset[pieceId][osn] + xAGOffset[pieceId][osn][isn] + offset
```
斜对角通信：向 x 轴上其他 rank 转发 y 轴方向的数据片。

#### 4.7.2 xy 平面斜对角阶段 (osn = xyConnerStep..outerStepNum-1)

此时 z 轴方向也展开为斜对角，需要处理 zRankSize-1 片数据做 2D 通信。

**x 轴同轴步** (isn = 0..xInCornerStep-1):
```
对 x 轴上每个 rank:
    遍历 z 轴所有 rank (outSliceNum = 0..zRankSize-1):
        跳过自己 (outSliceNum == zAxis)
        pieceId = outSliceNum * xRankSize * yRankSize + yAxis * xRankSize + oneDid
        sliceSize = xAGDataSize[pieceId][osn][isn]
        inputOffset = xyAGOffset[pieceId][osn] + xAGOffset[pieceId][osn][isn] + offset
```

**x 轴斜对角步** (isn = xInCornerStep..innerStepNum-1):
```
对 x 轴上每个 rank:
    遍历 z 轴所有 rank (outSliceNum):
        跳过自己 (outSliceNum == zAxis)
        遍历 y 轴所有 rank (connerDataSlice):
            跳过自己 (connerDataSlice == yAxis) 且 yRankSize > 1
            pieceId = outSliceNum * xRankSize * yRankSize + connerDataSlice * xRankSize + oneDid
            sliceSize = xAGDataSize[pieceId][osn][isn]
            inputOffset = xyAGOffset[pieceId][osn] + xAGOffset[pieceId][osn][isn] + offset
```
这是最复杂的情况：z 轴斜对角 + y 轴斜对角，需要三重循环遍历。

### 4.8 阶段七：y 轴 StepSliceInfo 构建 (L1162-1301)

y 轴的构建逻辑与 x 轴完全对称，只是：
- 遍历方向从 x 轴变为 y 轴
- 使用 `yAGDataSize` 和 `yAGOffset` 代替 `xAGDataSize` 和 `xAGOffset`
- 斜对角遍历的是 x 轴方向的数据片

结构同样分为：
1. **xy 平面同轴阶段**: y 轴同轴步 + y 轴斜对角步
2. **xy 平面斜对角阶段**: z 展开后的 y 轴同轴步 + y 轴斜对角步

### 4.9 阶段八：组装返回 (L1302-1308)

```cpp
struct OmniPipeSliceInfo dataSliceInfoxyz;
dataSliceInfoxyz.dataSliceLevel0 = dataSliceLevelx;  // x 轴
dataSliceInfoxyz.dataSliceLevel1 = dataSliceLevely;  // y 轴
dataSliceInfoxyz.dataSliceLevel2 = dataSliceLevelz;  // z 轴
return dataSliceInfoxyz;
```

---

## 5. 子函数详解

### 5.1 CalcBandwidth2D

```cpp
double CalcBandwidth2D(double xB, double yB, u64 xRankSize, u64 yRankSize, int maxStepNum);
```

**功能**: 计算 2D 拓扑的等效带宽（y 轴快）。

**算法**:
1. 退化情况: yRankSize==1 返回 xB; xRankSize==1 返回 yB
2. 调用 `CalAllgatherDataSizeRatio2D` 以 dataSize=1 计算每步数据比例
3. 等效带宽 = xB * 1.0 / sum(xAGDataSize) (慢轴总时间反推)

**物理含义**: 将 x/y 两轴的非均匀带宽"打平"为一个等效单轴带宽，用于与 z 轴比较决定外层通信策略。

### 5.2 CalAllgatherDataSize2D

```cpp
u64 CalAllgatherDataSize2D(u64* xStepP2pDataSize, u64* yStepP2pDataSize,
                           double xB, double yB, u64 xRankSize, u64 yRankSize,
                           u64 dataSizeEachRank, u64 maxStep);
```

**功能**: 计算 2D AllGather 每步数据片大小，返回通信步数。

**核心算法**:
1. 计算带宽比 `bandwidthRatio = yB / xB`
2. 计算斜对角等比 `omniPipeRatio = (xRankSize - 1) / bandwidthRatio`
3. 计算放大系数 `scale` (当步数超过 maxStep 时需要放大初始数据量)
4. 计算通信步数 `step`:
   - 若 `omniPipeRatio == 1`: `step = bandwidthRatio + 1`
   - 否则: `step = ceil(log(xRankSize - bandwidthRatio) / log(omniPipeRatio)) + 1`
   - 若 step <= maxStep, scale = 1 (无需放大)
5. 第一步: `xStepP2pDataSize[0] = scale * dataSizeEachRank / bandwidthRatio` (128 对齐)
6. 后续步: `yStepP2pDataSize[i] = xStepP2pDataSize[i-1]`, `xStepP2pDataSize[i] = yStepP2pDataSize[i] * (xRankSize-1) / bandwidthRatio`
7. 最后一步: 剩余数据切分

**对齐**: 所有数据量按 `HCCL_MIN_SLICE_ALIGN = 128` 字节对齐。

### 5.3 CalAllgather2DOffset

```cpp
void CalAllgather2DOffset(u64* xAGOffset, u64* yAGOffset, u64 stepNum,
                          u64 xRankSize, u64 yRankSize, u64* xAGDataSize, u64* yAGDataSize);
```

**功能**: 计算 2D AllGather 每步数据片的偏移量（y 轴快）。

**算法**:
- 第一步: x/y 偏移均为 0 (同轴发送)
- 第二步: y 偏移为 0 (斜对角发送)
- 后续步: y 偏移累加前一步 y 数据量; x 偏移累加前一步 x 数据量
- 最后一步: 特殊处理，慢轴发前半块，快轴发后半块

### 5.4 OmniPipeSplitSliceInfoListAssign

```cpp
std::vector<OmniPipeSplitSliceInfo> OmniPipeSplitSliceInfoListAssign(
    const std::vector<u64> dataWholeSize, u64 rankSize, u64 dataTypeSize);
```

**功能**: 为每个 rank 计算其数据片的偏移、大小和元素个数。

**算法**: 顺序遍历 rankSize 个 rank，累加 offset，每个 rank 的 size 取自 dataWholeSize[i]，count = size / dataTypeSize。

### 5.5 BuffInfoAssign

```cpp
void BuffInfoAssign(BuffInfo& bi, u64 inBuffBaseOff, u64 outBuffBaseOff, u64 hcclBuffBaseOff = 0);
```

**功能**: 设置 BuffInfo 的三个基地址偏移。在本函数中均设为 0。

---

## 6. 三维拓扑通信模型

### 6.1 拓扑结构

```
         z 轴 (机间)
         │
         │
    ┌────┼────┐
    │    │    │
    │  y 轴   │   (机内 clos, 快轴)
    │  (快)   │
    │         │
    └────x────┘   (机内 mesh, 慢轴)
```

- **x 轴 (LEVEL0)**: 机内 mesh，带宽最低，慢轴
- **y 轴 (LEVEL1)**: 机内 clos，带宽最高，快轴
- **z 轴 (LEVEL2)**: 机间，带宽居中

### 6.2 通信策略

函数采用 **分层 2D 通信** 策略实现 3D AllGather:

1. **外层**: 将 xy 平面视为一个整体，与 z 轴做 2D AllGather
   - 先做同轴通信 (z 轴方向)
   - 再做斜对角通信 (跨 z 轴)

2. **内层**: 在 xy 平面内做 2D AllGather
   - 先做同轴通信 (x 轴方向)
   - 再做斜对角通信 (跨 y 轴)

3. **带宽感知**: 根据带宽比例动态调整每步数据量，使各步通信时间均衡

### 6.3 同轴 vs 斜对角

| 通信类型 | 描述 | 偏移计算 |
|---------|------|---------|
| **同轴** | 沿同一轴方向传输数据，每个 rank 只处理自己的数据片 | offset = 轴偏移 |
| **斜对角** | 跨轴传输数据，需要处理其他 rank 的数据片 | offset = 轴偏移 + rank 数据片偏移 |

**判断依据**:
- 同轴步: `stepInputSliceStride` / `stepOutputSliceStride` 设为 rank 数据片的 offset
- 斜对角步: `stepInputSliceStride` / `stepOutputSliceStride` 设为 0

---

## 7. 关键数据流

```
OmniPipeSliceParam
    │
    ├── levelRankSize ──→ xRankSize, yRankSize, zRankSize
    ├── endpointAttrBw ──→ xB, yB, zB ──→ xyB (CalcBandwidth2D)
    ├── levelRankId ──→ xAxis, yAxis, zAxis ──→ rankid
    ├── dataSizePerLoop ──→ omniPipeSplitSliceInfoListPerLoop
    ├── dataWholeSize ──→ omniPipeSplitSliceInfoListTotal
    │
    ├── [xyB vs zB] ──→ CalAllgatherDataSize2D (外层)
    │                    └── CalAllgatherDataSize2D (内层)
    │                    └── CalAllgather2DOffset (z 轴偏移)
    │
    ├── z 轴 StepSliceInfo (同轴 + 斜对角)
    │
    ├── CalAllgather2DOffset (x/y 轴偏移)
    │
    ├── x 轴 StepSliceInfo (同轴 + 斜对角, 含 z 展开情况)
    ├── y 轴 StepSliceInfo (同轴 + 斜对角, 含 z 展开情况)
    │
    └── OmniPipeSliceInfo {dataSliceLevel0, dataSliceLevel1, dataSliceLevel2}
```

---

## 8. 常量与约束

| 常量 | 值 | 含义 |
|------|-----|------|
| `MAX_STEP_NUM` | 5 | 最大通信步数 |
| `HCCL_MIN_SLICE_ALIGN` | 128 | 数据片对齐字节数 |
| `UB_MAX_DATA_SIZE` | 256MB | 单次传输数据上限 |
| `BANDWIDTH_RATIO_BOUND` | 10 | 带宽比阈值 (RS 用) |
| `LOOP_SCALING_FACTOR` | 0.9 | Loop 缩放因子 |

---

## 9. 注意事项与优化点

1. **代码注释提到的优化点**: "uI 向 uO 拷贝可以和其他卡读写并发，省去本地拷贝的时间"
2. **数据对齐**: 所有数据量按 128 字节对齐，可能导致实际分配略大于理论值
3. **步数限制**: 最大步数限制为 5，超过时通过 scale 放大初始数据量来适配
4. **退化处理**: 当某轴 rankSize 为 1 时，退化为低维通信
5. **processedDataEachRank**: 预留偏移参数，当前填 0，未来可能用于增量计算
6. **VLA 使用**: 代码中使用了变长数组 (VLA) 如 `u64 zAGDataSize[rankSize][maxStepNum]`，这在 C++ 标准中非严格合规，但 GCC/Clang 扩展支持
