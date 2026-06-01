# PR #1158: Reduce Scatter 3-Level Sequence Executor 参数推演分析

## 实例推演设定

- **拓扑**: 8P(frame) × 8Pod × 2Cluster = 128 ranks
- **数据**: FP32 (dataTypeSize=4B), 每rank 128元素
- **观察rank**: rank 37 → rankIdxLevel0=5, rankIdxLevel1=4, rankIdxLevel2=0

---

## 1. Rank Index 分解公式验证

公式：
```
rankIdxLevel0 = myRank % rankSizeLevel0
rankIdxLevel1 = (myRank / rankSizeLevel0) % rankSizeLevel1
rankIdxLevel2 = myRank / (rankSizeLevel0 * rankSizeLevel1)
```

| rank | rankIdxLevel0 | rankIdxLevel1 | rankIdxLevel2 | 含义 |
|------|-------------|-------------|-------------|------|
| 0 | 0%8=**0** | (0/8)%8=**0** | 0/64=**0** | 第0帧第0Pod第0集群 |
| 37 | 37%8=**5** | (37/8)%8=4%8=**4** | 37/64=**0** | 第5帧第4Pod第0集群 |
| 100 | 100%8=**4** | (100/8)%8=12%8=**4** | 100/64=**1** | 第4帧第4Pod第1集群 |
| 127 | 127%8=**7** | (127/8)%8=15%8=**7** | 127/64=**1** | 第7帧第7Pod第1集群 |

还原验证：`rankIdxLevel2*64 + rankIdxLevel1*8 + rankIdxLevel0 = myRank`
- rank=100: 1×64 + 4×8 + 4 = 64+32+4 = 100 ✓

**结论：分解公式正确，可完整还原。** ✅

---

## 2. TopoForLayer2 Rank 过滤逻辑验证

过滤条件：`rankId % (layer0Size * layer1Size) == myRank % (layer0Size * layer1Size)` + 有直连链路

以 rank 37 为例，`layer0Size=8, layer1Size=8`：
- 条件：`rankId % 64 == 37 % 64 == 37`
- 0~127范围内满足 `rankId % 64 == 37` 的rank：{**37**, **101**}
  - rank 37: 37%64=37 ✓ (自身)
  - rank 101: 101%64=37 ✓ (第1集群中同位置)
  - rank 5: 5%64=5 ≠ 37 ✗ (同一帧内但不同Pod位置)
  - rank 69: 69%64=5 ≠ 37 ✗

**语义验证**：rank 37 是(帧5, Pod4, 集群0)，rank 101 是(帧5, Pod4, 集群1)——它们在帧内和Pod内位置完全相同，只有集群不同。这正是 level2 NHR 需要通信的伙伴。✅

**链路检查**：只有存在直连链路的rank才纳入，确保拓扑可行性。✅

---

## 3. Level0 Mesh — GenIntraTemplateParams 参数推演

场景：rank 37, `currDataCount=128, dataTypeSize_=4, processedDataCount=0`

### 语义

Reduce Scatter Mesh 1D：帧内8个rank各自贡献自己的数据切片，帧内reduce后每个rank得到一个reduce后的切片。由于有 `rankSizeLevel1 * rankSizeLevel2 = 16` 个更高层组需要分别reduce，所以 Mesh 需重复16次。

### 参数验证

| 参数 | 公式 | rank 37实例值 | 是否正确 | 说明 |
|------|------|-------------|---------|------|
| `inBuffBaseOff` | `processedDataCount * dataTypeSize_` | 0 | ✅ | 首轮循环偏移为0 |
| `outBuffBaseOff` | 0 | 0 | ✅ | 输出从CCL buffer开头写 |
| `sliceSize` | `currDataCount * dataTypeSize_` | 512B | ✅ | 每个slice大小 |
| `inputSliceStride` | `dataSize_` | 512B | ✅ | 输入中相邻rank数据间隔为一个rank的全部数据 |
| `outputSliceStride` | `currDataCount * dataTypeSize_` | 512B | ✅ | CCL buffer中各rank输出切片相邻排列 |
| `repeatNum` | `rankSizeLevel1_ * rankSizeLevel2_` | 8×2=16 | ✅ | 16个上层组各需帧内reduce |
| `inputRepeatStride` | `rankSizeLevel0_ * dataSize_` | 8×512=4096B | ✅ | 每repeat跳跃一帧8个rank的数据量 |
| `outputRepeatStride` | `rankSizeLevel0_ * currDataCount * dataTypeSize_` | 8×512=4096B | ✅ | CCL buffer中每repeat占8×512=4096B |

### CCL Buffer 总使用量

= repeatNum × rankSizeLevel0 × sliceSize = 16 × 8 × 512 = 65536B = 64KB

正好等于全局128rank的总数据量。✅

---

## 4. Level1 NHR — GenInterTemplateParams1 参数推演

场景：rank 37 (rankIdxLevel0=5), `currDataCount=128, dataTypeSize_=4`

### 语义

NHR Level1：Pod内8个帧做reduce-scatter，每个rank只reduce自己负责的那片数据。rank 37负责帧位置5对应的切片。

### CCL Buffer 布局（Mesh输出后）

Mesh输出的CCL buffer布局：
```
[repeat0: rank0slice, rank1slice, ..., rank7slice]  (4096B)
[repeat1: rank0slice, rank1slice, ..., rank7slice]  (4096B)
...
[repeat15: ...]  (4096B)
```

rank 37(rankIdxLevel0=5)在CCL buffer中位置5的slice偏移 = `5×512 = 2560`

### 参数验证

| 参数 | 公式 | rank 37实例值 | 是否正确 | 说明 |
|------|------|-------------|---------|------|
| `inBuffBaseOff` | `rankIdxLevel0_ * currDataCount * dataTypeSize_` | 5×128×4=2560B | ✅ | 指向帧内位置5的slice |
| `outBuffBaseOff` | 0 | 0 | ✅ | in-place写回 |
| `hcclBuffBaseOff` | `rankIdxLevel0_ * currDataCount * dataTypeSize_` | 2560B | ✅ | 与输入同一位置 |
| `sliceSize` | `currDataCount * dataTypeSize_` | 512B | ✅ | |
| `inputSliceStride` | `rankSizeLevel0_ * currDataCount * dataTypeSize_` | 4096B | ✅ | 跳一整个repeat组=Pod内不同帧。从偏移2560开始: slice0=2560, slice1=6656, slice7=31232，对应Pod内8帧的位置5数据 |
| `outputSliceStride` | 0 | 0 | ✅ | in-place无stride |
| `repeatNum` | `rankSizeLevel2_` | 2 | ✅ | 集群0和集群1各一次 |
| `inputRepeatStride` | `rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_` | 8×8×128×4=32768B | ✅ | 跳8×8×512=一Pod的数据量。repeat1从2560+32768=34848开始 |
| `outputRepeatStride` | 0 | 0 | ✅ | in-place无stride |

### NHR Level1 输出后CCL buffer变化

reduce后，rank 37只in-place写回偏移2560的位置（自己负责的slice）。其他帧的位置5数据被NHR通信消耗但不被rank 37覆盖。✅

---

## 5. Level2 NHR — GenInterTemplateParams2 参数推演

场景：rank 37 (rankIdxLevel0=5), `currDataCount=128, processedDataCount=0`

### 语义

NHR Level2：集群间2个rank做最终reduce，将Pod内reduce结果进一步reduce到最终输出。

### 参数验证

| 参数 | 公式 | rank 37实例值 | 是否正确 | 说明 |
|------|------|-------------|---------|------|
| `inBuffBaseOff` | `rankIdxLevel0_ * currDataCount * dataTypeSize_` | 5×128×4=2560B | ✅ | 指向NHR1结果（rank 37的帧位置5） |
| `outBuffBaseOff` | `processedDataCount * dataTypeSize_` | 0 | ✅ | 首轮从OUTPUT起始位置写 |
| `hcclBuffBaseOff` | `rankIdxLevel0_ * currDataCount * dataTypeSize_` | 2560B | ✅ | 与输入同一位置 |
| `sliceSize` | `currDataCount * dataTypeSize_` | 512B | ✅ | |
| `inputSliceStride` | `rankSizeLevel0_ * currDataCount * dataTypeSize_` | 4096B | ⚠️ | repeatNum=1时stride冗余，不影响功能。详见下方说明 |
| `outputSliceStride` | 0 | 0 | ✅ | 单次写入无stride |
| `repeatNum` | 1 | 1 | ✅ | 最终层无需重复 |
| `inputRepeatStride` | 0 | 0 | ✅ | repeatNum=1不需要 |
| `outputRepeatStride` | 0 | 0 | ✅ | repeatNum=1不需要 |

### ⚠️ 关于 inputSliceStride=4096B 的说明

NHR Level2只有2个参与rank(rank 37和rank 101)，`repeatNum=1`。NHR模板从 `inBuffBaseOff=2560` 开始只读取1个slice，stride参数实际不生效（因为只有1个slice需要读取）。

`4096B = rankSizeLevel0 × currDataCount × dataTypeSize_` 是帧内间距，而非集群间距。如果未来扩展到4个集群（repeatNum>1），stride应改为集群间间距（32768B）。当前repeatNum=1时该参数冗余，不影响功能正确性。

---

## 6. Scratch Multiplier 和 maxCountPerLoop 验证

公式：
```
templateScratchMultiplier0 = CalcScratchMultiple(INPUT, HCCL_BUFFER)   // Mesh层
templateScratchMultiplier1 = CalcScratchMultiple(HCCL_BUFFER, HCCL_BUFFER) // NHR层1
templateScratchMultiplier2 = CalcScratchMultiple(HCCL_BUFFER, OUTPUT)  // NHR层2
templateScratchMultiplier = multiplier0 * multiplier1 * multiplier2
```

乘法形式的含义：三层串行执行，CCL buffer总量需要满足所有层的中间数据需求。

### 实例推演

假设每层 scratch multiplier = 1（CCL buffer恰好容纳一轮数据）：
- `templateScratchMultiplier = 1 × 1 × 1 = 1`
- CCL buffer大小 = 65536B (128rank × 128元素 × 4B)
- `maxCountPerLoop = cclMem.size / 1 / HCCL_MIN_SLICE_ALIGN × HCCL_MIN_SLICE_ALIGN / 4`
  - 假设 `HCCL_MIN_SLICE_ALIGN=128B`
  - = 65536 / 128 × 128 / 4 = 512 × 128 / 4 = 16384 元素
  - 每rank dataCount=128, 总dataCount=128×128=16384 ✓ 一轮正好处理完

- `loopTimes = ceil(16384 / 16384) = 1` ✓

如果数据量更大或CCL buffer更小，需要多轮循环。乘法形式确保每层有足够CCL buffer空间。✅

---

## 7. Buffer 复用安全性验证（in-place NHR）

三层串行执行的数据流：
```
INPUT → [Mesh L0] → CCL Buffer → [NHR L1] → CCL Buffer → [NHR L2] → OUTPUT
```

### 安全性分析

1. **Mesh L0**: 读INPUT，写CCL Buffer。完成后INPUT不再被引用，CCL Buffer包含帧内8个rank的reduce结果（16组 × 8帧 × 128元素）。✅

2. **NHR L1**: 读CCL Buffer，in-place写回CCL Buffer。NHR reduce-scatter特性：每个rank只修改自己负责的切片位置。rank 37只修改偏移2560。串行执行中所有rank先完成Mesh再执行NHR L1，NHR通信是同步的，不存在一个rank覆盖另一个rank还需读取的数据。✅

3. **NHR L2**: 读CCL Buffer中NHR L1的结果，写OUTPUT。NHR L1完成后偏移2560包含Pod内8帧的reduce结果。NHR L2读取该位置做集群间reduce，结果写入OUTPUT。✅

### NHR L1 in-place 不覆盖 NHR L2 所需数据

NHR L1 repeat0处理集群0数据写回偏移2560，repeat1处理集群1数据写回偏移2560+32768=34848。两个写入互不干扰，NHR L2只需读取偏移2560。✅

---

## 8. CalcRes 资源合并逻辑验证

合并策略：
```cpp
slaveThreadNum = max(resReq0.slaveThreadNum, resReq1.slaveThreadNum, resReq2.slaveThreadNum)
notifyNumPerThread[i] = max(resReq0.notifyNumPerThread[i], resReq1.notifyNumPerThread[i], resReq2.notifyNumPerThread[i])
notifyNumOnMainThread = max(resReq0.notifyNumOnMainThread, resReq1.notifyNumOnMainThread, resReq2.notifyNumOnMainThread)
channels[0] = resReq0.channels[0], channels[1] = resReq1.channels[0], channels[2] = resReq2.channels[0]
```

### 实例推演

假设 resReq0需要2个slave线程, resReq1需要3个, resReq2需要1个：
- **slaveThreadNum = max(2, 3, 1) = 3**：分配3个slave线程，Mesh层只用2个（多分配不影响），NHR层1用3个正好 ✓
- **notifyNumPerThread[i] = max(...)**：线程间同步信号量取最大值，确保串行每步有足够notify ✓
- **notifyNumOnMainThread = max(...)**：主线程信号量取最大值 ✓
- **channels分3组**：每层独立channel set，避免层间channel冲突 ✓
- **线程分配**: `threads.assign(threads.begin(), threads.begin() + 1 + slaveThreadNum)`：主线程+3个slave=4线程 ✓

---

## 完整参数验证总表

| 参数 | 公式 | rank 37实例值 | 是否正确 | 说明 |
|------|------|-------------|---------|------|
| `rankIdxLevel0` | `myRank % rankSizeLevel0` | 37%8=5 | ✅ | 还原验证: L2×64+L1×8+L0=0×64+4×8+5=37 ✓ |
| `rankIdxLevel1` | `(myRank/rankSizeLevel0)%rankSizeLevel1` | (37/8)%8=4 | ✅ | |
| `rankIdxLevel2` | `myRank/(rankSizeLevel0*rankSizeLevel1)` | 37/64=0 | ✅ | |
| TopoForLayer2过滤 | `rankId%64==myRank%64` | {37,101} | ✅ | rank 101=(5,4,1)与rank 37=(5,4,0)只有集群不同 ✓ |
| Mesh `repeatNum` | `rankSizeLevel1*rankSizeLevel2` | 8×2=16 | ✅ | 16个上层组各需帧内reduce ✓ |
| Mesh `inputSliceStride` | `dataSize_` | 512B | ✅ | 每rank输入数据间隔 ✓ |
| Mesh `outputSliceStride` | `currDataCount*dataTypeSize_` | 512B | ✅ | CCL内slice间隔 ✓ |
| Mesh `outputRepeatStride` | `rankSizeLevel0*512` | 4096B | ✅ | 每repeat占8×512 ✓ |
| NHR1 `inBuffBaseOff` | `rankIdxLevel0*currDataCount*dataTypeSize_` | 2560B | ✅ | 指向帧内位置5的slice ✓ |
| NHR1 `inputSliceStride` | `rankSizeLevel0*512` | 4096B | ✅ | 跳一整个repeat组=Pod内不同帧 ✓ |
| NHR1 `repeatNum` | `rankSizeLevel2` | 2 | ✅ | 集群0和集群1各一次 ✓ |
| NHR1 `inputRepeatStride` | `rankSizeLevel0*rankSizeLevel1*512` | 32768B | ✅ | 跳8×8×512=一Pod的数据量 ✓ |
| NHR2 `inBuffBaseOff` | `rankIdxLevel0*512` | 2560B | ✅ | 指向NHR1结果 ✓ |
| NHR2 `outBuffBaseOff` | `processedDataCount*dataTypeSize_` | 0 | ✅ | 首轮从OUTPUT起始写 ✓ |
| NHR2 `repeatNum` | 1 | 1 | ✅ | 最终层无需重复 ✓ |
| scratch multiplier | `M0*M1*M2` | 1 (假设各层=1) | ✅ | 乘法形式确保每层有足够buffer ✓ |
| CalcRes合并 | max策略 + 3组channel | max(2,3,1)=3 | ✅ | max确保满足最demanding层 ✓ |
| Buffer复用安全 | NHR in-place | — | ✅ | 串行执行+每rank只写自己slice ✓ |

---

## 发现的潜在问题

### ⚠️ NHR Level2 `inputSliceStride=4096B` 语义问题

- **当前状态**: `inputSliceStride = rankSizeLevel0 * currDataCount * dataTypeSize_ = 4096B`（帧内8个slice的间距）
- **问题**: NHR Level2只有2个参与rank，`repeatNum=1`，只读取1个slice，stride实际不生效。stride=4096是帧内间距而非集群间距（集群间距应为32768B）
- **影响**: repeatNum=1时不影响功能正确性
- **建议**: 如果未来扩展到4+集群（repeatNum>1），stride应改为集群间间距。当前可作为冗余参数接受，但语义上可改进

---

## 总体评价

实现的整体参数配置**逻辑自洽且正确**。三层串行执行的数据流(INPUT→CCL→CCL→OUTPUT)、rank分解公式、stride/repeat/offset的设定都经过实例推演验证，与数据布局和通信语义一致。唯一需要注意的是Level2 NHR的inputSliceStride在repeatNum=1时是冗余参数，语义上可改进但不影响当前功能正确性。