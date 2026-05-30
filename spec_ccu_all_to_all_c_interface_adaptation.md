# CcuAllToAllMesh1DConcurrent C接口适配规格文档

## 一、背景与目标

### 1.1 背景

PR #782 正在将HCCL的CCU算子从C++接口（`hcomm`命名空间）迁移到C语言接口（`ccu_control_api`）。当前 `CcuAllToAllMesh1DConcurrent` 算子仍使用旧C++接口，需要适配到新C接口。

### 1.2 目标

将 `CcuAllToAllMesh1DConcurrent` 的完整调用链从C++接口迁移到C接口，包括：
- Kernel层：`CcuKernelAllToAllMesh1DMultiJetty` 类 → C风格自由函数
- Template层：`CcuTempAllToAllMesh1dMultiJetty` 类的 `CalcRes`/`KernelRun`/`FastLaunch` 适配
- Executor层：`InsV2AllToAllConcurrentExecutor` 的CCU相关调用适配

### 1.3 注册入口

```cpp
// ins_v2_all_to_all_concurrent_executor.cc:569-575
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLTOALL,
                                CcuAllToAllMesh1DConcurrent,
                                InsV2AllToAllConcurrentExecutor,
                                TopoMatchUBX,
                                CcuTempAllToAllMesh1dMultiJetty,
                                CcuTempAllToAllMesh1dMultiJetty);
```

---

## 二、涉及文件清单

| 层级 | 文件路径 | 变更类型 |
|------|----------|----------|
| Kernel层 | `src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.h` | 重写 |
| Kernel层 | `src/ops/all_to_all_v/template/ccu/kernel/ccu_kernel_all_to_all_mesh1d_multi_jetty.cc` | 重写 |
| Template层 | `src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh1d_multi_jetty.h` | 修改 |
| Template层 | `src/ops/all_to_all_v/template/ccu/ccu_temp_all_to_all_mesh1d_multi_jetty.cc` | 修改 |
| Executor层 | `src/ops/all_to_all_v/executor/ins_v2_all_to_all_concurrent_executor.cc` | 修改 |
| Executor层 | `src/ops/all_to_all_v/executor/ins_v2_all_to_all_concurrent_executor.h` | 无需修改 |
| 公共层 | `src/ops/op_common/template/ccu/ccu_kernel_alg_base.h` | 已由PR#782完成 |
| 公共层 | `src/ops/op_common/template/ccu/ccu_kernel_alg_base.cc` | 已由PR#782完成 |
| 公共层 | `src/ops/op_common/ccu_log.h` | 已由PR#782新增 |
| 公共层 | `src/ops/op_common/inc/alg_param.h` | 已由PR#782修改 |

---

## 三、旧接口依赖分析

### 3.1 调用链全景

```
CcuAllToAllMesh1DConcurrent (注册入口)
  └─ InsV2AllToAllConcurrentExecutor (executor层)
       ├─ CalcRes() → CcuTempAllToAllMesh1dMultiJetty::CalcRes() (template层)
       │    └─ 创建 CcuKernelInfo（旧：creator + kernelArg(shared_ptr<CcuKernelArg>)）
       ├─ OrchestrateLoop() → CcuTempAllToAllMesh1dMultiJetty::KernelRun() (template层)
       │    └─ HcclCcuKernelLaunch() + 创建 CcuTaskArg (旧：hcomm::CcuTaskArg子类)
       ├─ FastLaunch() → CcuTempAllToAllMesh1dMultiJetty::FastLaunch() (template层)
       │    └─ HcclCcuKernelLaunch() + 创建 CcuTaskArg
       └─ FastLaunchSaveCtx() → 保存CcuFastLaunchCtx
```

### 3.2 旧C++接口依赖清单

#### Kernel层（`ccu_kernel_all_to_all_mesh1d_multi_jetty.h/cc`）

| 旧接口/类型 | 所属命名空间/头文件 | 用途 |
|-------------|---------------------|------|
| `hcomm::CcuKernelArg` | `ccu_kernel.h` | Kernel参数基类，`CcuKernelArgAllToAllMesh1DMultiJetty` 继承自它 |
| `hcomm::CcuKernelSignature` | `ccu_kernel.h` | Kernel签名，用于区分不同kernel |
| `hcomm::CcuTaskArg` | `ccu_kernel.h` | Task参数基类，`CcuTaskArgAllToAllMesh1DMultiJetty` 继承自它 |
| `CcuKernelAlgBase` (继承`hcomm::CcuKernel`) | `ccu_kernel_alg_base.h` | Kernel算法基类 |
| `hcomm::CcuRep::Variable` | `ccu_kernel.h` | CCU变量 |
| `hcomm::CcuRep::LocalAddr` | `ccu_kernel.h` | 本地地址 |
| `hcomm::CcuRep::RemoteAddr` | `ccu_kernel.h` | 远端地址 |
| `hcomm::CcuRep::CompletedEvent` | `ccu_kernel.h` | 完成事件 |
| `CcuKernel::CreateVariable()` | `ccu_kernel.h` | 创建变量 |
| `CcuKernel::CreateVariable(ch, id, &var)` | `ccu_kernel.h` | 从channel资源创建变量 |
| `CcuKernel::CreateLocalAddr()` | `ccu_kernel.h` | 创建本地地址 |
| `CcuKernel::CreateRemoteAddr()` | `ccu_kernel.h` | 创建远端地址 |
| `CcuKernel::CreateCompletedEvent()` | `ccu_kernel.h` | 创建完成事件 |
| `CcuKernel::Load(var)` | `ccu_kernel.h` | 加载变量为kernel参数 |
| `CcuKernel::WriteNb()` | `ccu_kernel.h` | 非阻塞写 |
| `CcuKernel::NotifyRecord()` | `ccu_kernel.h` | 记录通知 |
| `CcuKernel::NotifyWait()` | `ccu_kernel.h` | 等待通知 |
| `CcuKernel::WaitEvent()` | `ccu_kernel.h` | 等待事件 |
| `CcuKernel::RecordEvent()` | `ccu_kernel.h` | 记录事件 |
| `CcuKernel::LocalCopyNb()` | `ccu_kernel.h` | 非阻塞本地拷贝 |
| `CcuKernel::CCU_IF()` | `ccu_kernel.h` | 条件执行宏 |
| `CcuKernelAlgBase::GroupCopy()` | `ccu_kernel_alg_base.h` | Group级别拷贝 |
| `CcuKernelAlgBase::CalGoSize()` | `ccu_kernel_alg_base.h` | 计算Group操作大小 |
| `CcuKernelAlgBase::CreateGroupOpSize()` | `ccu_kernel_alg_base.h` | 创建Group操作大小 |
| `CcuKernelAlgBase::AllocGoResource()` | `ccu_kernel_alg_base.h` | 分配Group资源 |
| `GenerateCcuKernelSignature()` | `ccu_kernel_utils.h` | 生成kernel签名 |

#### Template层（`ccu_temp_all_to_all_mesh1d_multi_jetty.cc`）

| 旧接口/类型 | 用途 |
|-------------|------|
| `kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {...}` | C++ lambda作为kernel工厂 |
| `kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1DMultiJetty>(...)` | C++智能指针管理kernel参数 |
| `HcclCcuKernelLaunch(comm, thread, kernelHandle, taskArgPtr)` | 旧版kernel launch |
| `hcomm::CcuRep::GetTokenInfo()` | 获取token信息 |
| `std::unique_ptr<hcomm::CcuTaskArg>` | Task参数智能指针 |

#### Executor层（`ins_v2_all_to_all_concurrent_executor.cc`）

| 旧接口/类型 | 用途 |
|-------------|------|
| `#include "hccl_ccu_res.h"` | CCU资源头文件 |
| `resReq0.ccuKernelInfos[0].channels = channelDescs0` | 旧版channel赋值（vector） |

---

## 四、C接口适配方案

### 4.1 适配模式参考

以PR #782中已完成的 `CcuAllGatherMesh1DKernel` 为参考模式，核心变化为：

| 维度 | 旧模式（C++） | 新模式（C） |
|------|---------------|-------------|
| Kernel入口 | `class XxxKernel : public CcuKernelAlgBase` + `Algorithm()` 虚函数 | `CcuResult XxxKernel(CcuKernelArg arg)` 自由函数 |
| Kernel参数 | `class XxxArg : public hcomm::CcuKernelArg` + `GetKernelSignature()` | `struct XxxArg : CcuKernelArgBase` (POD风格) |
| Task参数 | `class XxxTaskArg : public hcomm::CcuTaskArg` | 直接使用 `std::vector<uint64_t>` 或C结构体 |
| 变量类型 | `hcomm::CcuRep::Variable` | `ccu::Variable` |
| 地址类型 | `hcomm::CcuRep::LocalAddr/RemoteAddr` | `ccu::LocalAddr/RemoteAddr` |
| 事件类型 | `hcomm::CcuRep::CompletedEvent` | `ccu::Event` |
| 创建变量 | `CreateVariable()` | `ccu::CreateVariable()` |
| 从channel获取资源 | `CreateVariable(ch, id, &var)` | `ccu::GetResByChannel<ccu::Variable>(ch, id)` |
| 加载参数 | `Load(var)` | `ccu::LoadArg(var, argId++)` |
| 通知 | `NotifyRecord/NotifyWait(ch, idx, bit)` | `ccu::NotifyRecord/NotifyWait(ch, idx, bit)` |
| 写操作 | `WriteNb(ch, dst, src, len, event)` | `ccu::WriteNb(ch, dst, src, len, event)` |
| 等待事件 | `WaitEvent(event)` | `ccu::WaitEvent(event)` |
| 记录事件 | `RecordEvent(event)` | `ccu::RecordEvent(event)` |
| Group操作 | `GroupCopy(dst, src, goSize)` (成员方法) | `GroupCopy(ctx, dst, src, goSize)` (自由函数) |
| 返回值 | `HcclResult` | `CcuResult` |
| 错误检查 | `CHK_RET()` | `CCU_CHK_RET()` |
| 条件执行 | `CCU_IF(cond) { ... }` | `CCU_IF(cond) { ... }` (保持不变) |
| 上下文 | 类成员变量 | `struct XxxContext : CcuKernelCtxBase` |

### 4.2 Kernel层适配

#### 4.2.1 头文件重写：`ccu_kernel_all_to_all_mesh1d_multi_jetty.h`

**旧结构**：
```cpp
class CcuKernelArgAllToAllMesh1DMultiJetty : public hcomm::CcuKernelArg { ... };
class CcuTaskArgAllToAllMesh1DMultiJetty : public hcomm::CcuTaskArg { ... };
class CcuKernelAllToAllMesh1DMultiJetty : public CcuKernelAlgBase { ... };
```

**新结构**：
```cpp
struct CcuKernelArgAllToAllMesh1DMultiJetty : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    std::vector<uint32_t>                   jettyNums;
};

struct AllToAllMesh1DMultiJettyContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllMesh1DMultiJetty *arg;

    ccu::Variable input;                              // 本rank的input变量
    ccu::Variable output;                             // 本rank的output变量
    ccu::Variable token;                              // 本rank的token变量
    std::vector<ccu::Variable> peerInput;             // 各peer的input变量
    std::vector<ccu::Variable> peerOutput;            // 各peer的output变量
    std::vector<ccu::Variable> peerToken;             // 各peer的token变量
    ccu::Variable sliceSize;
    ccu::Variable srcStride;
    ccu::Variable srcOffset;
    ccu::Variable dstOffset;
    GroupOpSizeVars goSize;
    std::vector<ccu::Variable> jettySlice;
    std::vector<ccu::Variable> jettySliceTail;
    std::vector<ccu::Event>    eventList;
};

CcuResult CcuAllToAllMesh1DMultiJettyKernel(CcuKernelArg arg);
```

**关键变更点**：
1. `CcuKernelArgAllToAllMesh1DMultiJetty` 从 `hcomm::CcuKernelArg` 子类改为 `CcuKernelArgBase` 子类
2. 删除 `GetKernelSignature()` 方法（C接口不再需要签名机制）
3. 删除 `CcuTaskArgAllToAllMesh1DMultiJetty` 类（Task参数改为直接传 `uint64_t` 数组）
4. 删除 `CcuKernelAllToAllMesh1DMultiJetty` 类，改为自由函数 `CcuAllToAllMesh1DMultiJettyKernel`
5. 新增 `AllToAllMesh1DMultiJettyContext` 上下文结构体，替代原类成员变量
6. 删除 `#include "ccu_kernel.h"`，改为 `#include "ccu_kernel_alg_base.h"` + `#include "ccu_log.h"`

#### 4.2.2 源文件重写：`ccu_kernel_all_to_all_mesh1d_multi_jetty.cc`

**旧 `Algorithm()` 方法**的6步流程保持不变，但每步实现需适配C接口：

```
Algorithm() → CcuAllToAllMesh1DMultiJettyKernel(arg)
  ├─ InitResource()  → static CcuResult InitResource(ctx)
  ├─ LoadArgs()      → static CcuResult LoadArgs(ctx)
  ├─ PreSync()       → static CcuResult PreSync(ctx)
  ├─ CalcAddrs()     → static CcuResult CalcAddrs(ctx)
  ├─ DoAllToAll()    → static CcuResult DoAllToAll(ctx)
  └─ PostSync()      → static CcuResult PostSync(ctx)
```

**各步骤适配细节**：

##### InitResource

| 旧代码 | 新代码 |
|--------|--------|
| `CreateVariable()` | `ccu::CreateVariable()` |
| `CreateVariable(ch, id, &var)` | `ccu::GetResByChannel<ccu::Variable>(ch, id)` |
| `CreateCompletedEvent()` | `ccu::CreateEvent()` |
| `CreateGroupOpSize()` | `GroupOpSizeVars{}` (直接初始化) |
| `channels_.size() == 0` | `arg->channelCount == 0` |
| `CHK_RET(...)` | `CCU_CHK_RET(...)` |
| `return HcclResult::HCCL_SUCCESS` | `return CCU_SUCCESS` |

##### LoadArgs

| 旧代码 | 新代码 |
|--------|--------|
| `Load(input_[rankId_])` | `ccu::LoadArg(ctx.input, argId++)` |
| `Load(output_[rankId_])` | `ccu::LoadArg(ctx.output, argId++)` |
| `Load(token_[rankId_])` | `ccu::LoadArg(ctx.token, argId++)` |
| `Load(sliceSize_)` | `ccu::LoadArg(ctx.sliceSize, argId++)` |
| `Load(srcStride_)` | `ccu::LoadArg(ctx.srcStride, argId++)` |
| `Load(srcOffset_)` | `ccu::LoadArg(ctx.srcOffset, argId++)` |
| `Load(dstOffset_)` | `ccu::LoadArg(ctx.dstOffset, argId++)` |
| `Load(groupOpSize_)` | `ccu::LoadArg(ctx.goSize.addrOffset, argId++)` 等4个字段 |
| `Load(jettySlice_[i])` | `ccu::LoadArg(ctx.jettySlice[i], argId++)` |
| `Load(jettySliceTail_[i])` | `ccu::LoadArg(ctx.jettySliceTail[i], argId++)` |

##### PreSync

| 旧代码 | 新代码 |
|--------|--------|
| `NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID)` | `ccu::WriteVariableWithNotify(ch, ctx.input, INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID)` |
| `NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID)` | `ccu::WriteVariableWithNotify(ch, ctx.output, OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID)` |
| `NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID)` | `ccu::WriteVariableWithNotify(ch, ctx.token, TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID)` |
| `NotifyWait(channel, CKE_IDX_0, allBit)` | `ccu::NotifyWait(ch, CKE_IDX_0, allBit)` |

##### CalcAddrs

| 旧代码 | 新代码 |
|--------|--------|
| `CreateLocalAddr()` | `ccu::LocalAddr{}` (直接构造) |
| `CreateRemoteAddr()` | `ccu::RemoteAddr{}` (直接构造) |
| `localSrc_.addr = srcOffset_` | `localSrc.addr = ctx.srcOffset` |
| `localSrc_.token = token_[rankIdx]` | `localSrc.token = ctx.peerToken[rankIdx]` |
| `remoteDst_[rankIdx].addr = output_[rankIdx]` | `remoteDst[rankIdx].addr = ctx.peerOutput[rankIdx]` |

##### DoAllToAll

| 旧代码 | 新代码 |
|--------|--------|
| `WriteNb(channel, remoteDst, remoteSrc, sliceLength, event)` | `ccu::WriteNb(ch, remoteDst, remoteSrc, sliceLength, event)` |
| `RecordEvent(event)` | `ccu::RecordEvent(event)` |
| `WaitEvent(event)` | `ccu::WaitEvent(event)` |
| `GroupCopy(localDst_, localSrc_, groupOpSize_)` | `GroupCopy(ctx, localDst, localSrc, ctx.goSize)` |
| `eventList_[r].SetMask(mask)` | `eventList[r].mask = mask` |

##### PostSync

| 旧代码 | 新代码 |
|--------|--------|
| `NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID)` | `ccu::NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID)` |
| `NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID)` | `ccu::NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID)` |

##### Kernel入口函数

```cpp
CcuResult CcuAllToAllMesh1DMultiJettyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllMesh1DMultiJetty *>(arg);

    AllToAllMesh1DMultiJettyContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(CalcAddrs(ctx));
    CCU_CHK_RET(DoAllToAll(ctx));
    CCU_CHK_RET(PostSync(ctx));

    return CCU_SUCCESS;
}
```

### 4.3 Template层适配

#### 4.3.1 `ccu_temp_all_to_all_mesh1d_multi_jetty.cc` 修改

##### CalcRes 适配

**旧代码**：
```cpp
CcuKernelInfo kernelInfo;
kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
    return std::make_unique<CcuKernelAllToAllMesh1DMultiJetty>(arg);
};
kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1DMultiJetty>(
    templateRankSize_, myRank_, param, subCommRanks_, jettyNums_);
```

**新代码**：
```cpp
CcuKernelInfo kernelInfo;
strcpy(kernelInfo.kernelFuncName, "CcuAllToAllMesh1DMultiJettyKernel");
kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllToAllMesh1DMultiJettyKernel);

auto kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1DMultiJetty>();
kernelArg->rankSize = templateRankSize_;
kernelArg->rankId = myRank_;
kernelArg->opParam = param;
kernelArg->subCommRanks = subCommRanks_;
kernelArg->jettyNums = jettyNums_;
kernelInfo.setKernelArg(kernelArg);
```

**变更要点**：
1. `creator` lambda → `kernelFuncName` + `kernelFunc` 函数指针
2. `kernelArg` 从 `std::shared_ptr<hcomm::CcuKernelArg>` → 通过 `setKernelArg()` 设置为 `CcuKernelArgBase*`
3. `CcuKernelArgAllToAllMesh1DMultiJetty` 从 `hcomm::CcuKernelArg` 子类 → `CcuKernelArgBase` 子类

##### KernelRun 适配

**旧代码**：
```cpp
std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllMesh1DMultiJetty>(
    inputAddr, outputAddr, sliceSize, jettySlice, jettySliceTail, token, srcOffset, dstOffset, srcStride);
void* taskArgPtr = static_cast<void*>(taskArg.get());
HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);
```

**新代码**：
```cpp
LoopGroupConfig config{};
config.msInterleave = CCU_MS_INTERLEAVE;
config.loopCount    = CCU_MS_DEFAULT_LOOP_COUNT;
config.memSlice     = CCU_MS_SIZE;
auto goSize = CalGoSize(sliceSize, config);

std::vector<uint64_t> taskArgs = {
    inputAddr, outputAddr, token, sliceSize, srcStride, srcOffset, dstOffset,
    goSize[0], goSize[1], goSize[2], goSize[3]
};
for (uint32_t i = 0; i < templateRankSize_; i++) {
    taskArgs.push_back(jettySlice[i]);
}
for (uint32_t i = 0; i < templateRankSize_; i++) {
    taskArgs.push_back(jettySliceTail[i]);
}
uint64_t argSize = 8;

CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0],
    templateResource.ccuKernels[0], taskArgs.data(), argSize);
if (launchRet != CCU_SUCCESS) {
    HCCL_ERROR("[CcuTempAllToAllMesh1dMultiJetty::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
    return ConvertCcuToHccl(launchRet);
}
```

**变更要点**：
1. `CcuTaskArgAllToAllMesh1DMultiJetty` → `std::vector<uint64_t>` 直接传参
2. `HcclCcuKernelLaunch(comm, thread, handle, taskArgPtr)` → `HcommCcuKernelLaunch(thread, handle, taskArgs.data(), argSize)`
3. `CalGoSize(sliceSize)` → `CalGoSize(sliceSize, config)` (新接口需要传入config)
4. 返回值检查从 `HcclResult` → `CcuResult` + `ConvertCcuToHccl()`
5. `FillCachedArgs` 需适配新的参数列表

##### FastLaunch 适配

**旧代码**：
```cpp
CcuTaskArgAllToAllMesh1DMultiJetty taskArg(
    PointerToAddr(buffInfo_.inputPtr) + args[0],
    PointerToAddr(buffInfo_.outputPtr) + args[1],
    args[2], jettySlice, jettySliceTail, args[3], args[4], args[5], args[6]);
void* taskArgPtr = static_cast<void*>(&taskArg);
CHK_RET(HcclCcuKernelLaunch(param.hcclComm, tempFastLaunchCtx.threads[0],
    tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgPtr));
```

**新代码**：
```cpp
uint64_t *cachedArgs = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
cachedArgs[0] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + cachedArgs[inputOffsetIdx];
cachedArgs[1] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + cachedArgs[outputOffsetIdx];

void *taskArgs = reinterpret_cast<void*>(cachedArgs);
uint64_t argSize = 8;
CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
    tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle, taskArgs, argSize);
if (launchRet != CCU_SUCCESS) {
    HCCL_ERROR("[CcuTempAllToAllMesh1dMultiJetty::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
    return ConvertCcuToHccl(launchRet);
}
```

**变更要点**：
1. 不再构造 `CcuTaskArgAllToAllMesh1DMultiJetty` 对象
2. 直接修改 `cachedArgs` 中的地址偏移，然后作为 `void*` 传入
3. `HcclCcuKernelLaunch` → `HcommCcuKernelLaunch`（参数签名不同）

##### GetToken 适配

**旧代码**：
```cpp
token = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.inputPtr),
                                     static_cast<uint64_t>(buffInfo_.inputSize));
```

**新代码**：
```cpp
// CcuAlgTemplateBase::GetToken() 已有实现，无需修改
// 内部需将 hcomm::CcuRep::GetTokenInfo 替换为 ccu::GetTokenInfo
CHK_RET(GetToken(buffInfo_, token));
```

#### 4.3.2 `ccu_temp_all_to_all_mesh1d_multi_jetty.h` 修改

- 新增 `#include "ccu_kernel_alg_base.h"` 和 `#include "ccu_log.h"`
- 删除对 `ccu_kernel.h` 的间接依赖
- 类声明本身无需修改（仍继承 `CcuAlgTemplateBase`）

### 4.4 Executor层适配

#### 4.4.1 `ins_v2_all_to_all_concurrent_executor.cc` 修改

##### 头文件替换

```cpp
// 旧
#include "hccl_ccu_res.h"

// 新
#include "ccu_control_api.h"
#include "ccu_log.h"
```

##### CalcRes 中 channel 赋值适配

**旧代码**：
```cpp
resReq0.ccuKernelInfos[0].channels = channelDescs0;
resReq1.ccuKernelInfos[0].channels = channelDescs1;
```

**新代码**：
```cpp
// channel信息已通过 CcuKernelArgBase::channels 数组传递
// 在 CalcRes 阶段由 template 的 CalcRes 设置到 kernelArg 中
// 此处无需额外赋值
```

**注意**：需要确认 channel 的传递路径。在新的C接口中，channel信息通过 `CcuKernelArgBase::channels[]` 数组传递，由 `HcclGetChannelForCcu()` 统一填充。因此 executor 层的 `resReq0.ccuKernelInfos[0].channels = channelDescs0` 需要保留，因为 `HcclGetChannelForCcu` 仍从 `kernelInfo.channels` 读取。

##### FastLaunch 适配

Executor 层的 `FastLaunch` 和 `FastLaunchSaveCtx` 方法本身不直接调用 `hcomm` 接口，主要依赖 `TemplateFastLaunchCtx` 中的 `ccuKernelSubmitInfos`，该结构在新接口中保持兼容。但需注意：

1. `HcclCcuKernelLaunch` → `HcommCcuKernelLaunch` 的签名差异
2. `CcuFastLaunchCtx` 中 `ccuKernelNum` 和 `ccuKernelSubmitInfos` 的布局是否兼容

---

## 五、参数映射对照表

### 5.1 Kernel Arg 参数映射

| 旧字段（C++类成员） | 新字段（C结构体） | 类型变化 |
|---------------------|-------------------|----------|
| `rankSize_` | `rankSize` | `uint64_t` 不变 |
| `rankId_` | `rankId` | `uint32_t` 不变 |
| `opParam_` | `opParam` | `OpParam` 不变 |
| `subCommRanks_` | `subCommRanks` | `std::vector<std::vector<uint32_t>>` 不变 |
| `jettyNums_` | `jettyNums` | `std::vector<uint32_t>` 不变 |
| `channels` (继承自CcuKernelArg) | `channels[]` + `channelCount` (继承自CcuKernelArgBase) | `std::vector` → C数组 |

### 5.2 Task Arg 参数映射

旧方式：`CcuTaskArgAllToAllMesh1DMultiJetty` 类对象

新方式：`std::vector<uint64_t>` 数组，布局如下：

| 索引 | 参数 | 旧字段 |
|------|------|--------|
| 0 | inputAddr | `inputAddr_` |
| 1 | outputAddr | `outputAddr_` |
| 2 | token | `token_` |
| 3 | sliceSize | `sliceSize_` |
| 4 | srcStride | `srcStride_` |
| 5 | srcOffset | `srcOffset_` |
| 6 | dstOffset | `dstOffset_` |
| 7 | goSize[0] (addrOffset) | `CalGoSize()` 返回值 |
| 8 | goSize[1] (loopIterNum) | `CalGoSize()` 返回值 |
| 9 | goSize[2] (loopExtendNum) | `CalGoSize()` 返回值 |
| 10 | goSize[3] (tailSize) | `CalGoSize()` 返回值 |
| 11 ~ 11+rankSize-1 | jettySlice[0..rankSize-1] | `jettySlice_` |
| 11+rankSize ~ 11+2*rankSize-1 | jettySliceTail[0..rankSize-1] | `jettySliceTail_` |

### 5.3 GeneArgs 废弃

旧模式中 `GeneArgs()` 方法将 `CcuTaskArg` 转换为 `std::vector<uint64_t>` 供框架缓存。新模式中，Task参数直接以 `uint64_t` 数组形式传入kernel launch，`GeneArgs()` 不再需要。

---

## 六、CMake适配

### 6.1 头文件搜索路径

`src/CMakeLists.txt` 中需确认已添加（PR#782已添加）：
```cmake
${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu_new
```

### 6.2 源文件编译

`src/ops/all_to_all_v/template/ccu/CMakeLists.txt` 中已包含 `ccu_kernel_all_to_all_mesh1d_multi_jetty.cc`，无需修改。

### 6.3 模块启用

`src/ops/CMakeLists.txt` 中需取消 `add_subdirectory(all_to_all_v)` 的注释：
```cmake
# 旧（PR#782中已注释）
# add_subdirectory(all_to_all_v)

# 新（适配完成后恢复）
add_subdirectory(all_to_all_v)
```

`src/CMakeLists.txt` 中需恢复 all_to_all_v 相关 executor 源文件的编译。

---

## 七、验证要点

### 7.1 功能验证

| 验证项 | 验证方法 |
|--------|----------|
| AllToAll Mesh1D MultiJetty 基本功能 | 单卡/多卡 AllToAll 操作正确性 |
| Jetty切分逻辑 | 不同jetty数量下的数据正确性 |
| GroupCopy | 本地拷贝数据一致性 |
| PreSync/PostSync | 多rank间同步正确性 |
| FastLaunch路径 | loopTimes==1 时的快速下发路径 |
| Concurrent执行 | 两个template并发执行的正确性 |

### 7.2 回归验证

- 确保适配后 `CcuAllToAllMesh1DConcurrent` 的行为与旧接口完全一致
- 对比适配前后的kernel launch参数（taskArgs内容）是否一致
- 验证 `CalGoSize` 新接口（需传入config）的返回值与旧接口一致

### 7.3 编译验证

- 确认无 `hcomm::CcuKernelArg`、`hcomm::CcuTaskArg`、`hcomm::CcuKernel` 等旧类型的引用
- 确认无 `ccu_kernel.h` 的include
- 确认无 `GenerateCcuKernelSignature` 的调用

---

## 八、风险与注意事项

### 8.1 高风险项

1. **Task参数布局变更**：旧模式通过 `CcuTaskArg` 对象传参，新模式通过 `uint64_t` 数组传参。Kernel内部解析参数的索引必须与Template层构造参数的顺序严格一致，否则会导致数据错乱
2. **channel传递路径**：新接口中channel通过 `CcuKernelArgBase::channels[]` C数组传递，需确认 `HcclGetChannelForCcu` 正确填充

### 8.2 中风险项

3. **`CcuAlgTemplateBase::GetToken` 内部实现**：当前使用 `hcomm::CcuRep::GetTokenInfo`，需替换为 `ccu::GetTokenInfo`
4. **`GroupCopy` 签名变更**：从成员方法变为自由函数，参数从 `CcuRep::LocalAddr` 变为 `ccu::LocalAddr`，需确认类型兼容

### 8.3 低风险项

5. **`FillCachedArgs` 适配**：需根据新的taskArgs布局调整缓存参数的索引
6. **调试日志**：适配过程中可临时增加调试日志，但合入前必须清理

---

## 九、实施步骤

| 步骤 | 内容 | 依赖 |
|------|------|------|
| 1 | 重写 `ccu_kernel_all_to_all_mesh1d_multi_jetty.h`：定义新结构体和kernel函数声明 | PR#782的 `ccu_kernel_alg_base.h` |
| 2 | 重写 `ccu_kernel_all_to_all_mesh1d_multi_jetty.cc`：实现C风格kernel函数 | 步骤1 |
| 3 | 修改 `ccu_temp_all_to_all_mesh1d_multi_jetty.cc`：适配CalcRes/KernelRun/FastLaunch | 步骤1-2 |
| 4 | 修改 `ins_v2_all_to_all_concurrent_executor.cc`：头文件替换和channel赋值适配 | 步骤3 |
| 5 | 恢复CMake编译：取消all_to_all_v模块的注释 | 步骤4 |
| 6 | 编译验证 + 功能测试 | 步骤5 |
