# PR #782 分析报告：HCCL CCU C语言接口适配

## 一、PR 概述

| 项目 | 内容 |
|------|------|
| PR地址 | https://gitcode.com/cann/hccl/pull/782 |
| 分支名 | `c_new_hcomm` |
| 变更文件数 | 60 |
| 代码变更量 | +4079 / -3807 |
| 核心目标 | 将HCCL中CCU算子从C++接口（`hcomm`命名空间）迁移到C语言接口（`ccu_control_api`） |

---

## 二、变更背景与动机

HCCL（Huawei Collective Communication Library）中的CCU（Collective Communication Unit）算子原本基于 `hcomm::CcuKernel` C++类继承体系实现，存在以下问题：

1. **C++运行时依赖重**：C++的类继承、智能指针、异常等特性增加了CCU底层驱动的耦合度
2. **跨语言互操作性差**：C++ ABI兼容性问题限制了其他语言绑定的可能性
3. **部署复杂**：C++动态库的ABI兼容性在不同编译器版本间容易出问题

该PR将CCU相关接口从C++面向对象风格迁移到C语言过程式风格，实现底层通信接口的轻量化与解耦。

---

## 三、核心变更详解

### 3.1 新增 CCU 日志与错误码转换模块

**文件**: `src/ops/op_common/ccu_log.h`（新增）

提供两个核心工具：

- **`CCU_CHK_RET` 宏**：统一检查CCU函数返回值，失败时自动记录错误日志并返回，减少重复的错误检查样板代码
- **`ConvertCcuToHccl` 函数**：将 `CcuResult` 错误码映射回 `HcclResult`，实现两套错误码体系的桥接，覆盖了 SUCCESS、PARA、PTR、MEMORY、INTERNAL、NOT_SUPPORT 等20+种错误码的映射

### 3.2 CCU Kernel 注册机制重构

**文件**: `src/ops/op_common/op_common.cc`

`HcclGetCcuKernel` 函数的注册流程从C++风格改为C风格：

**旧方式（C++）**：
```cpp
// 使用 creator 函数对象 + shared_ptr 管理参数
void* kernelArgPtr = static_cast<void*>(kernelInfo.kernelArg.get());
void* creatorPtr = static_cast<void*>(&kernelInfo.creator);
HcclCcuKernelRegister(comm, &handle, creatorPtr, kernelArgPtr);
HcclCcuKernelRegisterFinish(comm);
```

**新方式（C）**：
```cpp
// 分步注册：开始 → 逐个注册 → 结束
HcommCcuKernelRegisterStart(insHandle);
HcommCcuKernelRegister(insHandle, kernelFuncName,
    kernelFunc, kernelArg, &kernelHandle);
HcommCcuKernelRegisterEnd(insHandle);
```

关键差异：
- 新增 `HcclCommQueryCcuIns` 查询CCU实例，校验实例数量
- 引入 `CcuInsHandle` 实例句柄概念，注册操作绑定到具体实例
- kernel注册从"函数对象+智能指针"变为"函数名+C函数指针+void*参数"
- 注册结果直接返回 `CcuResult`，通过 `ConvertCcuToHccl` 转换后返回

### 3.3 `CcuKernelInfo` 结构体重构

**文件**: `src/ops/op_common/inc/alg_param.h`

| 字段 | 旧（C++） | 新（C） |
|------|-----------|---------|
| kernel标识 | `hcomm::KernelCreator creator` | `char kernelFuncName[64]` + `void* kernelFunc` |
| kernel参数 | `std::shared_ptr<hcomm::CcuKernelArg>` | `void* kernelArg` + `std::shared_ptr<CcuKernelArgBase>` 管理生命周期 |
| 通道信息 | `std::vector<ChannelHandle>`（在kernelArg中） | `ChannelHandle channels[CCU_MAX_RANK_SIZE]`（C数组） |

新增 `CcuKernelArgBase` 结构体，使用固定大小C数组（`CCU_MAX_RANK_SIZE = 16`）替代 `std::vector`，并通过 `setKernelArg<T>()` 模板方法统一管理智能指针生命周期。

通道赋值方式也相应变更：
```cpp
// 旧：vector赋值
kernelInfo.kernelArg->channels = kernelChannels;

// 新：C数组逐元素赋值 + 计数
auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
for (u32 i = 0; i < channelNum; ++i) {
    kernelArgBase->channels[i] = kernelChannels[i];
}
kernelArgBase->channelCount = channelNum;
```

### 3.4 `CcuKernelAlgBase` 类彻底重构

**文件**: `src/ops/op_common/template/ccu/ccu_kernel_alg_base.h` 和 `.cc`

这是变更量最大的部分（约2000行变更），核心变化如下：

#### 3.4.1 从继承类改为自由函数

- **旧**：`CcuKernelAlgBase` 继承自 `hcomm::CcuKernel`，所有操作都是类成员方法
- **新**：不再继承，改为 `ops_hccl` 命名空间下的自由函数，状态通过 `CcuKernelCtxBase` 上下文结构传递

#### 3.4.2 数据结构C化

| 旧类型（C++） | 新类型（C风格） |
|----------------|-----------------|
| `CcuRep::Variable` | `ccu::Variable` |
| `CcuRep::CcuBuf` | `ccu::CcuBuffer` |
| `CcuRep::CompletedEvent` | `ccu::Event` |
| `CcuRep::LocalAddr/RemoteAddr` | `ccu::LocalAddr/RemoteAddr` |
| `CcuRep::Executor` | 移除 |
| `CcuRep::LoopCall` | 移除 |
| `std::vector<T>` | `ccu::Array<T>` 或 C数组 |

#### 3.4.3 新增核心数据结构

- **`CcuKernelCtxBase`**：运行时上下文，管理loop实体（`CcuLoopEntity`）、loopMap、资源分配状态等
- **`LoopGroupConfig`**：替代原 `GroupOpConfig`，配置loop组的步长、并行数、内存切片
- **`LoopGroupResource`**：替代原 `GroupOpSizeResource`，使用 `ccu::Array` 管理事件和缓冲区
- **`GroupOpSizeVars`**：替代原 `GroupOpSize`，存储loopGroup运算的偏移、迭代数、展开参数、尾数据
- **`GroupReduceVar`/`GroupBroadcastVar`/`GroupCopyVar`**：替代原来嵌套在类中的成员变量，使状态管理更清晰

#### 3.4.4 关键函数签名变更

```cpp
// 旧
HcclResult CcuKernelAlgBase::GroupBroadcast(
    const std::vector<ChannelHandle>& channels,
    std::vector<CcuRep::RemoteAddr> dst,
    CcuRep::LocalAddr src, GroupOpSize goSize);

// 新
CcuResult GroupBroadcast(CcuKernelCtxBase &ctx,
    const size_t channels[], uint32_t channelCount,
    ccu::LocalAddr localDst,
    std::vector<ccu::RemoteAddr> dst,
    ccu::LocalAddr src, GroupOpSizeVars goSize);
```

返回值从 `HcclResult` 改为 `CcuResult`，通道参数从 `std::vector` 改为C数组+长度。

### 3.5 各算子模板适配

#### AllGather 算子（6个文件）

| 文件 | 变更说明 |
|------|----------|
| `ccu_kernel_all_gather_mesh1d.cc/h` | 从C++类方法改为C风格函数，使用 `ccu::` API |
| `ccu_kernel_all_gather_mesh1d_mem2mem.cc/h` | 同上 |
| `ccu_kernel_all_gather_nhr1d_mem2mem.cc/h` | 同上 |
| `ccu_kernel_all_gather_nhr1d_multi_jetty_mem2mem.cc/h` | 同上 |

#### Reduce 算子（5个文件）

| 文件 | 变更说明 |
|------|----------|
| `ccu_kernel_reduce_mesh1d.cc/h` | 从C++类方法改为C风格函数 |
| `ccu_kernel_reduce_mesh1d_mem2mem.cc/h` | 同上 |
| `ccu_kernel_reduce_nhr1d_mem2mem.cc/h` | 同上 |

#### ReduceScatter 算子（3个文件）

| 文件 | 变更说明 |
|------|----------|
| `ccu_kernel_reduce_scatter_mesh1d.cc/h` | C接口适配 |
| `ccu_kernel_reduce_scatter_mesh1d_mem2mem.h` | 新增头文件 |

### 3.6 Reduce FastLaunch 修复

**文件**: `src/ops/reduce/executor/reduce_sole_executor.cc`

修复了多kernel场景下的线程数问题：

```cpp
// 旧：硬编码为1
u32 threadNum = 1;

// 新：使用实际submitInfo数量
u32 threadNum = templateAlgRes.submitInfos.size();
```

同时修复了线程句柄赋值：
```cpp
// 旧：仅赋值第一个
threads[0] = templateAlgRes.threads[0];

// 新：循环赋值所有
for (int i = 0; i < threadNum; i++) {
    threads[i] = templateAlgRes.threads[i];
}
```

### 3.7 CMake构建系统调整

**文件**: `src/CMakeLists.txt`、`src/ops/CMakeLists.txt`

- 新增头文件搜索路径：`pkg_inc/hcomm/ccu` 和 `pkg_inc/hcomm/ccu_new`
- **大量executor源文件被注释掉**，仅保留以下模块的编译：
  - `op_common`（公共模块）
  - `reduce_scatter`（ReduceScatter算子）
  - `all_gather`（AllGather算子）
  - `reduce`（Reduce算子）
  - `batch_send_recv`
  - `interface_graph_mode`

被禁用的模块：scatter、broadcast、all_reduce、all_to_all_v、send、recv、reduce_scatter_v、all_gather_v

### 3.8 其他清理

- 删除了 `GenerateCcuKernelSignature` 函数（C++接口特有的签名机制，C接口不再需要）
- `#include "ccu_kernel.h"` 被替换为 `#include "ccu_api.hpp"` 和 `#include "ccu_control_api.h"`
- `#include "hccl_ccu_res.h"` 被替换为 `#include "ccu_types.h"`
- `HcclLib.xml` 中多个AIV op文件被注释，仅保留 `hccl_aiv_reduce_scatter_op_910_95.o` 和 `hccl_aiv_reduce_op_910_95.o`

---

## 四、被禁用的算子/模板清单

### ReduceScatter 算子中被注释的变体

| 变体名 | 说明 |
|--------|------|
| `CcuReduceScatterMesh1DMem2Mem` | Mesh 1D Mem2Mem |
| `CcuReduceScatterNHR1DMem2Mem` | NHR 1D Mem2Mem |
| `CcuReduceScatterMeshMem2Mem1D2Die` | Mesh 1D 2Die Mem2Mem |
| `CcuReduceScatterMesh2Die` | Mesh 2Die |
| `CcuReduceScatterNhr1DMem2MemMultiJetty` | NHR 1D Multi-Jetty Mem2Mem |

### AllGather 算子中被注释的变体

| 变体名 | 说明 |
|--------|------|
| `CcuKernelAllGather2DiesMeshMem2Mem1D` | 2Dies Mesh Mem2Mem |
| `CcuKernelAllGather2DiesMesh1D` | 2Dies Mesh |

### 被禁用编译的模块

scatter、broadcast、all_reduce、all_to_all_v、send、recv、reduce_scatter_v、all_gather_v

---

## 五、潜在问题与风险

### 🔴 高风险

1. **大量模块被禁用**：scatter、broadcast、all_reduce、send、recv等模块的编译被注释掉，合入后这些功能将不可用。如果此PR合入master，将导致大量集合通信算子功能缺失，**必须确认这是否为预期行为，以及是否有分阶段合入的策略**

2. **`HcclLib.xml` 交付包不完整**：多个AIV op文件被注释，影响交付包的完整性，可能导致下游用户无法正常使用

### 🟡 中风险

3. **调试日志未清理**：`reduce_sole_executor.cc` 中包含 `xjhlog2`/`xjhlog3` 调试日志，不应合入主分支：
   ```cpp
   HCCL_INFO("[ReduceSoleExecutor][FastLaunch] xjhlog2 kernelHandle [%llu]", ...);
   HCCL_INFO("[ReduceSoleExecutor][FastLaunch] xjhlog3 kernelHandle [%llu]", ...);
   ```

4. **注释掉的代码过多**：大量旧代码以注释方式保留（而非删除），包括：
   - `ccu_kernel_alg_base.h` 中约100行被注释的旧代码
   - `ccu_kernel_alg_base.cc` 中大量被注释的旧实现
   - `reduce_scatter_op.cc` 中约100行被注释的测试代码
   - 多个executor文件中被注释的 `REGISTER_EXEC_V2` 宏调用
   
   这降低了代码可读性，建议清理或通过版本控制历史保留

5. **`reduce_scatter_op.cc` 中包含大段注释的测试代码**：应删除或移至专门的测试文件

### 🟢 低风险

6. **Commit message编码问题**：部分commit message出现中文乱码（如 `reduce c鎺ュ彛淇敼`），不影响代码功能但影响可读性和代码审查

7. **`CCU_MAX_RANK_SIZE = 16` 硬编码**：固定最大rank数为16，如果未来硬件支持更多rank，需要修改此常量

---

## 六、接口映射对照表

### 头文件映射

| 旧头文件 | 新头文件 |
|----------|----------|
| `ccu_kernel.h` | `ccu_api.hpp` + `ccu_control_api.h` |
| `hccl_ccu_res.h` | `ccu_types.h` |
| 无 | `ccu_log.h`（新增） |

### 命名空间/类型映射

| 旧（C++） | 新（C风格） |
|------------|-------------|
| `hcomm::CcuKernel` | 自由函数 + `CcuKernelCtxBase` |
| `CcuRep::Variable` | `ccu::Variable` |
| `CcuRep::CcuBuf` | `ccu::CcuBuffer` |
| `CcuRep::CompletedEvent` | `ccu::Event` |
| `CcuRep::LocalAddr` | `ccu::LocalAddr` |
| `CcuRep::RemoteAddr` | `ccu::RemoteAddr` |
| `CcuRep::Executor` | 移除 |
| `CcuRep::LoopCall` | 移除 |
| `std::vector<T>` | `ccu::Array<T>` 或 C数组 |
| `HcclResult`（返回值） | `CcuResult`（返回值） |

### 函数映射

| 旧函数 | 新函数 |
|--------|--------|
| `HcclCcuKernelRegister(comm, ...)` | `HcommCcuKernelRegisterStart(insHandle)` + `HcommCcuKernelRegister(insHandle, ...)` + `HcommCcuKernelRegisterEnd(insHandle)` |
| `HcclCcuKernelRegisterFinish(comm)` | `HcommCcuKernelRegisterEnd(insHandle)` |
| 无 | `HcclCommQueryCcuIns(comm, ...)` |
| `GenerateCcuKernelSignature(...)` | 已删除 |

---

## 七、总结与建议

### 当前状态

该PR处于**半完成状态**，仅 all_gather、reduce、reduce_scatter 三个算子完成了C接口适配，其余算子的编译被临时禁用。已完成的适配中，部分变体（如2Die、MultiJetty等）也被注释掉。

### 合入建议

1. **不建议直接合入master**：大量模块被禁用会导致功能回退
2. **建议分阶段推进**：
   - 第一阶段：完成所有算子的C接口适配，确保功能对等
   - 第二阶段：清理注释代码和调试日志
   - 第三阶段：恢复所有被禁用的模块编译
   - 第四阶段：合入master
3. **合入前必须清理**：
   - 删除 `xjhlog2`/`xjhlog3` 调试日志
   - 清理被注释的旧代码（依赖版本控制历史保留）
   - 移除 `reduce_scatter_op.cc` 中的测试代码
4. **建议增加回归测试**：确保C接口适配后，all_gather、reduce、reduce_scatter 的功能与性能与旧接口一致
