# 自定义通信算子 - AllGatherBatch 通信

本样例实现 `HcclAllGatherBatch`，用于把多路 `AllGather` 合并到一次 Host 下发中执行。

当前目录实现走 CCU 自定义 kernel 路径，不再使用 AIV `.asc` 二进制下发。

## 编译与运行

本样例推荐和 [examples/05_custom_ops_allgather/README.md](/Users/lajiaojiang/work/project/hccl/examples/05_custom_ops_allgather/README.md) 一样使用：

- 直接依赖机器上已安装的官方 toolkit / HCCL 环境
- 只编译当前 example 目录
- 不重新编译整个 HCCL run 包
- 不覆盖 toolkit 自带的 built-in `aicpu_hccl.tar.gz`

### 环境准备

以非 root 用户安装到家目录为例：

```bash
export ASCEND_CANN_PACKAGE_PATH=/home/x00958740/Ascend/ascend-toolkit/latest
source $ASCEND_CANN_PACKAGE_PATH/set_env.sh
```

此外需要提前安装并配置：

- `cmake`
- `make`
- `mpic++`
- `mpirun`

### 编译

在当前目录下直接执行：

```bash
mkdir -p build
cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/home/x00958740/Ascend/ascend-toolkit/latest
make -j
```

说明：

- `cmake ..` 表示使用上一级目录的 `CMakeLists.txt` 配置工程
- `-DASCEND_CANN_PACKAGE_PATH=...` 表示显式把 toolkit 安装路径传给 CMake

如果 shell 环境变量已经正确传递，理论上也可以直接执行：

```bash
cmake ..
```

但为了避免 CMake 回退到默认的 `/usr/local/Ascend/ascend-toolkit/latest`，更推荐显式带上 `-DASCEND_CANN_PACKAGE_PATH=...`。

如果 `ASCEND_CANN_PACKAGE_PATH` 没有设置正确，`op_host` 和 `testcase` 的头文件、库文件会找不到。

### 运行 Host UT

`Host UT` 不依赖 NPU 和 MPI，适合先做纯 Host 逻辑验证：

```bash
cd build
./testcase/test_custom_allgather_batch_host_ut
```

预期会看到：

```text
[SUMMARY] host UT passed
```

其中负例会主动触发参数校验日志，这属于测试预期，不代表用例失败。

### 运行集成 testcase

`test_custom_allgather_batch` 是真实集成测试，依赖：

- ACL runtime
- HCCL 通信域初始化
- MPI 多进程
- 可正常 `aclrtSetDevice` 的设备环境

运行命令：

```bash
cd build
export LD_LIBRARY_PATH=$(pwd)/op_host:$ASCEND_CANN_PACKAGE_PATH/lib64:$LD_LIBRARY_PATH
mpirun -n 2 ./testcase/test_custom_allgather_batch
```

如果当前用户是 root，需要改成：

```bash
mpirun --allow-run-as-root -n 2 ./testcase/test_custom_allgather_batch
```

### 说明

- 本样例当前的推荐验证路径是：`Host UT -> 集成 testcase`
- 如果集成 testcase 在 `aclrtSetDevice()` 阶段失败，优先排查机器运行环境，不要先重编 HCCL run 包
- 如果只是调试本样例，不建议使用 `bash build.sh --full --pkg` 去替换 toolkit 自带的 HCCL 基础包

## 目标

`HcclAllGatherBatch` 的目标不是把多路数据并行执行，而是把多次 `HcclAllGather` 的 Host 固定开销合并掉。

典型场景是同一 rank 需要对多块异构数据分别做 AllGather，例如：

- 一路 `int8 token`
- 一路 `fp32 scale`

如果分别调用两次 `HcclAllGather`，会支付两次 Host launch 成本。  
`HcclAllGatherBatch` 把这些 item 合成一次调用，在一个 CCU task 内按顺序执行全部 item。

语义上等价于：

```c
for (uint32_t i = 0; i < itemCount; ++i) {
    HcclAllGather(items[i].sendBuf, items[i].recvBuf,
                  items[i].sendCount, items[i].dataType,
                  comm, stream);
}
```

但只支付一次 Host 下发成本。

## 对外接口

对外接口定义在 [inc/hccl_custom_allgather_batch.h](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/inc/hccl_custom_allgather_batch.h)。

```c
typedef struct {
    void *sendBuf;
    uint64_t sendCount;
    HcclDataType dataType;
    void *recvBuf;
} HcclAllGatherItem;

HcclResult HcclAllGatherBatch(
    const HcclAllGatherItem *items,
    uint32_t itemCount,
    HcclComm comm,
    aclrtStream stream);
```

约束：

- `items`、`comm`、`stream` 非空
- `1 <= itemCount <= 8`
- 每个 `item.sendBuf`、`item.recvBuf` 非空
- 每个 `item.sendCount > 0`
- `item.dataType` 必须是当前实现支持的类型

当前支持：

- `int8`
- `int32`
- `fp16`
- `fp32`

## 目录结构

```text
├── CMakeLists.txt
├── README.md
├── inc/
│   ├── hccl_custom_allgather_batch.h
│   ├── all_gather_batch_item.h
│   ├── ccu_kernel_all_gather_batch_mesh1d.h
│   ├── common.h
│   ├── log.h
│   └── sync_interface.h
├── op_host/
│   ├── CMakeLists.txt
│   ├── all_gather_batch.cc
│   ├── launch_kernel.cc
│   └── launch_kernel.h
├── op_kernel/
│   ├── CMakeLists.txt
│   └── ccu_kernel_all_gather_batch_mesh1d.cc
└── testcase/
    ├── CMakeLists.txt
    ├── Makefile
    └── main.cc
```

## 模块关系

### 1. API 层

[inc/hccl_custom_allgather_batch.h](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/inc/hccl_custom_allgather_batch.h)

职责：

- 对外暴露 `HcclAllGatherBatch`
- 定义业务侧使用的 `HcclAllGatherItem`

这里的结构体是“业务描述”，只表达调用方想做什么。

### 2. 公共参数层

[inc/common.h](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/inc/common.h)

职责：

- 定义内部使用的 `OpParam`
- 提供 `MAX_ITEM_COUNT`
- 提供 `GetDataTypeSize()` 和 `IsSupportedDataType()`

这里的 `OpParam` 是 Host 侧标准化后的参数容器：

```cpp
struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    uint32_t rank;
    uint32_t rankSize;
    uint32_t itemCount;
    HcclAllGatherItem items[MAX_ITEM_COUNT];
};
```

它仍然保留 `items[]`，没有把 `sendBuf/recvBuf/sendCount/dataType` 再拆成多组数组。

### 3. Host 入口层

[op_host/all_gather_batch.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_host/all_gather_batch.cc)

职责：

- 参数校验
- 组装 `OpParam`
- 维护 CCU context 缓存
- 调用 launch

这里是整个自定义算子的 Host 入口。

### 4. Host 下发层

[op_host/launch_kernel.h](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_host/launch_kernel.h)  
[op_host/launch_kernel.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_host/launch_kernel.cc)

职责：

- 申请 CCU channel
- 构造并注册 CCU kernel
- 每次调用构造 task 参数并下发

这层负责把 Host 的 `OpParam` 转成 CCU 能执行的 task。

### 5. CCU Kernel 描述层

[inc/ccu_kernel_all_gather_batch_mesh1d.h](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/inc/ccu_kernel_all_gather_batch_mesh1d.h)

这里定义了三类关键结构：

#### `CcuAllGatherBatchItem`

这是发给 CCU 的精简执行描述：

```cpp
struct CcuAllGatherBatchItem {
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t token;
    uint64_t offset;
    uint64_t sliceSize;
};
```

它只保留设备执行真正需要的信息。

#### `CcuKernelArgAllGatherBatchMesh1D`

这是 kernel 级静态参数，主要包含：

- `rankSize`
- `rankId`
- `itemCount`
- `channels`

它在 kernel 注册时准备好。

#### `CcuTaskArgAllGatherBatchMesh1D`

这是 task 级动态参数，主要包含：

- `itemCount`
- 每一路 `CcuAllGatherBatchItem`

它在每次调用时根据当前 `items[]` 生成。

### 6. CCU Device 执行层

[op_kernel/ccu_kernel_all_gather_batch_mesh1d.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_kernel/ccu_kernel_all_gather_batch_mesh1d.cc)

职责：

- 反序列化 task 参数
- 对每一路 item 做同步
- 对每一路 item 执行 `GroupBroadcast`
- 最后做全局收敛

这是设备侧真正执行 batch allgather 的地方。

## 执行流程

### 第 1 步：业务侧调用接口

业务侧准备 `HcclAllGatherItem items[]`，然后调用：

```c
HcclAllGatherBatch(items, itemCount, comm, stream);
```

测试入口可以看 [testcase/main.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/testcase/main.cc)。

### 第 2 步：Host 校验所有 item

[op_host/all_gather_batch.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_host/all_gather_batch.cc)

`CheckItemValid()` 会逐路检查：

- `sendBuf` 非空
- `recvBuf` 非空
- `sendCount > 0`
- `dataType` 合法
- `sendCount * sizeof(type)` 不溢出

### 第 3 步：Host 组装 `OpParam`

入口函数会：

- 取 `commName`
- 生成 `tag`
- 查询 `rank`
- 查询 `rankSize`
- 把用户 `items[]` 拷到 `OpParam.items[]`

到这里，Host 已经把业务调用整理成统一内部格式。

### 第 4 步：初始化或复用 CCU context

同一个 `(commName, itemCount)` 会复用一个 `CcuContext`。

`CcuContext` 里缓存：

- `channels`
- `kernelArg`
- `kernelCreator`
- `kernelHandle`

这样就不需要每次调用都重新注册 kernel。

### 第 5 步：申请 CCU channel

[op_host/launch_kernel.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_host/launch_kernel.cc)

`BuildChannelRequests()` 会：

- 遍历当前 rank 到每个远端 rank 的 link
- 为每个远端 rank 构造 `HcclChannelDesc`
- 使用 `COMM_ENGINE_CCU` 调用 `HcclChannelAcquire`

这一步拿到的是 CCU kernel 后续做同步和广播需要的通信通道。

### 第 6 步：注册 CCU kernel

`InitCcuContext()` 会创建：

- `CcuKernelArgAllGatherBatchMesh1D`
- `kernelCreator`

然后调用：

- `HcclCcuKernelRegister`
- `HcclCcuKernelRegisterFinish`

注册完成后，`kernelHandle` 会被缓存到 `CcuContext` 中。

### 第 7 步：把业务 item 转成 CCU task item

`LaunchKernel()` 会把每个 `HcclAllGatherItem` 转成 `CcuAllGatherBatchItem`：

- `inputAddr = sendBuf`
- `outputAddr = recvBuf`
- `sliceSize = sendCount * sizeof(dataType)`
- `token = GetTokenInfo(sendBuf, sliceSize)`
- `offset = rank * sliceSize`

其中 `offset` 表示本 rank 的数据在 `recvBuf` 中应该落到第几段。

### 第 8 步：Host 触发 CCU task 下发

`LaunchKernel()` 里会：

- 用 `HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, ...)` 获取 thread
- 构造 `CcuTaskArgAllGatherBatchMesh1D`
- 调用 `HcclCcuKernelLaunch`

此时控制权转到 CCU kernel。

### 第 9 步：CCU kernel 初始化资源

[op_kernel/ccu_kernel_all_gather_batch_mesh1d.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/op_kernel/ccu_kernel_all_gather_batch_mesh1d.cc)

`InitResource()` 会：

- 为每个远端 rank 创建 `peerOutput_`
- 为每个远端 rank 创建 `peerToken_`
- 为每个 item 创建本地变量：
  - `input`
  - `output`
  - `token`
  - `offset`
  - `groupOpSize`

这是把“远端同步信息”和“本地 item 执行槽位”准备好。

### 第 10 步：CCU kernel 反序列化参数

`LoadArgs()` 会把 Host 传来的 task 参数加载到本地变量。

这里对应设计里的“参数反序列化”。

### 第 11 步：逐路 PreSync

对每一路 item，先执行 `PreSync(item)`：

- 向所有 channel 广播本 rank 当前 item 的 `output`
- 向所有 channel 广播本 rank 当前 item 的 `token`
- 等待所有远端 rank 的当前 item 地址和 token 就绪

这样每个 rank 才知道这一轮数据该往谁的哪里写。

### 第 12 步：逐路执行 AllGather

然后执行 `DoAllGather(item)`：

- `src` 指向当前 item 的 `input`
- `dst` 为每个 rank 构造输出目标地址
- 本 rank 写本地 `item.output + offset`
- 其他 rank 写远端 `peerOutput[rank] + offset`
- 调用 `GroupBroadcast`

这样一轮下来，当前 rank 的这一段数据会被写到所有 rank 的目标缓冲区对应段上。

所有 rank 都做完这一轮后，这一路 item 的 allgather 就成立了。

### 第 13 步：全部 item 完成后做 PostSync

`PostSync()` 是最终收敛：

- 对所有 channel 发后同步 notify
- 等待所有 channel 完成

作用是防止不同 rank 在 batch 尾部发生跨轮交错。

## 数据在各层的形态

### 业务层

`HcclAllGatherItem`

表达的是“用户想做什么”。

### Host 层

`OpParam`

表达的是“本次调用的标准化参数”。

### CCU 执行层

`CcuAllGatherBatchItem`

表达的是“设备真正要使用哪些地址、token、偏移和字节数”。

这个分层的好处是：

- 业务字段和执行字段解耦
- Host 校验逻辑更清楚
- Device 侧只关心最小必要信息

## 当前实现特征

- 单服务器 Mesh1D 场景
- `rankSize <= 8`
- `itemCount <= 8`
- 每一路 item 顺序执行
- 优化点在于减少 Host launch 开销，而不是多路并行搬运

## 和旧版 AIV 思路的区别

当前版本已经明确切到 CCU 路线，和之前误引入的 AIV 方向不同：

- 不再使用 `.asc` kernel
- 不再调用 `aclrtLaunchKernelWithHostArgs`
- 不再维护 AIV buffer / AIV 地址表
- 不再使用 AIV 风格 `buffIn/xRankSize/...` 参数
- 改为使用 CCU 的：
  - channel acquire
  - kernel register
  - task launch
  - `KernelArg + TaskArg`
  - `PreSync + GroupBroadcast + PostSync`

## 测试

功能测试入口在 [testcase/main.cc](/Users/lajiaojiang/work/project/hccl/examples/06_custom_ops_allgather_batch/testcase/main.cc)。

当前测试构造了两路 item：

- 路 0：`fp32`
- 路 1：`int32`

执行完成后分别校验两路 `recvBuf` 是否按 rank 顺序拼接正确。

## 说明

这版 README 解释的是当前目录下已经整理出的 CCU 版实现关系和执行流程。  
如果后续要继续补齐编译链，还需要结合本地 CANN/CCU 环境验证以下内容：

- `ccu_kernel.h` 及相关头文件的实际可见性
- `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 的本地接口行为
- 该样例的 CMake 在本地环境中的 include/link 完整性
