# A3 AICPU 双输入双输出 AllGather 自定义算子样例

## 1. 样例目标

本样例位于 `examples/06_custom_ops_allgather_aicpu_2in2out`，目标是在 A3 上提供一个基于 AICPU 通信引擎的自定义算子样例：

- 对外接口固定为 `2 input -> 2 output`
- 将两次 `allgather` 的语义融合为一次自定义算子调用
- 优先覆盖 small-count 场景
- 不满足 fused 条件时回退到两次原生 `HcclAllGather`

## 2. 当前实现进度

当前代码已经完成到“最小可运行 fused 版本 + 双子线程 peer 分工 + AICPU 打包链补齐”阶段：

- Host 侧已完成参数检查、small-count 判定、AICPU kernel 加载与 launch
- 已完成 `engine ctx / thread / channel / ccl buffer / notify id` 资源准备
- `ExecOp` 已支持 `route0 -> route1` 顺序执行
- `thread1 / thread2` 已能分摊不同 peer 子集的 read + DATA_SIGNAL
- `testcase` 已提供 baseline/fused 功能校验与 benchmark 骨架
- 根 `CMakeLists.txt` 已补齐 `AICPU .so -> tar.gz -> 签名包 -> json` 的完整打包链

当前还没有完成的部分：

- `testcase` 尚未在当前环境实际编译运行验证
- `README` 中不承诺所有规模下都一定优于 baseline
- 更细的 stage 化增强、更多 benchmark 场景、examples 索引收口仍可继续补强

## 3. 目录说明

```text
06_custom_ops_allgather_aicpu_2in2out/
├── inc/                  # 对外头文件、公共结构定义
├── common/               # device 侧编排核心
├── op_host/              # Host 侧资源准备与 launch
├── op_kernel_aicpu/      # AICPU kernel 入口与 json 描述
├── scripts/              # AICPU tar 包签名配置
├── testcase/             # 功能校验与 benchmark
├── DESIGN.md             # 详细设计文档
├── todolist.md           # 分阶段任务清单
└── README.md             # 当前说明文档
```

## 4. 对外接口

头文件：`inc/hccl_custom_allgather_2in2out.h`

```c
HcclResult HcclAllGather2In2OutAicpuCustom(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream);
```

语义说明：

- `sendCount0` 和 `sendCount1` 的单位都是元素个数
- `recvBuf0` 的输出布局是标准 allgather：按 rank 顺序拼接 route0 的各 rank slice
- `recvBuf1` 同理，对应 route1
- 当 small-count 条件满足时，优先走 fused AICPU 自定义算子路径
- 当条件不满足时，内部退回两次原生 `HcclAllGather`

## 5. 构建链怎么看

这部分很重要，因为以后你写 `allreduce` 或别的 AICPU 自定义算子时，外层壳子大概率都长这样：

1. `op_host/` 先生成 Host 侧动态库
- 对外导出 `HcclAllGather2In2OutAicpuCustom()`
- 负责 small-count 判定、资源准备、加载 AICPU kernel、launch kernel

2. `op_kernel_aicpu/` 生成 device 侧 AICPU `.so`
- 入口是 `aicpu_kernel.cc`
- 真正的数据交换协议在 `common/exec_op.cc`

3. 根 `CMakeLists.txt` 再把 device `.so` 打成 tar 包并签名
- `package_aicpu_kernel()` 把 `.so` 打进 `aicpu_hccl_custom_allgather_2in2out.tar.gz`
- `sign_aicpu_kernel()` 再把 tar 包复制到 `signatures/` 并按 `scripts/*.xml` 进行签名

4. `liballgather_2in2out_aicpu_kernel.json` 描述 AICPU kernel 的入口信息
- Host 侧加载 json 后，才能按 `functionName` 找到 `HcclLaunchAllGather2In2OutAicpuKernel`

你可以把这条链记成一句话：

`Host 接口库` 负责“让用户能调到这个算子”，`AICPU kernel .so + tar + json` 负责“让 device 侧真的有这个算子可执行”。

## 6. testcase 说明

`testcase/main.cc` 当前做了两件事：

1. 功能校验
- baseline：连续两次原生 `HcclAllGather`
- fused：一次 `HcclAllGather2In2OutAicpuCustom`
- route0 和 route1 使用不同的数值模式，便于识别串路问题

2. benchmark
- 支持 `warmup + iters`
- baseline/fused 使用一致的计时口径
- 输出 `avgUs / bestUs / speedup`

### 6.1 参数

支持如下命令行参数：

```bash
--count0=1024
--count1=2048
--warmup=20
--iters=100
```

其中：

- `count0` 控制 route0 的元素个数
- `count1` 控制 route1 的元素个数
- 两个数据量可以不同，便于验证“双路不等长”场景

### 6.2 编译

```bash
cd examples/06_custom_ops_allgather_aicpu_2in2out/testcase
make
```

### 6.3 运行

默认运行：

```bash
cd examples/06_custom_ops_allgather_aicpu_2in2out/testcase
make run
```

通过 `make run` 直接指定数据量：

```bash
make run NP=2 COUNT0=4096 COUNT1=8192 WARMUP=20 ITERS=100
```

或直接运行二进制：

```bash
./main --count0=4096 --count1=8192 --warmup=20 --iters=100
```

或使用脚本：

```bash
NP=2 COUNT0=1024 COUNT1=2048 WARMUP=20 ITERS=100 ./benchmark.sh
```

## 7. 看结果时关注什么

如果功能正确，日志中应至少看到：

- `baseline route0/route1 校验通过`
- `fused route0/route1 校验通过`
- `baseline avgUs=... bestUs=...`
- `fused avgUs=... bestUs=...`
- `Summary` 中的 `speedup`

## 8. 以后迁移到 allreduce 时复用什么

如果后面你要继续扩展到 `allreduce`，建议优先复用下面这几层：

- `op_host/resource.*` 的资源准备外壳
- `op_host/load_kernel.*` / `launch_kernel.*` 的 launch 外壳
- 根 `CMakeLists.txt` 里的 AICPU 打包和签名链
- `testcase/main.cc` 的初始化、校验、benchmark 外壳

真正需要替换的主要是：

- `common/exec_op.cc` 的协议与数据流
- route 的语义解释
- 结果校验规则
- `op_kernel_aicpu/lib*.json` 里的函数名和 so 名
