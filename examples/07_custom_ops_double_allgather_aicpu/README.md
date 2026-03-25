# 自定义 Double AllGather 样例

本样例在 `Atlas A3` 平台上提供一个双输入、双输出的自定义融合通信算子：

```cpp
HcclDoubleAllGatherCustom(sendBuf0, recvBuf0, count0, dtype0,
                          sendBuf1, recvBuf1, count1, dtype1,
                          comm, stream)
```

当前实现特征：

- 一次自定义接口调用
- 一次 kernel launch
- AICPU 自定义 kernel 内执行 fused ring schedule
- 同一个 step 循环里依次处理 `gather0` 和 `gather1`
- 支持 `count0/count1`、`dtype0/dtype1` 自定义
- 样例内置 `warmup/iters` 和耗时统计
- 首版不支持 inplace

## 目录结构

```text
07_custom_ops_double_allgather_aicpu/
├── common/
├── inc/
├── op_host/
├── op_kernel_aicpu/
├── testcase/
├── CMakeLists.txt
└── README.md
```

## 编译

```bash
export ASCEND_CANN_PACKAGE_PATH=/home/kk/Ascend
mkdir build
cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
make -j
```

## 运行

```bash
cd testcase
make run
```

也可以手动指定参数：

```bash
export ASCEND_CANN_PACKAGE_PATH=/home/kk/Ascend
export HCCL_CUSTOM_KERNEL_LAUNCH_ASC=1
export LD_LIBRARY_PATH=../build:${ASCEND_CANN_PACKAGE_PATH}/lib64:$LD_LIBRARY_PATH
mpirun -n 2 ./main --count0 1024 --count1 2048 --dtype0 fp32 --dtype1 int32 --warmup 5 --iters 20
```

## 路径说明

你这边的 Ascend 安装在 `/home/kk/` 下，因此本文档不再默认使用 `/usr/local/Ascend/...`。
编译和运行前，请先把 `ASCEND_CANN_PACKAGE_PATH` 指向 `/home/kk/` 下真实的 toolkit 根目录。

## 说明

这版样例不再调用现成 `HcclAllGather`，而是：

- Host 侧自己申请 `AICPU` 线程、邻居 channel 和中转 buffer
- kernel 内按 ring 规则完成两组 gather 的 step-by-step 传输
- testcase 负责结果校验和耗时统计

当前更偏“首版可重构骨架 + 核心执行路径”，是否完全跑通仍需你在 A3 实机验证。

