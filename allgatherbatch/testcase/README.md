# AllGatherBatch Testcase

目录总览可参考上一级的 `../README.md`。

这个目录下的 testcase 参考了 `04_custom_ops_p2p` testcase 的组织方式：

- `aclInit -> HcclGetRootInfo -> 多线程多 device -> HcclCommInitRootInfo`
- 每个线程独立创建 `stream`
- 每个线程独立申请 host/device buffer
- 调用自定义接口 `HcclAllGatherBatch`

## 运行前提

- 已正确 source Ascend/CANN 环境
- 自定义算子库已安装到 `${ASCEND_HOME_PATH}/opp/vendors/cust/`
- 运行环境中可以访问目标 NPU 设备

## 方式一：CMake 生成的可执行文件

如果已经通过上层 `CMake` 构建出 `allgatherbatch_testcase`，可以直接运行：

```bash
./allgatherbatch_testcase --token-bytes 327680 --scale-count 128 --devices 8
```

## 方式二：使用 testcase 目录下的 Makefile

这个目录额外提供了一个和样例风格接近的 `Makefile`：

```bash
make
make test-default
make test-fast
make test-single-item
```

几个预设目标的含义：

- `test-default`
  默认双 item 场景，贴近设计文档里 `token + scale` 的组合
- `test-fast`
  更小的数据量和较少 item，适合先做链路冒烟
- `test-single-item`
  只保留 token item，把 `scale-count` 固定为 `0`

## 方式三：使用 run.sh

如果你已经把可执行文件构建出来，也可以直接用目录里的轻量脚本切场景：

```bash
./run.sh
./run.sh fast
./run.sh single-item --no-verify
```

它会自动带上对应场景的默认参数；你仍然可以在后面继续追加自己的参数覆盖默认值。

## 自定义数据量

支持的关键参数：

- `--token-bytes N`
  每个 rank 的 int8 token 字节数
- `--scale-count N`
  每个 rank 的 fp32 scale 元素数
- `--devices N`
  使用的 device 数量
- `--print-count N`
  每个 rank preview 时打印的元素个数
- `--warmup N`
  正式计时前的 warmup 次数
- `--iters N`
  正式计时次数
- `--no-verify`
  跳过 host 侧结果校验

示例：

```bash
./allgatherbatch_testcase --token-bytes 65536 --scale-count 0 --devices 4 --warmup 2 --iters 20
./allgatherbatch_testcase --token-bytes 327680 --scale-count 128 --devices 8 --warmup 3 --iters 10
./allgatherbatch_testcase --token-bytes 327680 --scale-count 128 --devices 8 --no-verify
```

如果更习惯 `make test` 风格，也可以把参数写成变量：

```bash
make test TOKEN_BYTES=65536 SCALE_COUNT=0 DEVICES=4 WARMUP=2 ITERS=20
make test-default TOKEN_BYTES=327680 SCALE_COUNT=128 DEVICES=8
make test-fast
make test-single-item EXTRA_ARGS=--no-verify
```

## 当前 testcase 做了什么

- 构造 1 个或 2 个 item
  - token item: `int8`
  - scale item: `fp32`
- 每个 rank 的输入数据都按固定规则填充，便于校验输出
- 运行结束后会：
  - 打印每个 rank 的平均耗时
  - 打印输出 preview
  - 默认执行 host 侧结果校验

## 当前 testcase 没做什么

- 还没有覆盖更多异常路径
- 还没有和双 `HcclAllGather` 基线做同程序内对比
- 还没有做 profiling 接入



