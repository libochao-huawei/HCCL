# AllGatherBatch Testcase

目录总览可参考上一级的 [../README.md](../README.md)。

这个目录下的 testcase 参考了 `examples/04_custom_ops_p2p/testcase` 的组织方式，主流程如下：

- `aclInit -> HcclGetRootInfo -> 多线程多 device -> HcclCommInitRootInfo`
- 每个线程独立创建 `stream`
- 每个线程独立申请 host/device buffer
- 调用自定义接口 `HcclAllGatherBatch`
- 从 device 拷回 host 后做结果校验与简单计时

## 一、运行前提

运行前需要满足以下条件：

- 已正确 `source` Ascend/CANN 环境
- 自定义算子库已安装到 `${ASCEND_HOME_PATH}/opp/vendors/cust/`
- 运行环境中可以访问目标 NPU 设备

## 二、准备 Ascend/CANN 环境

通常先执行 Ascend Toolkit 的环境脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果你的环境脚本在其他位置，也可以按实际安装路径替换。执行后建议确认以下变量已经生效：

```bash
echo $ASCEND_HOME_PATH
echo $ASCEND_CANN_PACKAGE_PATH
```

如果 `ASCEND_CANN_PACKAGE_PATH` 没有单独设置，`hccl/build.sh` 也会优先使用 `ASCEND_HOME_PATH` 作为 Toolkit 安装路径。

## 三、编译自定义算子包

在 `hccl` 根目录下执行：

```bash
cd hccl
bash build.sh --vendor=cust --ops=allgatherbatch --custom_ops_path=./allgatherbatch
```

其中：

- `--vendor=cust`
  表示安装到 `opp/vendors/cust`
- `--ops=allgatherbatch`
  表示自定义算子名称
- `--custom_ops_path=./allgatherbatch`
  表示当前 custom op 工程目录

编译完成后，安装包会生成在 `hccl/build_out/` 目录下。

## 四、安装自定义算子包

可通过 `build_out` 目录下生成的 `.run` 安装包进行安装，命令形式参考如下：

```bash
cd hccl
./build_out/cann-allgatherbatch_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

其中：

- `<arch>` 是当前编译环境对应的系统架构，实际名称以 `build_out/` 中生成的文件为准
- `<ascend_cann_path>` 是 Ascend/CANN 安装目录；通常可直接写成 `${ASCEND_HOME_PATH}`

例如：

```bash
./build_out/cann-allgatherbatch_linux-x86_64.run --install --install-path=${ASCEND_HOME_PATH}
```

安装完成后，至少应能看到以下文件：

- 头文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/include/allgather_batch.h`
- 动态库：`${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_allgatherbatch.so`
- AICPU 算子描述文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/liballgatherbatch_aicpu_kernel.json`
- AICPU 算子包：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_allgatherbatch.tar.gz`

可以用下面的命令快速确认：

```bash
ls ${ASCEND_HOME_PATH}/opp/vendors/cust/include/allgather_batch.h
ls ${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_allgatherbatch.so
ls ${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/liballgatherbatch_aicpu_kernel.json
ls ${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_allgatherbatch.tar.gz
```

## 五、AICPU 相关注意事项

### 1. AICPU 算子验签

如果运行环境开启了自定义 AICPU 算子验签，可能需要先检查相关开关。参考 `04_custom_ops_p2p` 样例，可以先查询：

```bash
for i in {0..7}; do npu-smi info -t custom-op-secverify-enable -i $i; done
for i in {0..7}; do npu-smi info -t custom-op-secverify-mode -i $i; done
```

如果你的环境要求关闭验签或切换模式，请按现场规范执行；这一步通常依赖管理员权限。

### 2. AICPU 白名单

AICPU 默认只会加载白名单中允许的 tar 包。若你的环境启用了该限制，需要把 `aicpu_allgatherbatch.tar.gz` 加入白名单配置。

可参考 `04_custom_ops_p2p` 的做法，编辑：

```bash
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

追加类似内容：

```ini
name:aicpu_allgatherbatch.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
```

如果你的机器使用的是其他 CANN 安装路径，请按实际路径修改。

## 六、编译 testcase

### 方式一：通过上层 CMake 构建

如果你已经在 `hccl` 根目录完成 custom op 的 CMake 构建，并且不是 `KERNEL_MODE`，则会自动把 testcase 一起编进去。

对应目标是：

```bash
cd hccl
cmake -S . -B build \
  -DENABLE_CUSTOM=ON \
  -DCUSTOM_OPS_PATH=./allgatherbatch \
  -DCUSTOM_OPS_NAME=allgatherbatch \
  -DCUSTOM_OPS_VENDOR=cust
cmake --build build -j
```

生成的可执行文件通常位于：

```bash
hccl/build/allgatherbatch/testcase/allgatherbatch_testcase
```

### 方式二：使用 testcase 目录下的 Makefile

这个目录额外提供了一个更接近样例风格的 `Makefile`，依赖已安装到 `${ASCEND_HOME_PATH}/opp/vendors/cust/` 的头文件和库。

在 `hccl/allgatherbatch/testcase` 目录下执行：

```bash
make
```

如果编译成功，会生成：

```bash
./allgatherbatch_testcase
```

## 七、运行 testcase

### 方式一：直接运行可执行文件

```bash
./allgatherbatch_testcase --token-bytes 327680 --scale-count 128 --devices 8
```

### 方式二：使用 Makefile 预设目标

```bash
make test-default
make test-fast
make test-single-item
```

几个预设目标的含义：

- `test-default`
  默认双 item 场景，贴近设计文档里的 `token + scale` 组合
- `test-fast`
  更小的数据量和较少 item，适合先做链路冒烟
- `test-single-item`
  只保留 token item，把 `scale-count` 固定为 `0`

### 方式三：使用 run.sh

如果已经构建出 `allgatherbatch_testcase`，也可以用目录里的轻量脚本切场景：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
./run.sh
./run.sh fast
./run.sh single-item --no-verify
```

注意：

- `run.sh` 只负责拼默认参数，不会自动补 `LD_LIBRARY_PATH`
- 因此使用 `run.sh` 前，建议先手工导出上面的库路径

## 八、自定义数据量

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

## 九、当前 testcase 做了什么

- 构造 1 个或 2 个 item
  - token item: `int8`
  - scale item: `fp32`
- 每个 rank 的输入数据都按固定规则填充，便于校验输出
- 运行结束后会：
  - 打印每个 rank 的平均耗时
  - 打印输出 preview
  - 默认执行 host 侧结果校验

## 十、当前 testcase 没做什么

- 还没有覆盖更多异常路径
- 还没有和双 `HcclAllGather` 基线做同程序内对比
- 还没有做 profiling 接入
