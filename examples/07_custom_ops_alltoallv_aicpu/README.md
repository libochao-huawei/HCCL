# 自定义通信算子 - AlltoAllV 通信 (AICPU模式)

## 样例介绍

本样例展示如何基于 HCCL AICPU 通信编程接口开发 AlltoAllV 自定义通信算子，主要功能点：

1. 基于 AICPU (AI CPU) 通信引擎实现 AlltoAllV 集合通信算子。
2. 支持变长数据交换（每个 rank 发送/接收的数据量可以不同）。
3. 包含 Host 侧算子逻辑与 AICPU Kernel 实现。
4. 提供完整的编译构建与测试验证流程。

## 与 AIV 版本的区别

本样例（07_custom_ops_alltoallv_aicpu）与 06_custom_ops_alltoallv 的主要区别：

| 特性 | AIV版本 | AICPU版本 |
|------|---------|-----------|
| 通信引擎 | AIV (AI Vector) | AICPU (AI CPU) |
| Kernel实现 | AscendC (.asc文件) | C++ (.cc文件) |
| 数据传输 | GM直接拷贝 | Thread + Channel模式 |
| 同步机制 | Flag等待/记录 | Thread Notify等待/记录 |
| 适用场景 | 高性能向量计算 | 复杂逻辑控制 |

## 目录结构

```text
├── CMakeLists.txt                      # 根目录编译/构建配置文件
├── op_host/
│   ├── CMakeLists.txt
│   ├── all_to_all_v.cc                 # HcclAlltoAllVCustomAicpu 算子Host侧实现
│   ├── launch_kernel.cc                # Kernel 下发逻辑实现
│   ├── launch_kernel.h                 # Kernel 下发接口定义
│   ├── load_kernel.cc                  # AICPU Kernel 加载逻辑
│   └── load_kernel.h                   # 加载接口定义
├── op_kernel_aicpu/
│   ├── CMakeLists.txt
│   ├── aicpu_kernel.cc                 # AICPU Kernel 入口函数
│   ├── exec_op.cc                      # AlltoAllV 算法编排逻辑
│   ├── exec_op.h                       # 编排接口定义
│   └── liballtoallv_aicpu_kernel.json  # AICPU Kernel 算子描述文件
├── inc/
│   ├── hccl_custom_alltoallv_aicpu.h   # 自定义算子对外接口头文件
│   ├── common.h                        # 公共类型定义与宏
│   └── log.h                           # 日志工具
├── scripts/
│   └── hccl_custom_alltoallv_aicpu_check_cfg.xml  # 签名配置文件
└── testcase/
    ├── CMakeLists.txt                  # 测试用例 CMake 配置文件
    ├── Makefile                        # 测试用例 Makefile
    └── main.cc                         # 测试用例主程序
```

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>
- <term>Atlas A2 训练系列产品</term>

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

以 root 用户默认安装路径为例：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

此外，运行测试用例需要 MPI 环境支持，请确保已安装并配置好 MPI。

## 二、编译自定义算子包

hccl代码仓提供了自定义算子编译打包工程，开发者需要在代码仓根目录下执行 `build.sh` 进行编译：

```bash
# 下载hccl代码仓
git clone https://gitcode.com/cann/hccl.git

# 编译自定义算子包
bash build.sh --vendor=cust --ops=alltoallv_aicpu --custom_ops_path=./examples/07_custom_ops_alltoallv_aicpu
```

## 三、安装自定义算子包

自定义算子安装包在 `./build_out` 目录下，通过 `--install` 参数进行安装：

```bash
./build_out/cann-hccl_custom_alltoallv_aicpu_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

自定义算子包安装信息如下：

- 头文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_alltoallv_aicpu.h`
- 动态库：`${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_alltoallv_aicpu.so`
- AICPU 算子描述文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/liballtoallv_aicpu_kernel.json`
- AICPU 算子包：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_alltoallv.tar.gz`

## 四、配置 AICPU 环境

### 1. 关闭 AICPU 算子验签功能

```bash
# 设置AI CPU算子验签模式，关闭验签
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done
```

### 2. 修改 AICPU 白名单

```bash
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

将下列内容追加到配置文件中：

```ini
name:aicpu_hccl_custom_alltoallv.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
load_as_per_soc:false
```

## 五、执行自定义算子

### 1. 编译样例

在 `examples/07_custom_ops_alltoallv_aicpu/testcase` 目录下执行：

```bash
make
```

### 2. 执行样例

```bash
make run

# 或直接执行
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
mpirun -n 2 ./main
```

### 3. 预期结果

运行成功后，终端将输出类似以下的日志信息：

```text
[INFO] MPI Initialized. World Size: 2
[INFO] Device 0 selected (Total devices: 2)
[INFO] Device 1 selected (Total devices: 2)
[INFO] HCCL Comm Initialized
[INFO] Buffers allocated and initialized
[INFO] Starting HcclAlltoAllVCustomAicpu...
[INFO] HcclAlltoAllVCustomAicpu completed and synchronized
[INFO] Test Passed!
```

## 六、接口说明

### HcclAlltoAllVCustomAicpu

自定义 AlltoAllV 算子接口，支持变长数据交换。

```c
HcclResult HcclAlltoAllVCustomAicpu(
    void *sendBuf,           // 发送缓冲区
    void *sendCounts,       // 每个 rank 发送的数据元素个数 (uint64_t 数组)
    void *sdispls,          // 每个 rank 发送数据的偏移量 (uint64_t 数组)
    void *recvBuf,          // 接收缓冲区
    void *recvCounts,       // 每个 rank 接收的数据元素个数 (uint64_t 数组)
    void *rdispls,          // 每个 rank 接收数据的偏移量 (uint64_t 数组)
    HcclDataType dataType,  // 数据类型
    HcclComm comm,          // HCCL 通信域
    aclrtStream stream      // ACL 流
);
```

### 参数说明

| 参数 | 输入/输出 | 说明 |
|------|----------|------|
| sendBuf | 输入 | 发送缓冲区起始地址 |
| sendCounts | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 发送的元素个数 |
| sdispls | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 发送数据的起始偏移 |
| recvBuf | 输出 | 接收缓冲区起始地址 |
| recvCounts | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 接收的元素个数 |
| rdispls | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 接收数据的起始偏移 |
| dataType | 输入 | 数据类型，支持 FP32、FP16、INT32、INT8 等 |
| comm | 输入 | HCCL 通信域 |
| stream | 输入 | ACL 流 |

## 七、实现原理

### AICPU 通信模式

AICPU通信使用以下核心机制：

1. **Thread机制**: 使用 `HcclThreadAcquire` 获取AICPU线程，用于执行通信任务
2. **Channel机制**: 使用 `HcclChannelAcquire` 获取通信通道，用于数据传输
3. **Notify机制**: 使用 `HcommThreadNotifyWaitOnThread` 和 `HcommThreadNotifyRecordOnThread` 进行同步
4. **数据传输**: 使用 `HcommWriteOnThread`、`HcommReadOnThread`、`HcommLocalCopyOnThread` 进行数据搬运

### AlltoAllV 算法流程

1. Host侧准备参数和资源（线程、通道、内存）
2. 下发AICPU Kernel
3. AICPU侧执行：
   - 对每个remote rank，通过Channel发送数据到对方的CCL Buffer
   - 从CCL Buffer拷贝数据到输出缓冲区
   - 使用Notify进行同步
4. Host侧等待完成通知