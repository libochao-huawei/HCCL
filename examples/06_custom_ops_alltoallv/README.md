# 自定义通信算子 - AlltoAllV 通信

## 样例介绍

本样例展示如何基于 HCCL AIV 通信编程接口开发 AlltoAllV 自定义通信算子，主要功能点：

1.  基于 AIV (AI Vector) 通信引擎实现 AlltoAllV 集合通信算子。
2.  支持变长数据交换（每个 rank 发送/接收的数据量可以不同）。
3.  包含 Host 侧算子逻辑与 Device 侧 Kernel 实现。
4.  提供完整的编译构建与测试验证流程。

## 目录结构

```text
├── CMakeLists.txt                      # 根目录编译/构建配置文件
├── op_host/
│   ├── CMakeLists.txt
│   ├── all_to_all_v.cc                 # HcclAlltoAllVCustom 算子Host侧实现
│   ├── launch_kernel.cc                # Kernel 下发逻辑实现
│   └── launch_kernel.h                 # Kernel 下发接口定义
├── op_kernel/
│   ├── CMakeLists.txt
│   └── launch_kernel_asc.asc           # 算子 Kernel 侧实现 (Ascend C)
├── inc/
│   ├── hccl_custom_alltoallv.h         # 自定义算子对外接口头文件
│   ├── common.h                        # 公共类型定义与宏
│   ├── aiv_all_to_all_v_mesh_1D.h      # AIV AlltoAllV 核心算法实现
│   ├── aiv_communication_base_v2.h     # AIV 通信基类
│   ├── log.h                           # 日志工具
│   ├── extra_args.h                    # 额外参数定义
│   └── sync_interface.h                # 同步接口定义
└── testcase/
    ├── CMakeLists.txt                  # 测试用例 CMake 配置文件
    ├── Makefile                        # 测试用例 Makefile (用于编译运行)
    └── main.cc                         # 测试用例主程序
```

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

以 root 用户默认安装路径为例：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

此外，运行测试用例需要 MPI 环境支持，请确保已安装并配置好 MPI。

## 二、编译与运行

本样例提供了基于 CMake 的构建流程以及基于 Makefile 的测试运行脚本。

### 1. 编译自定义算子库

在样例根目录下执行以下命令：

```bash
# 1. 创建构建目录
mkdir build

# 2. 进入构建目录
cd build

# 3. 执行 CMake 配置
cmake ..

# 4. 编译项目 (生成 libhccl_custom_alltoallv.so)
make
```

### 2. 运行测试用例

编译完成后，进入 `testcase` 目录执行测试：

```bash
# 5. 进入测试用例目录
cd ../testcase

# 6. 编译并运行测试用例
# 该命令会自动编译测试程序，设置 LD_LIBRARY_PATH 并使用 mpirun 运行
make run
```

### 3. 预期结果

运行成功后，终端将输出类似以下的日志信息（以 2 卡运行为例）：

```text
[INFO] MPI Initialized. World Size: 2
[INFO] Device 0 selected (Total devices: 8)
[INFO] Device 1 selected (Total devices: 8)
[INFO] HCCL Comm Initialized
[INFO] Buffers allocated and initialized
[INFO] Starting HcclAlltoAllVCustom...
[INFO] HcclAlltoAllVCustom completed and synchronized
[INFO] Test Passed!
```

## 三、接口说明

### HcclAlltoAllVCustom

自定义 AlltoAllV 算子接口，支持变长数据交换。

```c
HcclResult HcclAlltoAllVCustom(
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
| sdispls | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 发送数据的起始偏移（以元素为单位） |
| recvBuf | 输出 | 接收缓冲区起始地址 |
| recvCounts | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 接收的元素个数 |
| rdispls | 输入 | 指向 uint64_t 数组，长度为 rankSize，表示每个 rank 接收数据的起始偏移（以元素为单位） |
| dataType | 输入 | 数据类型，支持 FP32、FP16、INT32、INT8 等 |
| comm | 输入 | HCCL 通信域 |
| stream | 输入 | ACL 流 |

### 示例

```c
// 假设有 4 个 rank，每个 rank 发送不同数量的数据
uint64_t sendCounts[4] = {100, 200, 150, 250};
uint64_t sdispls[4] = {0, 100, 300, 450};
uint64_t recvCounts[4] = {200, 150, 250, 100};  // 每个 rank 接收的数量可能不同
uint64_t rdispls[4] = {0, 200, 350, 600};

HcclAlltoAllVCustom(sendBuf, sendCounts, sdispls, recvBuf, 
                   recvCounts, rdispls, HCCL_DATA_TYPE_FP32, 
                   hcclComm, stream);
```