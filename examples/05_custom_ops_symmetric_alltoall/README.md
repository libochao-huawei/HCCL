# 自定义通信算子 - 对称 AlltoAll

## 样例介绍

本样例展示如何基于 HCCL 通信编程接口开发对称 AlltoAll 集合通信算子，包含以下功能点：

1. 基于 AICPU 通信引擎实现对称 AlltoAll 通信算子
2. 支持对称内存（Symmetric Memory）实现 zero-copy 通信
3. 支持自定义算子包的独立构建、独立部署
4. 每个 rank 直接向所有其他 rank 发送/接收数据（Mesh 拓扑）

## 目录结构

```text
├── CMakeLists.txt                      # 编译/构建配置文件
├── op_host/
│   ├── symmetric_alltoall.cc          # HcclAllToAllCustom 算子实现源文件
│   ├── load_kernel.cc                  # AICPU Kernel 在 Host 侧的加载逻辑
│   └── launch_kernel.cc                # AICPU Kernel 在 Host 侧的下发逻辑
├── op_kernel_aicpu/
│   ├── libsymmetric_alltoall_aicpu_kernel.json  # AICPU Kernel 算子描述文件
│   ├── aicpu_kernel.cc                 # AICPU Kernel 实现逻辑
│   └── exec_op.cc                      # AICPU 算子编排逻辑（Mesh AlltoAll）
├── inc/
│   ├── hccl_custom_symmetric_alltoall.h  # 自定义 AlltoAll 算子接口头文件
│   ├── common.h                        # 公共类型头文件
│   └── log.h                           # 日志宏定义
├── scripts/
│   └── hccl_custom_symmetric_alltoall_check_cfg.xml  # 签名配置文件
└── testcase/
    ├── main.cc                         # 样例实现源文件（8卡测试）
    └── Makefile                        # 编译/构建配置文件
```

> 自定义算子编译工程依赖 HCCL 代码仓中的 [cmake](../../cmake) 配置和编译脚本 [build.sh](../../build.sh)

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品：

- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
```

## 二、编译自定义算子包

hccl代码仓提供了自定义算子编译打包工程，该工程依赖代码仓中的如下文件：

```text
├── build.sh                        # hccl代码仓根目录编译工程入口
├── CMakeLists.txt                  # hccl代码仓根目录编译/构建配置文件
├── cmake/
│   ├── config.cmake                # CMake变量定义
│   ├── func.cmake                  # CMake函数定义
│   ├── package.cmake               # 签名、打包函数定义
│   └── makeself_custom.cmake       # MakeSelf打包逻辑
└── scripts/
    ├── custom/install.sh           # 自定义算子包安装脚本
    └── sign/add_header_sign.py     # AICPU 算子包签名脚本
```

开发者首先需要下载hccl代码仓，然后在代码仓根目录下，执行 `build.sh` 进行编译：

```bash
# 下载hccl代码仓
git clone https://gitcode.com/cann/hccl.git

# 编译自定义算子包
bash build.sh --vendor=cust --ops=symmetric_alltoall --custom_ops_path=./examples/05_custom_ops_symmetric_alltoall
```

> 其中：
> - `--vendor` 参数表示自定义算子标识
> - `--ops` 参数表示自定义算子名称
> - `--custom_ops_path` 参数表示自定义算子工程路径

## 三、安装自定义算子包

自定义算子安装包在 `./build_out` 目录下，通过 `--install` 参数进行安装：

```bash
./build_out/cann-hccl_custom_symmetric_alltoall_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> 其中：
> - `<arch>` 是当前编译环境的系统架构
> - `<ascend_cann_path>` 是可选参数，表示 CANN 软件包安装目录

自定义算子包安装信息如下：

- 头文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_symmetric_alltoall.h`
- 动态库：`${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_symmetric_alltoall.so`
- AICPU 算子描述文件：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/libsymmetric_alltoall_aicpu_kernel.json`
- AICPU 算子包：`${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_symmetric_alltoall.tar.gz`
- 安装脚本：`${ASCEND_HOME_PATH}/opp/vendors/cust/scripts/install.sh`

## 四、执行自定义算子

### 1. 关闭 AICPU 算子验签功能

```bash
# 设置AI CPU算子用户自定义验签能力使能状态，使能开关
for i in {0..7}; do npu-smi set -t custom-op-secverify-enable -i $i -d 1; done

# 设置AI CPU算子验签模式，关闭验签
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done
```

### 2. 修改 AICPU 白名单

AICPU 默认只加载白名单中配置的包，用户自行开发的 AICPU 算子包需配置到白名单中：

```bash
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

将下列内容追加到 `ascend_package_load.ini` 中：

```ini
name:aicpu_hccl_custom_symmetric_alltoall.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
```

### 3. 编译样例

在 `examples/05_custom_ops_symmetric_alltoall/testcase` 代码目录下执行如下命令：

```bash
# 编译样例
make
```

### 4. 执行样例

```bash
# 运行样例
make test
```

### 5. 样例结果示例

对称 AlltoAll 通信后，每个 rank 收到的数据应该是各个 rank 的编号：

```text
Found 8 NPU device(s) available
Device 0: Initialized sendBuf with rank 0
Device 1: Initialized sendBuf with rank 1
Device 2: Initialized sendBuf with rank 2
Device 3: Initialized sendBuf with rank 3
Device 4: Initialized sendBuf with rank 4
Device 5: Initialized sendBuf with rank 5
Device 6: Initialized sendBuf with rank 6
Device 7: Initialized sendBuf with rank 7
Device 0: recvBuf[0]=0.000000, recvBuf[1]=1.000000, recvBuf[2]=2.000000, recvBuf[3]=3.000000
Device 0: Verification PASSED!
Device 1: recvBuf[0]=0.000000, recvBuf[1]=1.000000, recvBuf[2]=2.000000, recvBuf[3]=3.000000
Device 1: Verification PASSED!
...
All devices completed successfully!
```

## 五、对称内存原理

本样例展示了如何使用对称内存实现 zero-copy 的 AlltoAll 通信：

1. **内存分配**：使用 `aclrtReserveMemAddress` 预留虚拟地址，使用 `aclrtMallocPhysical` 分配物理地址，使用 `aclrtMapMem` 建立映射

2. **注册对称窗口**：使用 `HcclCommSymWinRegister` 将内存注册到通信域，获取对称窗口句柄

3. **获取对端地址**：在 AICPU Kernel 中使用 `HcommSymWinGetPeerPointer` 获取对端 rank 的内存地址

4. **Zero-copy 通信**：直接将对端地址用于 `HcommWriteOnThread`，实现数据直接从本地写入对端接收缓冲区，无需中间 buffer 中转

5. **资源释放**：退出时需逆序释放资源：去注册对称窗口 -> 解除映射 -> 释放物理内存 -> 释放虚拟地址
