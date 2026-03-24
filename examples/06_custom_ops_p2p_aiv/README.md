# 自定义通信算子 - 点对点通信 AIV

## 样例介绍

本样例展示如何基于 HCCL AIV 通信编程接口开发点对点自定义通信算子，主要功能点：

1. 基于 AIV 通信引擎实现 `Send/Recv` 点对点算子。
2. 包含 Host 侧算子逻辑与 Device 侧 Kernel 实现。
3. 提供完整的编译构建与测试验证流程。

## 目录结构

```text
├── CMakeLists.txt                      # 根目录编译/构建配置文件
├── op_host/
│   ├── CMakeLists.txt
│   ├── send.cc                         # HcclSendCustomAiv Host 侧实现
│   ├── recv.cc                         # HcclRecvCustomAiv Host 侧实现
│   ├── resource.cc                     # 资源申请与参数组织逻辑
│   ├── launch_kernel.cc                # Kernel 下发逻辑实现
│   └── launch_kernel.h                 # Kernel 下发接口定义
├── op_kernel/
│   ├── CMakeLists.txt
│   ├── p2p_aiv_kernel_base.h           # AIV Kernel 公共基类
│   ├── p2p_send_recv_kernel.h          # Send/Recv Kernel 实现
│   └── launch_kernel_asc.asc           # AIV Kernel 入口
├── inc/
│   ├── hccl_custom_p2p_aiv.h           # 自定义算子对外接口头文件
│   ├── common.h                        # 公共类型定义与宏
│   ├── kernel_types.h                  # Host/Kernel 共享参数结构
│   └── log.h                           # 日志工具
└── testcase/
    ├── CMakeLists.txt                  # 测试用例 CMake 配置文件
    ├── Makefile                        # 测试用例 Makefile
    └── main.cc                         # 测试用例主程序
```

## 一、环境准备

### 1. 环境要求

本样例支持以下昇腾产品：

- Atlas A3 训练系列产品
- Atlas A3 推理系列产品

### 2. 安装 CANN Toolkit 开发套件包

参考 [昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)，安装最新版本 CANN Toolkit 开发套件包。

### 3. 配置环境变量

以 root 用户默认安装路径为例：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

## 二、编译与运行

### 1. 编译自定义算子库

在样例根目录下执行以下命令：

```bash
mkdir build
cd build
cmake ..
make
```

### 2. 运行测试用例

执行 `make run` 前，请先在样例根目录完成一次 CMake 编译，`op_kernel` 构建产物会自动复制到 `testcase/` 目录。

编译完成后，进入 `testcase` 目录执行测试：

```bash
cd ../testcase
make run
```

### 3. 预期结果

运行成功后，终端将输出类似以下日志：

```text
rank 1 verify passed
Test Passed
```
