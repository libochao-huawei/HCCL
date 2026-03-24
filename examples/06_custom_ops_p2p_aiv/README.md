# 06_custom_ops_p2p_aiv

## 简介
本样例展示如何基于 `hcomm/hccl` 公开接口实现 **AIV kernel 版** 点对点自定义算子。

本样例支持以下昇腾产品：
- Atlas A3 训练系列产品
- Atlas A3 推理系列产品

首版能力范围：
- `HcclSendCustomAiv`
- `HcclRecvCustomAiv`
- `op_host + op_kernel` 的最小 AIV 自定义算子闭环
- 2 卡单向 `send -> recv`

## 目录
- `inc/`: 对外头文件、公共常量、kernel 参数结构、日志宏
- `op_host/`: 资源申请、远端同步区交换、kernel launch
- `op_kernel/`: AIV kernel 入口和 send/recv kernel 实现
- `testcase/`: 2 卡联调用例

## 当前实现说明
当前版本的执行链路如下：
- Host 侧申请 AIV thread、channel、本地 HCCL buffer、远端 HCCL buffer
- Host 侧通过 `HcclEngineCtxCreate + HcclCommMemReg + HcclChannelGetRemoteMems` 交换双方同步区地址
- Host 侧组织 `P2pAivKernelParam` 并通过 `aclrtLaunchKernelWithHostArgs` 下发 AIV kernel
- Send kernel 执行 `input -> local HCCL buffer`，置 `ready`
- Recv kernel 等待 `ready` 后执行 `remote HCCL buffer -> output`，置 `done`

## 构建
```bash
cmake -S . -B build -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake --build build -j
```

## 运行
```bash
./build/testcase/test_custom_p2p_aiv
```

运行前请确保：
- 机器上至少有 2 张可用 NPU
- `ASCEND_CANN_PACKAGE_PATH` 指向正确的 CANN 安装目录
- `LD_LIBRARY_PATH` 已包含 `${ASCEND_CANN_PACKAGE_PATH}/lib64` 和 `build/op_host`
- kernel object `hccl_custom_p2p_aiv_kernels.o` 已由构建阶段复制到 `build/testcase/`
