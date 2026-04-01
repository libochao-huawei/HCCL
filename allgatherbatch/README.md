# AllGatherBatch

`hccl/allgatherbatch` 是一个自包含的批量 AllGather 自定义算子实现。
它整体参考 `04_custom_ops_p2p` 的组织方式：
只修改本目录代码，并且仅依赖 `hcomm` 公共头文件与 ACL 公共头文件。

## 目录说明

```text
hccl/allgatherbatch/
├─ README.md
├─ CMakeLists.txt
├─ inc/
│  ├─ allgather_batch.h
│  ├─ common.h
│  └─ log.h
├─ op_host/
│  ├─ CMakeLists.txt
│  ├─ allgather_batch.cc
│  ├─ launch_kernel.cc
│  ├─ launch_kernel.h
│  ├─ load_kernel.cc
│  └─ load_kernel.h
├─ op_kernel_aicpu/
│  ├─ CMakeLists.txt
│  ├─ hccl_allgather_batch_aicpu_kernel.cc
│  ├─ exec_op.cc
│  ├─ exec_op.h
│  ├─ allgather_batch_small_count_executor.cc
│  ├─ allgather_batch_small_count_executor.h
│  ├─ window_range.h
│  ├─ all_gather_hd_stage_core.cc
│  ├─ all_gather_hd_stage_core.h
│  ├─ all_gather_nhr_core.cc
│  └─ all_gather_nhr_core.h
└─ testcase/
   ├─ CMakeLists.txt
   ├─ Makefile
   ├─ README.md
   ├─ main.cc
   └─ run.sh
```

- `inc/`
  放公开 API 头文件，以及 Host 和 Device 共用的协议定义。
- `op_host/`
  放 Host 侧入口、拓扑准备、资源申请、kernel load 和 launch 逻辑。
- `op_kernel_aicpu/`
  放 Device 侧 kernel 入口、执行器、HDStage core 和 NHR core。
- `testcase/`
  放可运行样例、Makefile 辅助入口和使用说明。

## 主流程

1. `HcclAllGatherBatch` 校验用户传入的 item，并构造 `OpParam`。
2. Host 侧准备 `TopoInfo`、`AlgResourceCtx`，然后下发 AICPU kernel。
3. Device 侧从 `ExecOp` 进入执行器主循环，按
   `Pack -> HDStageCore -> NHRCore -> Unpack`
   这条链路执行。
4. `testcase` 目录提供了可调 `token bytes / scale count / devices / warmup / iters` 的样例程序。

## 当前范围

- 支持单 server 和超节点内跨 server 场景。
- 不支持跨 superpod 链路。
- 当全部数据放不进单次本地窗口时，会自动走窗口化执行。
- 所有自定义逻辑都收敛在 `hccl/allgatherbatch` 目录下。

## 推荐阅读入口

- 公开 API：`inc/allgather_batch.h`
- 共享协议：`inc/common.h`
- Host 入口：`op_host/allgather_batch.cc`
- Device 入口：`op_kernel_aicpu/hccl_allgather_batch_aicpu_kernel.cc`
- 执行器：`op_kernel_aicpu/allgather_batch_small_count_executor.cc`
- 测试样例：`testcase/main.cc`
