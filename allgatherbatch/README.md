# AllGatherBatch

`hccl/allgatherbatch` 是一个自包含的批量 AllGather 自定义算子实现。
工程整体参考 `examples/04_custom_ops_p2p` 的组织方式，当前所有自定义逻辑都收敛在本目录内。

## 目录结构

```text
hccl/allgatherbatch/
├─ README.md
├─ CMakeLists.txt
├─ inc/
│  ├─ allgather_batch.h
│  ├─ common.h
│  ├─ log.h
│  └─ resource_request.h
├─ op_host/
│  ├─ CMakeLists.txt
│  ├─ allgather_batch.cc
│  ├─ load_kernel.cc
│  ├─ load_kernel.h
│  ├─ launch_kernel.cc
│  └─ launch_kernel.h
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
│  ├─ all_gather_nhr_core.h
│  └─ liballgatherbatch_aicpu_kernel.json
└─ testcase/
   ├─ CMakeLists.txt
   ├─ Makefile
   ├─ README.md
   ├─ main.cc
   └─ run.sh
```

## 模块说明

- `inc/`
  放公开 API、Host/Device 共享协议、日志宏和资源请求定义。
- `op_host/`
  放 Host 侧入口、拓扑解析、资源申请、AICPU kernel 加载和 launch 逻辑。
- `op_kernel_aicpu/`
  放 Device 侧入口、执行器、`HDStage` core、`NHR` core 和 AICPU kernel JSON。
- `testcase/`
  放可运行样例、Makefile 和使用说明。

## 主流程

1. `HcclAllGatherBatch` 校验输入并构造 `OpParam`。
2. Host 侧准备拓扑信息、资源请求和动态 `AlgResourceCtx`，然后加载并下发 AICPU kernel。
3. Device 侧从 `ExecOp` 进入执行器，按 `Pack -> HDStageCore -> NHRCore -> Unpack` 的链路执行。
4. `testcase` 目录提供可调 `token bytes / scale count / devices / warmup / iters` 的样例程序。

## 当前范围

- 支持单 server 和同一 superpod 内的跨 server 场景。
- 不支持跨 superpod 链路。
- 当前资源模型按 fullmesh 建链：每个 rank 与其余 rank 都建立 channel。
- 当全部数据无法放进单次本地窗口时，会自动切成多窗口执行。

## 推荐阅读入口

- 公开 API：`inc/allgather_batch.h`
- 共享协议：`inc/common.h`
- Host 主链：`op_host/allgather_batch.cc`
- Device 入口：`op_kernel_aicpu/hccl_allgather_batch_aicpu_kernel.cc`
- 执行器：`op_kernel_aicpu/allgather_batch_small_count_executor.cc`
- 测试样例：`testcase/main.cc`