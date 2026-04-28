# HCCL 仓库资料参考索引

> 基于 `/home/xpzhang/hccl_test_claw/hccl` 仓库整理，2026-04-27

---

## 一、文档类

| 资料 | 主题 | 路径 |
|------|------|------|
| 源码构建指南 | 环境准备、CANN 安装、编译流程、环境变量配置 | `docs/build.md` |
| API 文档索引 | 所有 HCCL API 分类索引 | `docs/api/context/index.md` |
| 集合通信 API | AllGather/AllReduce/Broadcast/ReduceScatter 等 11 个算子 API | `docs/api/context/Hccl*.md` |
| 点对点通信 API | Send/Recv/BatchSendRecv | `docs/api/context/HcclSend.md` 等 |
| 数据类型/结果码 | HcclDataType/HcclReduceOp/HcclResult/HcclComm | `docs/api/context/HcclDataType.md` 等 |
| 算法分析器使用指导 | ST 仿真原理、用例编写 5 步法、内存冲突定位、语义校验 | `test/st/algorithm/README.md` |
| HCCL 资料书架 | 用户指南链接、技术文章、B 站培训视频（原语/算法/业务开发） | `docs/README.md` |

## 二、代码参考类

### 现有算子实现（可直接参考的模板）

| 算子 | 位置 |
|------|------|
| AllGather | `src/ops/all_gather/` |
| AllGatherV | `src/ops/all_gather_v/` |
| AllReduce | `src/ops/all_reduce/` |
| Broadcast | `src/ops/broadcast/` |
| ReduceScatter | `src/ops/reduce_scatter/` |
| ReduceScatterV | `src/ops/reduce_scatter_v/` |
| Scatter | `src/ops/scatter/` |
| Reduce | `src/ops/reduce/` |
| AllToAllV | `src/ops/all_to_all_v/` |
| Send/Recv | `src/ops/send/`, `src/ops/recv/` |
| BatchSendRecv | `src/ops/batch_send_recv/` |

### 公共基础设施

| 资料 | 作用 | 路径 |
|------|------|------|
| Template 基类 | 纯虚接口定义：KernelRun/CalcRes/GetRes/CalcScratchMultiple | `src/ops/op_common/template/alg_v2_template_base.h` |
| Executor 基类 | 执行器接口：CalcAlgHierarchyInfo/CalcRes/Orchestrate/FastLaunch | `src/ops/op_common/executor/executor_v2_base.h` |
| 注册宏定义 | REGISTER_EXEC_V2：tag → Executor → TopoMatch → Template | `src/ops/op_common/executor/registry/coll_alg_v2_exec_registry.h` |
| Selector 注册 | 选择器注册和匹配逻辑 | `src/ops/op_common/selector/` |
| TopoMatch 类族 | TopoMatch1D/UBX/PcieMix/Multilevel/Base | `src/ops/op_common/topo/` |
| 通道管理 | CreateChannelRequestByRankId/CalcChannelRequestMesh1D | `src/ops/op_common/executor/channel/` |
| 数据搬运 Wrapper | LocalCopy/SendRecvRead/SendRecvWrite/DataSlice | `src/ops/op_common/template/wrapper/alg_data_trans_wrapper.cc` |
| 参数定义 | OpParam/TemplateDataParams/TemplateResource/DATATYPE_SIZE_TABLE | `src/ops/op_common/inc/alg_param.h` |
| CMakeLists 模板 | 每个算子各层级的 CMakeLists 写法 | `src/ops/<op>/template/<engine>/CMakeLists.txt` |

## 三、示例代码类

| 资料 | 主题 | 路径 |
|------|------|------|
| 集合通信示例（9 个） | AllReduce/Broadcast/AllGather/ReduceScatter 等的调用示例 + README | `examples/02_collectives/` |
| 点对点通信示例（2 个） | SendRecv / BatchSendRecv Ring | `examples/01_point_to_point/` |
| 自定义算子示例 | 基于 AICPU 引擎实现自定义 Send/Recv 和 AllGather 算子 | `examples/04_custom_ops_p2p/`, `05_custom_ops_allgather/` |
| AI 框架集成 | PyTorch / TensorFlow 集成示例 | `examples/03_ai_framework/` |

## 四、测试类

| 资料 | 主题 | 路径 |
|------|------|------|
| ST 测试用例（6 个） | AllGather(AICPU+DPU)/AllReduce(普通+Parallel)/Scatter/ReduceScatter | `test/st/algorithm/testcase/` |
| ST CMakeLists | 算子源文件如何加入 ST 编译 | `test/st/algorithm/utils/src/aicpu/CMakeLists.txt` |
| ST 公共库 CMakeLists | hccl-model-proxy 链接配置 | `test/st/algorithm/utils/src/CMakeLists.txt` |

## 五、编译脚本类

| 资料 | 主题 |
|------|------|
| `build.sh` | 编译入口：--pkg/--full/--ut/--clean 参数解析 |
| 根 `CMakeLists.txt` | 项目级编译配置、third_party 依赖 |

## 六、Skills（已安装）

| Skill | 定位 | 模式 |
|-------|------|------|
| `hccl-algorithm-dev` | 开发新 HCCL 算法 | 交互式 |
| `hccl-algorithm-dev-quick` | 快速开发新 HCCL 算法 | 自主式 |
| `hccl-algorithm-opt` | 优化已有 HCCL 算法 | 交互式 |
| `hccl-algorithm-opt-quick` | 快速优化 HCCL 算法 | 自主式 |
| `hccl-st-testcase-dev` | 为算法添加 ST 测试用例 | 交互式 |
| `hccl-st-testcase-quick` | 快速添加 ST 测试用例 | 自主式 |
| `hccl-local-compile` | 本地编译 HCCL | 自主式 |
| `hccl-local-compile-interactive` | 交互式编译 HCCL | 交互式 |

---

## 资料与开发步骤的对应关系

### 新算法开发（AICPU 为例）

| 步骤 | 需要的资料 | 对应资料 | 支撑度 |
|------|-----------|---------|--------|
| 1. 理解算子架构 | 四层架构、各层职责 | 现有算子代码（`src/ops/all_gather/`）+ Skill | ✅ 充分 |
| 2. 设计算法 | 通信拓扑、数据流、通道需求 | ST README（原理介绍）+ 现有 Template 代码 | ✅ 充分 |
| 3. 实现 Template | 继承基类、实现接口 | `alg_v2_template_base.h` + 现有 Template .h/.cc + `alg_data_trans_wrapper.cc` | ✅ 充分 |
| 4. 注册 Executor | REGISTER_EXEC_V2 宏用法 | `coll_alg_v2_exec_registry.h` + 现有 Executor .cc | ✅ 充分 |
| 5. 添加 Selector | 匹配逻辑写法 | `all_gather_auto_selector.cc`（含多维度匹配示例） | ✅ 充分 |
| 6. 更新 CMakeLists | CMake 写法 | `src/ops/all_gather/template/aicpu/CMakeLists.txt` | ✅ 充分 |
| 7. 编译验证 | 编译命令 | `docs/build.md` + `build.sh` + `hccl-local-compile` Skill | ✅ 充分 |

### ST 验证

| 步骤 | 需要的资料 | 对应资料 | 支撑度 |
|------|-----------|---------|--------|
| 1. 理解 ST 原理 | 仿真模型、Task 图、校验机制 | `test/st/algorithm/README.md`（原理 + 5 步法 + 问题定位） | ✅ 充分 |
| 2. 添加源文件 | ST CMakeLists 写法 | `test/st/algorithm/utils/src/aicpu/CMakeLists.txt` | ✅ 充分 |
| 3. 编写测试用例 | TopoMeta 写法、Run 函数模式、校验函数 | `all_gather_aicpu_testcase.cc`（完整模板） | ✅ 充分 |
| 4. 编译 ST | cmake 参数、环境变量 | `hccl-st-testcase-dev` Skill + ST README | ✅ 充分 |
| 5. 运行验证 | gtest filter、日志分析 | ST README（结果示例 + 问题定位方法） | ✅ 充分 |

### 已知不足

| 缺失项 | 说明 |
|--------|------|
| 算法设计理论指南 | 没有文档说明"什么场景该选什么算法"，只能靠看代码推断 |
| 性能基准数据 | 没有各算法的性能对比数据，无法评估新算法的优劣 |
| 硬件拓扑详解 | 没有 HCCS/RoCE/PCIe 的带宽差异和延迟特性文档 |
| 上板测试流程 | ST 只是仿真，真实 NPU 测试需要 HCCL Test 工具，仓库里没有对应指导 |

---

*最后更新：2026-04-27*
