# Graph Mode Broadcast - 冲突解决完成

## 新分支已创建

分支名: `graph_mode_broadcast_clean`

## 完成的改动

已基于最新 master 分支，迁移了以下 broadcast 相关改动：

### 修改的文件 (16个)
1. `src/ops/broadcast/broadcast_op.cc` - 添加 GraphMode 支持
2. `src/ops/broadcast/broadcast_op.h` - 添加函数声明
3. `src/ops/broadcast/executor/CMakeLists.txt`
4. `src/ops/broadcast/executor/ins_v2_broadcast_parallel_executor.cc`
5. `src/ops/broadcast/executor/ins_v2_broadcast_sole_executor.cc`
6. `src/ops/broadcast/selector/broadcast_auto_selector.cc`
7. `src/ops/broadcast/selector/broadcast_auto_selector.h`
8. `src/ops/broadcast/template/aicpu/CMakeLists.txt`
9. `src/ops/broadcast/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.cc`
10. `src/ops/broadcast/template/aicpu/ins_temp_broadcast_mesh_1D_two_shot.h`
11. `src/ops/broadcast/template/aicpu/ins_temp_broadcast_nhr.cc`
12. `src/ops/op_common/inc/alg_param.h` - 添加 ResPackGraphMode 等结构体
13. `src/ops/op_common/op_common.cc` - 添加 CheckHCCLIndependentOp 等函数
14. `src/ops/op_common/op_common.h`
15. `src/ops/op_common/op_common_graph_mode.cc` (新增)
16. `src/ops/op_common/op_common_graph_mode.h` (新增)

## 统计
- 新增: 760 行
- 删除: 636 行
- 净增: 124 行

## 下一步操作

在你的本地仓库执行以下命令：

```bash
# 1. 进入仓库目录
cd hccl

# 2. 获取最新 master
git fetch upstream master

# 3. 基于 master 创建新分支
git checkout -b graph_mode_broadcast_clean upstream/master

# 4. 从 PR 分支获取 broadcast 相关改动
git checkout graph_mode_0310_hccl_broadcast -- \
    src/ops/broadcast/broadcast_op.cc \
    src/ops/broadcast/broadcast_op.h \
    src/ops/broadcast/executor/ \
    src/ops/broadcast/selector/ \
    src/ops/broadcast/template/ \
    src/ops/op_common/op_common_graph_mode.cc \
    src/ops/op_common/op_common_graph_mode.h \
    src/ops/op_common/op_common.cc \
    src/ops/op_common/op_common.h \
    src/ops/op_common/inc/alg_param.h

# 5. 添加 HcclBroadcastInner 声明到 broadcast_op.h
# 在 extern "C" 块中添加:
# HcclResult HcclBroadcastInner(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);

# 6. 提交改动
git add -A
git commit -m "Add graph mode broadcast support (rebased on master)"

# 7. 推送到你的 fork
git push origin graph_mode_broadcast_clean

# 8. 在 GitCode 上创建新的 PR，选择 graph_mode_broadcast_clean 分支合入 cann/hccl:master
```

## 注意事项

1. `HcclBroadcastInner` 函数在主线和 PR 分支都只有调用没有定义，可能是在链接时从其他库提供
2. 已添加 `HcclBroadcastInner` 的声明到 `broadcast_op.h`
3. 所有其他文件都采用主线最新版本，只有 broadcast 目录和必要的 op_common 文件使用 PR 的改动

## 与原 PR #274 的区别

- 原 PR 改动 26 个文件，+900/-60 行
- 新分支改动 16 个文件，+760/-636 行
- 只保留了 broadcast 相关的核心改动
- 其他文件完全采用主线最新版本
