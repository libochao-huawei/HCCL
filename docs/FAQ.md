# HCCL 开发常见问题 FAQ

> 基于实际开发经验整理，适用于 HCCL 代码仓库（https://gitcode.com/cann/hccl）开发者。
> 最后更新：2026-04-27

---

## 一、环境准备

### 1.1 `dlog_pub.h`、`securec.h` 头文件找不到

**现象**：编译时报 `fatal error: dlog_pub.h: No such file or directory` 或 `securec.h` 找不到。

**原因**：CANN 环境变量未正确加载。

**解决**：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/cann/set_env.sh

# 验证路径
echo $ASCEND_HOME_PATH
ls $ASCEND_HOME_PATH/x86_64-linux/include/dlog_pub.h
```

> ⚠️ 如果用 `exec` 执行命令，默认用 `sh` 不支持 `source`，需改为：
> ```bash
> bash -c 'source /usr/local/Ascend/cann/set_env.sh && ...'
> ```

### 1.2 `ASCEND_HOME_PATH` 环境变量未设置或路径不完整

**现象**：CMake 配置时报头文件找不到，或路径指向错误目录。

**原因**：环境变量未设置，或 CMakeLists 中拼接路径后不完整。

**解决**：编译时显式通过 `-D` 传入：

```bash
cmake -S . -B build -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
```

ST 编译时推荐写法：

```bash
bash -c '
  source /usr/local/Ascend/cann/set_env.sh
  cmake -S . -B build -DBUILD_OPEN_PROJECT=ON -DASCEND_HOME_PATH=$ASCEND_HOME_PATH
  make -j8
'
```

### 1.3 CANN Toolkit 版本怎么确认？

```bash
cat /usr/local/Ascend/cann/x86_64-linux/ascend_toolkit_install.info
```

输出示例：
```
package_name=Ascend-cann-toolkit
version=9.0.0
innerversion=V100R001C25B046
```

---

## 二、编译构建

### 2.1 编译 HCCL 的命令是什么？

在 HCCL 仓库根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh <参数>
```

常用参数：

| 参数 | 作用 | 说明 |
|------|------|------|
| `--pkg` | 生成 `.run` 安装包 | 日常开发推荐 |
| `--full` | 同时编译 host + device | **打包必需** |
| `--ut` | 运行单元测试 | 按需 |
| `--clean` | 清理后编译 | 首次或大改后 |

常用组合：

| 组合 | 用途 |
|------|------|
| `--pkg --full` | 编译 + 打包（最常用） |
| `--pkg --full --ut` | 编译 + 打包 + 测试 |
| `--ut` | 仅跑测试（已编译过时跳过） |

### 2.2 不带 `--full` 打包为什么会失败？

`--full` 同时编译 host 端和 device 端。不带 `--full` 时缺少 device 端文件：

- `build_device/aicpu_hccl.tar.gz`
- `build_device/aicpu_custom/libscatter_aicpu_kernel.so`

打包脚本会检查这些文件，缺失则报错。

**解决**：加 `--full` 重新编译。

### 2.3 编译后产物在哪？怎么确认编译成功？

```bash
# 检查 .run 安装包
ls build_out/cann-hccl_*.run

# 检查 host 端动态库
ls build/libhccl.so

# 检查 device 端文件（仅 --full 模式）
ls build_device/aicpu_hccl.tar.gz
ls build_device/aicpu_custom/libscatter_aicpu_kernel.so
```

编译成功的标志：
- 终端显示 `CPack: - package: cann-hccl_xxx.run`
- `build_out/` 目录下生成 `.run` 文件
- 无 error 日志

### 2.4 host 和 device 编译产物有什么区别？

| 维度 | host（CPU 侧） | device（NPU 侧） |
|------|---------------|-----------------|
| 产物 | `libhccl.so`、头文件 | `aicpu_hccl.tar.gz`、`.o` 文件 |
| 运行在 | x86_64 / ARM CPU | 昇腾 NPU（910B/910C） |
| 职责 | API 层、通信域管理、算子调度 | AICPU 算子实现、AIV 矢量算子 |

### 2.5 编译慢怎么加速？

- 使用 `ccache`（可选依赖）提高二次编译速度
- 设置并行编译线程数：`export CPU_NUM=16`（根据实际 CPU 核数调整）
- 非首次编译不需要 `--clean`，增量编译更快

### 2.6 卸载已安装的 HCCL 包

```bash
bash ./build_out/cann-hccl_<版本号>_linux-<架构>.run --uninstall
```

---

## 三、Git 操作

### 3.1 `git pull` 和 `git fetch + rebase` 有什么区别？

| 方式 | 效果 | 推荐 |
|------|------|------|
| `git pull` | 拉取后产生 merge commit | ❌ 不推荐，污染提交历史 |
| `git fetch + rebase` | 拉取后变基，保持线性历史 | ✅ 推荐 |

```bash
# 推荐做法
git fetch upstream
git rebase upstream/master
```

### 3.2 rebase 后 push 为什么要加 `-f`？

`git rebase` 会重写本地提交历史（commit hash 变化），与远程分支产生分歧。

```bash
git push zhangxp1030 feature/xxx -f
```

> ⚠️ 仅在个人 Fork 分支上使用，不要对公共分支执行 force push。

### 3.3 Commit Message 格式规范

```
type(scope): 简短描述

详细说明（可选）
- 具体改动 1
- 具体改动 2
```

**type**：`feat` / `fix` / `docs` / `refactor` / `perf` / `test` / `chore`

**scope**：变更影响的模块名

**示例**：
```
docs(topo): add HCCL topology module development guide
feat(algorithm): add ring pipeline algorithm for AllGather
fix(selector): fix data size threshold matching logic
```

### 3.4 MR 创建后怎么触发门禁检查？

在 MR 页面评论框中输入：

```
/compile
```

等待门禁检查完成后，PR 会被分配给审查者。

### 3.5 MR 描述模板

```markdown
## 描述
<!-- 在这里详细描述你的改动，包括改动的原因和所采取的方法。 -->

## 关联的Issue
<!-- 如果这个PR是为了解决特定的Issue，请在这里提供Issue链接。 -->
<!-- 如果这个PR不涉及Issue，可填写"NA"。 -->
NA

## 测试
<!-- 描述进行了哪些测试来验证你的改动。包括但不限于构造对应xx测试用例、二级冒烟、算子泛化等。 -->
纯文档变更，不涉及代码逻辑改动，无需功能测试。
# 或：已完成本地编译和 LLT 测试。

## 文档更新
<!-- 如果这个PR包含文档的更新，请在这里指出。 -->
新增以下文件：
- `docs/分类名/xxx.md` — 主文档
- `docs/figures/分类名/xxx.*` — 附图

## 类型标签
<!-- [x] 表示选中 -->
- [ ] Bug修复
- [ ] 新特性
- [ ] 性能优化
- [x] 文档更新  # 按需勾选
- [ ] 其他，请描述：
```

---

## 四、代码提交流程

### 4.1 首次环境配置

```bash
# 1. 配置 Git 用户信息
git config --global user.name "<你的用户名>"
git config --global user.email "<你的邮箱>"  # 与 CLA 签署邮箱一致

# 2. 配置 SSH 公钥（仅需首次）
ssh-keygen -t rsa -C "<你的邮箱>"
cat ~/.ssh/id_rsa.pub   # 复制到 GitCode → 个人设置 → 安全设置 → SSH 公钥
ssh -T git@gitcode.com  # 验证：应显示 Welcome to GitCode, <用户名>

# 3. Fork 仓库
# 在 GitCode 上打开 https://gitcode.com/cann/hccl，点击 Fork 创建个人分支

# 4. 克隆并配置上游
git clone git@gitcode.com:<你的用户名>/hccl.git
cd hccl
git remote add upstream https://gitcode.com/cann/hccl.git
git remote add <你的用户名> git@gitcode.com:<你的用户名>/hccl.git
```

### 4.2 完整提交流程

**步骤 1：创建个人开发分支**

```bash
cd hccl
git checkout master
git fetch upstream
git rebase upstream/master    # 保持与上游同步
git checkout -b feature/xxx   # xxx 为本次修改的简短描述
```

**步骤 2：修改代码或文档**

| 内容类型 | 放置路径 | 说明 |
|---------|---------|------|
| 文档 Markdown | `docs/分类名/xxx.md` | 如 `docs/topo/xxx.md` |
| 文档图片 | `docs/figures/分类名/xxx.svg` | 文档中用相对路径 `../figures/分类名/xxx.svg` 引用 |
| 代码 | 按模块放到对应目录 | 参考已有结构 |

**步骤 3：本地构建验证（代码变更时）**

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --full   # 编译
bash build.sh --ut           # 单元测试
```

纯文档变更可跳过此步。

**步骤 4：同步上游 master**

```bash
# 在 feature/xxx 分支上执行
git fetch upstream
git rebase upstream/master
```

> ⚠️ **重要**：始终使用 `fetch + rebase`，不要用 `git pull`（会产生合并提交）。

**步骤 5：提交本地变更**

```bash
git add <修改的文件>
git commit -m "type(scope): 简短描述

详细说明（可选）
- 具体改动 1
- 具体改动 2"
```

如需在上次提交基础上继续修改：

```bash
git add <新增修改>
git commit --amend
```

**步骤 6：推送到个人 Fork**

```bash
git push <你的用户名> feature/xxx
```

若执行过 `git rebase` 导致本地与远程基准不同：

```bash
git push <你的用户名> feature/xxx -f
```

**步骤 7：创建 Merge Request**

1. 访问 `https://gitcode.com/<你的用户名>/hccl/merge_requests/new?source_branch=feature/xxx`
2. 设置目标分支为 `cann/hccl → master`
3. 填写 MR 描述（见 3.5 MR 描述模板）
4. 提交创建

**步骤 8：触发门禁检查**

在 MR 页面评论框中输入：

```
/compile
```

等待门禁检查完成后，PR 会被分配给审查者。

**步骤 9：查看审查意见并修复**

门禁通过后，根据审查意见修改：

```bash
# 修改代码后
git add <修改的文件>
git commit --amend                    # 修改上次提交
# 或 git commit -m "fix: address review comments"

git push <你的用户名> feature/xxx -f   # 强制推送更新
```

### 4.3 常用操作速查

**回退提交**：

```bash
git checkout -b myrevert
git fetch upstream
git rebase upstream/master
git revert SHA          # 回退单个提交
git push origin myrevert
```

**处理冲突**：

```bash
git checkout master
git fetch upstream
git rebase upstream/master
git checkout feature/xxx
git rebase master          # 解决冲突后
git add <修改的文件>
git rebase --continue
git push -f origin feature/xxx
```

**合并提交**：

```bash
git log                    # 查看提交历史
git rebase -i HEAD~n       # n 为要合并的提交数
# 将 pick 改为 squash，保留一个 pick
# 编辑提交信息后保存
git push -f origin feature/xxx
```

### 4.4 关键注意事项

1. **永远用 `fetch + rebase` 同步上游**，不要用 `git pull`
2. **SSH 公钥必须配置**，否则每次 push 都要输入密码
3. **个人 Fork 不能直接 push 到 cann/hccl**，必须通过 MR 合入
4. **rebase 后 push 必须加 `-f`**（强制推送）
5. **Commit Message 要清晰**，type 和 scope 必填
6. **MR 创建后需手动评论 `/compile` 触发门禁**
7. **门禁通过后才会有审查者分配**

---

## 五、常见问题速查表

| 场景 | 现象 | 原因 | 快速解决 |
|------|------|------|---------|
| 编译失败 | `dlog_pub.h: No such file` | CANN 环境未加载 | `bash -c 'source /usr/local/Ascend/cann/set_env.sh && ...'` |
| 打包失败 | 缺少 device 文件 | 未加 `--full` | `bash build.sh --pkg --full` |
| ST 编译失败 | `ASCEND_HOME_PATH` 相关报错 | CMake 未收到路径 | `cmake ... -DASCEND_HOME_PATH=$ASCEND_HOME_PATH` |
| ST 运行失败 | 找不到 stub 库 | `LD_LIBRARY_PATH` 未设置 | `export LD_LIBRARY_PATH="${PWD}/build/utils/src/hccl_depends_stub:${LD_LIBRARY_PATH}"` |
| 链接错误 | `undefined reference to Xxx` | 缺源文件 | 在 CMakeLists 的 `scatter_aicpu_kernel` 中添加对应 `.cc` |
| 链接错误 | `TopoMatchPcieMix` 未定义 | ST CMakeLists 缺 `topo_match_pcie_mix.cc` | 在 ST 的 CMakeLists 中添加该源文件 |

---

*本文档基于实际开发经验整理，后续持续补充。*
