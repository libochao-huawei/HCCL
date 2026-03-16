#!/bin/bash

JSON_FILE="/mnt/hccl/topo.json"

# 1. 设置 trap，在脚本退出时（包括正常结束、段错误、Ctrl+C）自动执行单引号内的恢复命令
trap 'sed -i '\''s/"protocols": \["UB_MEM"\]/"protocols": \["UB_CTP"\]/g'\'' "$JSON_FILE"; echo -e "\n[清理完成] 已将 $JSON_FILE 中的协议恢复为 UB_CTP"' EXIT

# 2. 将 UB_CTP 修改为 UB_MEM
echo "[准备阶段] 正在将 $JSON_FILE 中的 UB_CTP 修改为 UB_MEM..."
sed -i 's/"protocols": \["UB_CTP"\]/"protocols": \["UB_MEM"\]/g' "$JSON_FILE"

# 3. 切换到 testcase 目录
echo "[执行阶段] 切换到 ../testcase 目录..."
cd ../testcase || { echo "错误: 找不到 ../testcase 目录"; exit 1; }

# 4. 执行测试
echo "[执行阶段] 开始运行 make run..."
make run

# 5. 测试执行完毕（无论 make run 成功还是段错误，bash 会继续执行下一行），切换到 build 目录
echo "[状态切换] make run 结束，准备切换到 ../build 目录..."
cd ../build || { echo "警告: 找不到 ../build 目录"; }

# 脚本执行到末尾自动退出，此时会触发最上方的 trap 恢复 JSON 文件