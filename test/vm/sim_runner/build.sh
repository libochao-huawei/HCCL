#!/bin/bash
set -e
trap 'echo "❌ Error occurred in build.sh at line $LINENO"; exit 1' ERR

# 获取shell脚本目录并获取TOP_DIR(work_code)目录
SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
TOP_DIR="$SHELL_DIR"
for ((i = 0; i < 4; i++)); do
    TOP_DIR=$(dirname "$TOP_DIR")
done
echo $TOP_DIR

# 进入work_code目录创建st目录并进去
cd $TOP_DIR
mkdir -p ./hcclso_build && cd ./hcclso_build/ && rm -rf ../hcclso_build/* && rm -rf ../output/ascend/*

# 执行HCCL编译并拷贝所需so
echo "HCCL build start..."
# 编译host侧hccl.so等
cmake ../cmake/superbuild/ -DHOST_PACKAGE="hccl" -DPRODUCT=ascend
TARGETS=hccl make host -j20

# 编译device侧ccl_kernel.so等
TARGETS=ccl_kernel make host -j20

# 编译成功并退出
echo "HCCL build finish..."
exit 0