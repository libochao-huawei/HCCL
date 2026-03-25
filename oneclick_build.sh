#!/bin/bash

set -e

echo "========================================"
echo "Ascend 一键编译脚本"
echo "版本: 1.1.0"
echo "功能: 智能编译 hcomm 或 hccl 仓库"
echo "特点: 自动检测现有代码，跳过不必要的步骤"
echo "========================================"

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请指定要编译的仓库类型"
    echo "用法: $0 hcomm <repository-url> [--dir <directory>]"
    echo "       $0 hccl [--dir <directory>]"
    exit 1
fi

REPO_TYPE="$1"
REPO_URL="$2"
CUSTOM_DIR=""

# 解析命令行参数
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)
            if [ $# -lt 2 ]; then
                echo "错误: --dir 参数需要指定目录"
                exit 1
            fi
            CUSTOM_DIR="$2"
            shift 2
            ;;
        *)
            if [ -z "$REPO_URL" ]; then
                REPO_URL="$1"
            fi
            shift
            ;;
    esac
done

# 通用配置
CANN_VERSION="8.5.0"
CANN_BETA_VERSION="8.5.0-beta.1"
SYSTEM_DEPS="gcc g++ cmake ccache python3-pip unzip"

# 根据仓库类型设置特定配置
case "$REPO_TYPE" in
    "hcomm")
        DEFAULT_REPO_DIR="hcomm"
        CANN_INSTALLER="Ascend-cann-toolkit_${CANN_BETA_VERSION}_linux-x86_64.run"
        CANN_VERSION_USE="${CANN_BETA_VERSION}"
        BRANCH="upstream/8.5.0-beta.1"
        BUILD_CMD="bash build.sh --pkg -p ~/Ascend/ascend-toolkit/latest/cann-${CANN_BETA_VERSION}"
        OUTPUT_DIR="output"
        ;;
    "hccl")
        DEFAULT_REPO_DIR="hccl"
        CANN_INSTALLER="Ascend-cann-toolkit_${CANN_VERSION}_linux-x86_64.run"
        CANN_VERSION_USE="${CANN_VERSION}"
        BRANCH="8.5.0"
        BUILD_CMD="bash build.sh --pkg"
        OUTPUT_DIR="build_out"
        ;;
    *)
        echo "错误: 不支持的仓库类型 '$REPO_TYPE'"
        echo "支持的类型: hcomm, hccl"
        exit 1
        ;;
esac

# 智能目录检测
REPO_DIR=""
if [ -n "$CUSTOM_DIR" ]; then
    # 使用自定义目录
    REPO_DIR="$CUSTOM_DIR"
    echo "[智能检测] 使用指定目录: $REPO_DIR"
elif [ -d "$DEFAULT_REPO_DIR" ]; then
    # 使用默认目录
    REPO_DIR="$DEFAULT_REPO_DIR"
    echo "[智能检测] 使用默认目录: $REPO_DIR"
elif [ -f "build.sh" ]; then
    # 当前目录就是仓库目录
    REPO_DIR="."
    echo "[智能检测] 使用当前目录"
else
    # 需要克隆仓库
    REPO_DIR="$DEFAULT_REPO_DIR"
    echo "[智能检测] 需要克隆仓库到: $REPO_DIR"
fi

# 步骤1: 克隆仓库（仅当需要时）
if [ ! -d "$REPO_DIR" ] || [ ! -f "$REPO_DIR/build.sh" ]; then
    echo "\n[步骤1] 克隆 $REPO_TYPE 仓库..."
    
    if [ "$REPO_TYPE" = "hcomm" ]; then
        if [ -z "$REPO_URL" ]; then
            echo "错误: 编译 hcomm 需要提供仓库地址"
            echo "用法: $0 hcomm <repository-url>"
            exit 1
        fi
        git clone "$REPO_URL" "$REPO_DIR"
        cd "$REPO_DIR"
        echo "[步骤1] 添加 upstream 远程仓库..."
        git remote add upstream "$REPO_URL" 2>/dev/null || echo "[步骤1] upstream 远程仓库已存在"
        echo "[步骤1] 切换到 $BRANCH 分支..."
        git checkout "$BRANCH" 2>/dev/null || echo "[步骤1] 已在 $BRANCH 分支"
    elif [ "$REPO_TYPE" = "hccl" ]; then
        git clone "https://gitcode.com/cann/hccl.git" "$REPO_DIR"
        cd "$REPO_DIR"
        echo "[步骤1] 切换到 $BRANCH 分支..."
        git checkout "$BRANCH" 2>/dev/null || echo "[步骤1] 已在 $BRANCH 分支"
    fi
else
    echo "\n[智能检测] 仓库已存在，跳过克隆步骤"
    cd "$REPO_DIR"
    # 确保在正确的分支
    if git branch --show-current | grep -q "$BRANCH"; then
        echo "[智能检测] 已在正确分支: $(git branch --show-current)"
    else
        echo "[智能检测] 切换到正确分支: $BRANCH"
        git checkout "$BRANCH" 2>/dev/null || echo "[智能检测] 分支切换失败，使用当前分支"
    fi
fi

# 步骤2: 安装系统依赖
echo "\n[步骤2] 检查系统依赖..."

# 检查依赖是否已安装
MISSING_DEPS=""
for dep in $SYSTEM_DEPS; do
    if ! dpkg -l | grep -q "^ii.*$dep\s"; then
        MISSING_DEPS="$MISSING_DEPS $dep"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "[步骤2] 安装缺失的依赖: $MISSING_DEPS"
    sudo apt update
    sudo apt install -y $MISSING_DEPS
else
    echo "[智能检测] 所有系统依赖已安装，跳过安装步骤"
fi

# 步骤3: 检查并安装 CANN Toolkit
echo "\n[步骤3] 检查 CANN Toolkit ${CANN_VERSION_USE}..."

# 检查 CANN Toolkit 是否已安装
CANN_INSTALLED=false
CANN_ACTUAL_PATH=""

# 检查实际的 CANN 安装路径
ACTUAL_CANN_PATHS=(
    "$HOME/Ascend/ascend-toolkit/latest/cann-${CANN_VERSION_USE}"
    "$HOME/Ascend/ascend-toolkit/latest/cann-8.5.0-beta.1"
    "$HOME/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit/latest/cann-${CANN_VERSION_USE}"
    "/usr/local/Ascend/ascend-toolkit/latest/cann-8.5.0-beta.1"
    "/usr/local/Ascend/ascend-toolkit/latest"
)

for CANN_PATH in "${ACTUAL_CANN_PATHS[@]}"; do
    if [ -d "$CANN_PATH" ]; then
        if [ -f "$CANN_PATH/include/hccl/hccl.h" ] || [ -f "$CANN_PATH/x86_64-linux/include/hccl/hccl.h" ]; then
            echo "[智能检测] CANN Toolkit 已安装在: $CANN_PATH"
            CANN_INSTALLED=true
            CANN_ACTUAL_PATH="$CANN_PATH"
            break
        fi
    fi
done

# 检查环境变量
if [ "$CANN_INSTALLED" = false ]; then
    if [ -n "$ASCEND_HOME_PATH" ] || [ -n "$ASCEND_CANN_PACKAGE_PATH" ]; then
        echo "[智能检测] 通过环境变量检测到 CANN Toolkit"
        CANN_INSTALLED=true
    fi
fi

if [ "$CANN_INSTALLED" = false ]; then
    echo "[步骤3] 跳过 CANN Toolkit 安装（假设已通过其他方式安装）"
    echo "[步骤3] 继续设置环境变量..."
    CANN_INSTALLED=true
else
    echo "[智能检测] CANN Toolkit 已安装，跳过安装步骤"
fi

# 步骤4: 设置 CANN 环境变量
echo "\n[步骤4] 设置 CANN 环境变量..."
CANN_ENV_SCRIPT=""
if [ -n "$CANN_ACTUAL_PATH" ]; then
    if [ -f "$CANN_ACTUAL_PATH/bin/setenv.bash" ]; then
        CANN_ENV_SCRIPT="$CANN_ACTUAL_PATH/bin/setenv.bash"
    elif [ -f "$CANN_ACTUAL_PATH/x86_64-linux/bin/setenv.bash" ]; then
        CANN_ENV_SCRIPT="$CANN_ACTUAL_PATH/x86_64-linux/bin/setenv.bash"
    fi
else
    # 尝试默认路径
    if [ -f "$HOME/Ascend/ascend-toolkit/latest/cann-8.5.0-beta.1/bin/setenv.bash" ]; then
        CANN_ENV_SCRIPT="$HOME/Ascend/ascend-toolkit/latest/cann-8.5.0-beta.1/bin/setenv.bash"
    elif [ -f "$HOME/Ascend/ascend-toolkit/latest/cann-8.5.0-beta.1/x86_64-linux/bin/setenv.bash" ]; then
        CANN_ENV_SCRIPT="$HOME/Ascend/ascend-toolkit/latest/cann-8.5.0-beta.1/x86_64-linux/bin/setenv.bash"
    fi
fi

if [ -n "$CANN_ENV_SCRIPT" ]; then
    echo "[步骤4] 加载 CANN 环境变量: $CANN_ENV_SCRIPT"
    source "$CANN_ENV_SCRIPT"
else
    echo "警告: 未找到 CANN 环境变量脚本，将在编译时自动设置"
fi

# 步骤5: 检查并构建第三方依赖
echo "\n[步骤5] 检查第三方依赖..."

# 检查第三方依赖是否已构建
THIRD_PARTY_BUILT=false
if [ -d "output/third_party" ] && [ "$(ls -A output/third_party 2>/dev/null)" ]; then
    echo "[智能检测] 第三方依赖已构建，跳过构建步骤"
    THIRD_PARTY_BUILT=true
elif [ -d "third_party" ] && [ "$(ls -A third_party 2>/dev/null)" ]; then
    echo "[智能检测] 第三方依赖目录已存在，跳过构建步骤"
    THIRD_PARTY_BUILT=true
fi

if [ "$THIRD_PARTY_BUILT" = false ]; then
    if [ -f "build_third_party.sh" ]; then
        echo "[步骤5] 构建第三方依赖..."
        # 创建必要的目录
        mkdir -p output/third_party 2>/dev/null
        if [ "$REPO_TYPE" = "hccl" ]; then
            bash build_third_party.sh --output_path=./output/third_party
        else
            bash build_third_party.sh
        fi
    else
        echo "警告: build_third_party.sh 文件不存在，跳过第三方依赖构建"
    fi
else
    echo "[智能检测] 第三方依赖已构建，跳过构建步骤"
fi

# 步骤6: 编译代码
echo "\n[步骤6] 编译 $REPO_TYPE 代码..."
# 为 HCCL 构建命令添加正确的 CANN 路径
if [ "$REPO_TYPE" = "hccl" ] && [ -n "$CANN_ACTUAL_PATH" ]; then
    # 检查 build.sh 是否支持自定义 CANN 路径
    if grep -q "--ascend-toolkit-path" build.sh 2>/dev/null; then
        BUILD_CMD="bash build.sh --pkg --ascend-toolkit-path=$CANN_ACTUAL_PATH"
    else
        # 如果不支持，使用环境变量
        BUILD_CMD="ASCEND_CANN_PACKAGE_PATH=$CANN_ACTUAL_PATH bash build.sh --pkg"
    fi
fi
echo "执行命令: $BUILD_CMD"
$BUILD_CMD

# 步骤7: 验证编译结果
echo "\n[步骤7] 验证编译结果..."
if [ -d "$OUTPUT_DIR" ]; then
    echo "编译成功! 输出目录: $OUTPUT_DIR/"
    echo "\n输出文件列表:"
    ls -la "$OUTPUT_DIR/"
    
    # 检查关键文件
    if [ "$REPO_TYPE" = "hcomm" ]; then
        # 检查 hcomm 关键库文件
        KEY_LIBS=("libhccd.so" "libccl_kernel.so")
        for lib in "${KEY_LIBS[@]}"; do
            if find "$OUTPUT_DIR" -name "$lib" -type f | grep -q "$lib"; then
                echo "\n✓ 找到 $lib"
            else
                echo "\n✗ 未找到 $lib"
            fi
        done
    elif [ "$REPO_TYPE" = "hccl" ]; then
        # 检查 hccl 关键包文件
        PKG_FILES=$(find "$OUTPUT_DIR" -name "cann-hccl_*.run" -type f)
        if [ -n "$PKG_FILES" ]; then
            echo "\n✓ 找到编译生成的 HCCL 安装包:"
            echo "$PKG_FILES"
        else
            echo "\n✗ 未找到编译生成的 HCCL 安装包"
        fi
    fi
else
    echo "错误: 编译失败，未生成 $OUTPUT_DIR 目录"
    exit 1
fi

echo "\n========================================"
echo "编译完成!"
echo "========================================"
echo "\n构建信息:"
echo "- 仓库类型: $REPO_TYPE"
if [ "$REPO_TYPE" = "hcomm" ] && [ -n "$REPO_URL" ]; then
    echo "- 仓库地址: $REPO_URL"
fi
echo "- 工作目录: $(pwd)"
echo "- 分支: $(git branch --show-current 2>/dev/null || echo "未知")"
echo "- CANN 版本: $CANN_VERSION_USE"
echo "- 输出目录: $(pwd)/$OUTPUT_DIR"
echo "\n使用方法:"
if [ "$REPO_TYPE" = "hccl" ]; then
    PKG_FILE=$(find "$OUTPUT_DIR" -name "cann-hccl_*.run" -type f | head -n 1)
    if [ -n "$PKG_FILE" ]; then
        echo "1. 安装 HCCL:"
        echo "   bash $PKG_FILE --full"
    fi
fi
echo "2. 运行测试:"
echo "   bash build.sh --ut"
echo "\n下次编译建议:"
echo "- 如果代码已更新，直接运行: bash build.sh --pkg"
echo "- 如果需要重新编译整个项目: rm -rf $OUTPUT_DIR build && bash build.sh --pkg"
echo "- 如果在其他目录编译: ../oneclick_build.sh $REPO_TYPE"
echo "========================================"
