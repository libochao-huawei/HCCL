#!/bin/bash

# --- 路径定义 ---
PROJECT_DIR=$(cd $(dirname $0); pwd)
BUILD_DIR="${PROJECT_DIR}/build"

log() { echo -e "\033[32m[INFO]\033[0m $1"; }
err() { echo -e "\033[31m[ERROR]\033[0m $1"; exit 1; }

# --- 默认参数 ---
RUN_INSTALL=false
RUN_PACKAGE=false
RUN_RESET=false

# --- 参数解析 ---
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --install  执行安装 (make install)"
    echo "  -p, --package  执行打包 (cpack)"
    echo "  -r, --reset    完全清空 build 目录重新编译"
    echo "  -h, --help     显示此帮助"
    exit 0
}

# 处理长参数兼容
for arg in "$@"; do
    case $arg in
        -i|--install) RUN_INSTALL=true ;;
        -p|--package) RUN_PACKAGE=true ;;
        -r|--reset)   RUN_RESET=true ;;
        -h|--help)    usage ;;
    esac
done

# 1. 环境检查
log "Checking environment..."
[ -z "${ASCEND_HOME_PATH}" ] && err "ASCEND_HOME_PATH is not set! Run 'source set_env.sh' to set CANN env first."

# 2. 重置逻辑
if [ "$RUN_RESET" = true ]; then
    log "Resetting build environment..."
    rm -rf "$BUILD_DIR"
    # 注意：install 目录现在由 CMake 的 CMAKE_INSTALL_PREFIX 控制
fi

# 3. 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 4. 配置与编译
log "Configuring project..."
# 提示：在这里可以确保 CMAKE_INSTALL_PREFIX 始终指向我们想要的 install 文件夹
cmake .. -DCMAKE_INSTALL_PREFIX="${PROJECT_DIR}/hccl_vm_install"
[ $? -ne 0 ] && err "CMake configuration failed!"

log "Building project..."
cmake --build . -j$(nproc) || err "Build failed!"

# 5. 可选步骤：安装
if [ "$RUN_INSTALL" = true ]; then
    log "Installing to ${PROJECT_DIR}/hccl_vm_install ..."
    cmake --install . || err "Installation failed!"
fi

# 6. 可选步骤：打包 (使用 CPack)
if [ "$RUN_PACKAGE" = true ]; then
    log "Running CPack to create delivery package..."
    cpack || err "Packaging failed!"
    
    # 移动生成的包到根目录（可选，方便查看）
    # mv *.tar.gz "${PROJECT_DIR}/" 2>/dev/null
fi

log "------------------------------------------"
log "Task Completed Successfully!"
log "------------------------------------------"