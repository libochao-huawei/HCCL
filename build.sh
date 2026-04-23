#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
BUILD_DEVICE_DIR="${CURRENT_DIR}/build_device"
OUTPUT_DIR=${CURRENT_DIR}/build_out
OUTPUT_PATH=${CURRENT_DIR}/output
USER_ID=$(id -u)
CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)*2))
JOB_NUM="-j${CPU_NUM}"
ASAN="false"
COV="false"
CUSTOM_OPTION="-DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR}"
FULL_MODE="false"  # 新增变量，用于控制是否全量构建
STATIC_MODE="false"  # 新增变量，用于控制是否静态编译
KERNEL="false"  # 新增变量，用于控制是否只编译 ccl_kernel.so
CANN_3RD_LIB_PATH="${CURRENT_DIR}/third_party"
CUSTOM_SIGN_SCRIPT="${CURRENT_DIR}/scripts/sign/community_sign_build.py"
ENABLE_SIGN="false"
VERSION_INFO="8.5.0"
BUILD_AARCH="false"

ENABLE_UT="off"
ENABLE_ST="off"
CMAKE_BUILD_TYPE="Debug"
BUILD_CB_TEST="false"

# 自定义算子工程
ENABLE_CUSTOM="off"
CUSTOM_OPS_NAME=""
CUSTOM_OPS_PATH=""
CUSTOM_OPS_VENDOR=""

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function set_env()
{
    source $ASCEND_CANN_PACKAGE_PATH/set_env.sh || echo "0"
}

# 初始化交叉编译工具链变量
# 调用此函数后，可使用 ${AR}, ${CC}, ${CXX}, ${LD}, ${NM}, ${STRIP} 等变量
function init_toolchain()
{
    if [ "${BUILD_AARCH}" == "true" ]; then
        TOOLCHAIN_DIR="${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/"
        log "Info: Using cross-compile toolchain: ${TOOLCHAIN_DIR}"
        export AR="${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-ar"
        export CC="${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc"
        export CXX="${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++"
        export LD="${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-ld"
        export NM="${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-nm"
        export STRIP="${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-strip"
    else
        log "Info: Using native toolchain"
        export AR="ar"
        export CC="gcc"
        export CXX="g++"
        export LD="ld"
        export NM="nm"
        export STRIP="strip"
    fi
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi

    if [ -z "${TEST}" ] && [ -z "${KERNEL}" ];then
        if [ -n "${OUTPUT_DIR}" ];then
            rm -rf ${OUTPUT_DIR}
        fi
    fi

    mkdir -p ${BUILD_DIR}
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake ..  ${CUSTOM_OPTION} ${extra_option}
}

function build()
{
    log "Info: build target:$@ JOB_NUM:${JOB_NUM}"
    cmake --build . --target "$@" ${JOB_NUM} #--verbose
}

function build_package(){
    cmake_config
    log "Info: build_package"
    build package
}

function build_cb_test_verify(){
    cd ${CURRENT_DIR}/examples/
    bash build.sh
}

function build_test() {
    cmake_config

    LIBRARY_DIR="${BUILD_DIR}/test:${ASCEND_HOME_PATH}/lib64:"
    # 每日构建sdk包安装路径
    if [ -d "${ASCEND_HOME_PATH}/opensdk" ];then
        LIBRARY_DIR="${LIBRARY_DIR}${ASCEND_HOME_PATH}/opensdk/opensdk/gtest_shared/lib64:"
    fi

    # 社区sdk包安装路径
    if [ -d "${ASCEND_HOME_PATH}/../../latest/opensdk" ];then
        LIBRARY_DIR="${LIBRARY_DIR}${ASCEND_HOME_PATH}/../../latest/opensdk/opensdk/gtest_shared/lib64:"
    fi

    GCC_MAJOR=$(gcc -dumpversion | cut -d. -f1)
    if [ "${ASAN}" == "true" ];then
        ARCH=$(uname -m)
        if [[ $ARCH == "x86_64" || $ARCH == "i386" || $ARCH == "i686" ]]; then
            PRELOAD="/usr/lib/gcc/x86_64-linux-gnu/${GCC_MAJOR}/libasan.so:/usr/lib/gcc/x86_64-linux-gnu/${GCC_MAJOR}/libstdc++.so"
        elif [[ $ARCH == "aarch64" || $ARCH == "armv8l" || $ARCH == "armv7l" ]]; then
            PRELOAD="/usr/lib/gcc/aarch64-linux-gnu/${GCC_MAJOR}/libasan.so:/usr/lib/gcc/aarch64-linux-gnu/${GCC_MAJOR}/libstdc++.so"
        else
            echo "未知架构: $ARCH"
        fi
        echo "PRELOAD is ${PRELOAD}"
        ASAN_OPT="detect_leaks=0"
    fi

    if [ "${TEST_TASK_NAME}" == "open_hccl_test" ] || [ "$TEST" = "all" ];then
        build open_hccl_test
        export LD_LIBRARY_PATH=${LIBRARY_DIR}${LD_LIBRARY_PATH} && export LD_PRELOAD=${PRELOAD} && export ASAN_OPTIONS=${ASAN_OPT} \
        && ${BUILD_DIR}/test/open_hccl_test
    fi

    if [ "${TEST_TASK_NAME}" == "executor_hccl_test" ] || [ "$TEST" = "all" ];then
        build executor_hccl_test
        export LD_LIBRARY_PATH=${LIBRARY_DIR}${LD_LIBRARY_PATH} && export LD_PRELOAD=${PRELOAD} && export ASAN_OPTIONS=${ASAN_OPT} \
        && ${BUILD_DIR}/test/executor_hccl_test
    fi

    if [ "${TEST_TASK_NAME}" == "executor_reduce_hccl_test" ] || [ "$TEST" = "all" ];then
        build executor_reduce_hccl_test
        export LD_LIBRARY_PATH=${LIBRARY_DIR}${LD_LIBRARY_PATH} && export LD_PRELOAD=${PRELOAD} && export ASAN_OPTIONS=${ASAN_OPT} \
        && ${BUILD_DIR}/test/executor_reduce_hccl_test
    fi

    if [ "${TEST_TASK_NAME}" == "executor_pipeline_hccl_test" ] || [ "$TEST" = "all" ];then
        build executor_pipeline_hccl_test
        export LD_LIBRARY_PATH=${LIBRARY_DIR}${LD_LIBRARY_PATH} && export LD_PRELOAD=${PRELOAD} && export ASAN_OPTIONS=${ASAN_OPT} \
        && ${BUILD_DIR}/test/executor_pipeline_hccl_test
    fi

    # 除算法的测试用例都依赖编译出来的so文件，所以需要额外加入环境变量
    LIBRARY_DIR="${BUILD_DIR}/src:${BUILD_DIR}/src/algorithm:${BUILD_DIR}/src/framework:${BUILD_DIR}/src/platform: \
    ${BUILD_DIR}/test:${ASCEND_HOME_PATH}/lib64:"
    # 每日构建sdk包安装路径
    if [ -d "${ASCEND_HOME_PATH}/opensdk" ];then
        LIBRARY_DIR="${LIBRARY_DIR}${ASCEND_HOME_PATH}/opensdk/opensdk/gtest_shared/lib64:"
    fi

    # 社区sdk包安装路径
    if [ -d "${ASCEND_HOME_PATH}/../../latest/opensdk" ];then
        LIBRARY_DIR="${LIBRARY_DIR}${ASCEND_HOME_PATH}/../../latest/opensdk/opensdk/gtest_shared/lib64:"
    fi
}

function build_device(){
    cmake_config
    log "Info: build_device"
    TARGET_LIST="scatter_aicpu_kernel"
    echo "TARGET_LIST=${TARGET_LIST}"
    PKG_TARGET_LIST="generate_device_aicpu_package"
    echo "PKG_TARGET_LIST=${PKG_TARGET_LIST}"
    SIGN_TARGET_LIST="sign_aicpu_hccl"
    echo "SIGN_TARGET_LIST=${SIGN_TARGET_LIST}"
    build ${TARGET_LIST} ${PKG_TARGET_LIST} ${SIGN_TARGET_LIST}
}

function build_kernel() {
    cmake_config
    log "Info: build_kernel"
    build scatter_aicpu_kernel
}

function build_static() {
    log "Info: Starting static library build"

    # 初始化交叉编译工具链
    init_toolchain

    # 步骤1: 构建设备端AICPU包
    log "Info: Building device-side AICPU package"
    mkdir -p ${BUILD_DEVICE_DIR}
    cd ${BUILD_DEVICE_DIR}
    CURRENT_CUSTOM_OPTION="${CUSTOM_OPTION}"
    CUSTOM_OPTION="${CURRENT_CUSTOM_OPTION} -DFULL_MODE=ON -DDEVICE_MODE=ON -DKERNEL_MODE=ON -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT} -DENABLE_SIGN=${ENABLE_SIGN} -DVERSION_INFO=${VERSION_INFO}"
    build_device

    # 检查AICPU tar包是否生成
    local AICPU_TAR="${BUILD_DEVICE_DIR}/aicpu_hccl.tar.gz"
    if [ ! -f "${AICPU_TAR}" ]; then
        log "Error: AICPU tar package not found: ${AICPU_TAR}"
        exit 1
    fi
    log "Info: AICPU tar package generated: ${AICPU_TAR}"

    # 步骤2: 构建主机端静态库
    log "Info: Building host-side static library"
    cd "${CURRENT_DIR}" && cd "${BUILD_DIR}"
    CUSTOM_OPTION="${CURRENT_CUSTOM_OPTION} -DDEVICE_MODE=OFF -DSTATIC_MODE=ON"
    cmake_config

    # 构建hccl静态库目标
    build hccl

    # 检查静态库是否生成
    local STATIC_LIB="${BUILD_DIR}/src/libhccl_static.a"
    if [ ! -f "${STATIC_LIB}" ]; then
        log "Error: Static library not found at expected location: ${STATIC_LIB}"
        exit 1
    fi
    log "Info: Static library generated: ${STATIC_LIB}"

    # 步骤3: 解压静态库为.o文件
    log "Info: Extracting object files from static library"
    local EXTRACT_DIR="${BUILD_DIR}/static_extract"
    mkdir -p ${EXTRACT_DIR}
    cd ${EXTRACT_DIR}

    # 解压静态库中的所有.o文件
    if ! ${AR} -x "${STATIC_LIB}"; then
        log "Error: Failed to extract object files from static library"
        exit 1
    fi

    local OBJ_COUNT=0
    if ls -1 *.o 1>/dev/null 2>&1; then
        OBJ_COUNT=$(ls -1 *.o | wc -l)
    fi
    log "Info: Extracted ${OBJ_COUNT} object files"

    # 步骤4: 将AICPU tar包转换为二进制对象文件
    # 必须用纯文件名调用 ld，否则路径会嵌入符号名
    # (如 _binary__root_xxx_aicpu_hccl_tar_gz_start 而非 _binary_aicpu_hccl_tar_gz_start)
    log "Info: Converting AICPU tar package to binary object"
    local AICPU_OBJ="${EXTRACT_DIR}/aicpu_hccl_tar.o"
    cp "${AICPU_TAR}" "${EXTRACT_DIR}/aicpu_hccl.tar.gz"
    if ! (cd "${EXTRACT_DIR}" && ${LD} -r -b binary -o aicpu_hccl_tar.o aicpu_hccl.tar.gz); then
        log "Error: Failed to convert AICPU tar to binary object"
        exit 1
    fi

    # 步骤6: 将所有.o文件打包成最终的静态库
    log "Info: Creating final static library libhccl_static.a"
    cd ${EXTRACT_DIR}
    local FINAL_STATIC_LIB="${OUTPUT_DIR}/lib/libhccl_static_final.a"
    mkdir -p $(dirname ${FINAL_STATIC_LIB})

    # 创建最终的静态库
    if ! ${AR} rcs ${FINAL_STATIC_LIB} *.o; then
        log "Error: Failed to create final static library"
        exit 1
    fi

    # 验证最终静态库
    local FINAL_OBJ_COUNT=$(${AR} t ${FINAL_STATIC_LIB} | wc -l)
    log "Info: Final static library created: ${FINAL_STATIC_LIB}"
    log "Info: Contains ${FINAL_OBJ_COUNT} object files"

    # 步骤7: 复制到标准输出位置
    local OUTPUT_STATIC_LIB="${OUTPUT_DIR}/lib/libhccl_static.a"
    mkdir -p $(dirname ${OUTPUT_STATIC_LIB})
    if ! cp ${FINAL_STATIC_LIB} ${OUTPUT_STATIC_LIB}; then
        log "Error: Failed to copy final static library to output location"
        exit 1
    fi
    log "Info: Static library copied to: ${OUTPUT_STATIC_LIB}"

    # 步骤8: 清理临时目录
    log "Info: Cleaning up temporary files"
    rm -rf ${EXTRACT_DIR}

    # 清理build_device目录
    [ -n "${BUILD_DEVICE_DIR}" ] && rm -rf ${BUILD_DEVICE_DIR}

    log "Info: Static library build completed successfully"
    log "Info: Output: ${OUTPUT_STATIC_LIB}"
}

function package_static_tar() {
    log "Info: Starting static package creation"
    cd ${CURRENT_DIR}
    # 检查静态库是否存在
    local static_lib="${OUTPUT_DIR}/lib/libhccl_static.a"
    local include_dir="${CURRENT_DIR}/include"

    if [ ! -f "${static_lib}" ]; then
        log "Error: Static library not found: ${static_lib}"
        exit 1
    fi

    # 确定架构
    local ARCH=$(uname -m)
    local TAR_ARCH=""
    case "$ARCH" in
        x86_64|i386|i686)
            TAR_ARCH="x86_64"
            ;;
        aarch64|armv8l|armv7l)
            TAR_ARCH="arch64"
            ;;
        *)
            log "Error: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    # 创建临时打包目录
    local temp_dir="${OUTPUT_DIR}/static_package_temp"
    mkdir -p "${temp_dir}/include"
    mkdir -p "${temp_dir}/lib64"

    # 复制文件
    cp -r "${include_dir}/." "${temp_dir}/include/"
    cp "${static_lib}" "${temp_dir}/lib64/"

    # 创建tar包
    local tar_name="cann-hccl-static_${VERSION_INFO}_linux-${TAR_ARCH}.tar.gz"
    local tar_path="${OUTPUT_DIR}/${tar_name}"

    cd "${temp_dir}" && tar -czf "${tar_path}" include lib64

    # 验证和清理
    if [ -f "${tar_path}" ]; then
        local tar_size=$(stat -c%s "${tar_path}" 2>/dev/null || stat -f%z "${tar_path}")
        log "Info: Static package created: ${tar_name} (${tar_size} bytes)"
        rm -rf "${temp_dir}"
    else
        log "Error: Failed to create tar package"
        exit 1
    fi

    log "Info: Static package location: ${tar_path}"
}

function mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

# create build path
function build_ut() {
  echo "create build directory and build";
  mk_dir "${OUTPUT_PATH}"
  mk_dir "${BUILD_DIR}"
  local report_dir="${OUTPUT_PATH}/report/ut" && mk_dir "${report_dir}"
  cd "${BUILD_DIR}"

  local llt_kill_time=1200
  CMAKE_ARGS="-DPRODUCT_SIDE=host \
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
              -DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR} \
              -DASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} \
              -DENABLE_COV=${ENABLE_COV} \
              -DENABLE_TEST=${ENABLE_TEST} \
              -DENABLE_UT=${ENABLE_UT} \
              -DENABLE_ST=${ENABLE_ST} \
              -DOUTPUT_PATH=${OUTPUT_PATH} \
 	          -DLLT_KILL_TIME=${llt_kill_time}"

  echo "CMAKE_ARGS=${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ..
  if [ $? -ne 0 ]; then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi

  # make all
  cmake --build . -j${CPU_NUM}
  echo "build success!"
}

function run_ut() {
  if [[ "X$ENABLE_UT" = "Xon" ]]; then
    local ut_dir="${BUILD_DIR}/test"
    echo "ut_dir = ${ut_dir}"
    find "$ut_dir" -type f -executable | while read -r ut_exec; do
        filename=$(basename "$ut_exec")
        echo "Executing: $filename"
        ${ut_exec}
    done
  else
    echo "Unit tests is not enabled, sh build.sh with parameter -u or --ut to enable it"
  fi
}

function run_st() {
  if [[ "X$ENABLE_ST" = "Xon" ]]; then
    local st_build_shell="${CURRENT_DIR}/test/st/algorithm/build.sh"
    echo "st_build_shell = ${st_build_shell}"
    if [ -e ${st_build_shell} ]; then
      echo "开始执行st..."
      bash ${st_build_shell}
    else
      echo "${st_build_shell} 文件不存在!"
    fi
  else
    echo "System tests is not enabled, sh build.sh with parameter -s or --st to enable it"
  fi
}

function build_custom() {
    # 编译 Device 包
    log "Info: build_custom_device"
    mk_dir ${BUILD_DEVICE_DIR}
    cd ${BUILD_DEVICE_DIR}
    cmake_config "-DKERNEL_MODE=ON \
                  -DENABLE_CUSTOM=ON \
                  -DCUSTOM_OPS_PATH=${CUSTOM_OPS_PATH} \
                  -DCUSTOM_OPS_NAME=${CUSTOM_OPS_NAME} \
                  -DCUSTOM_OPS_VENDOR=${CUSTOM_OPS_VENDOR} \
                  -DENABLE_SIGN=${ENABLE_SIGN} \
                  -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT} \
                  -DVERSION_INFO=${VERSION_INFO}"
    # 编译 AICPU Kernel 包
    build custom_aicpu

    # 编译 Host 包
    log "Info: build_custom_host"
    cd ${BUILD_DIR}
    cmake_config "-DENABLE_CUSTOM=ON
                  -DCUSTOM_OPS_PATH=${CUSTOM_OPS_PATH} \
                  -DCUSTOM_OPS_NAME=${CUSTOM_OPS_NAME} \
                  -DCUSTOM_OPS_VENDOR=${CUSTOM_OPS_VENDOR} \
                  -DVERSION_INFO=${VERSION_INFO}"
    # 打包 run 包
    build package
}

# print usage message
function usage() {
  echo "Usage:"
  echo "bash build.sh --pkg [-h | --help] [-j<N>]"
  echo "              [--cann_3rd_lib_path=<PATH>] [-p|--package-path <PATH>]"
  echo "              [--asan]"
  echo "              [--sign-script <PATH>] [--enable-sign] [--version <VERSION>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --asan         Enable AddressSanitizer"
  echo "    -build-type=<TYPE>"
  echo "                   Specify build type (TYPE options: Release/Debug), Default: Release"
  echo "    -j<N>          Set the number of threads used for building, default is 8"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                   Set ascend third_party package install path, default ./output/third_party"
  echo "    -p|--package-path <PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --sign-script <PATH>"
  echo "                   Set sign-script's path to <PATH>"
  echo "    --enable-sign"
  echo "                   Enable to sign"
  echo "    --version <VERSION>"
  echo "                   Set sign version to <VERSION>"
  echo "    --cb_test_verify"
  echo "                   Run smoke tests"
  echo "    --custom_ops_path=<CUSTOM_OPS_PATH>"
  echo "                   Set custom ops project path to <CUSTOM_OPS_PATH>"
  echo "    --ops=<OPS>"
  echo "                   Set custom ops name to <OPS>"
  echo "    --vendor=<VENDOR>"
  echo "                   Set custom ops vendor to <VENDOR>"
  echo "    --static"
  echo "                   Enable static library build mode"
  echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
    -j*)
        JOB_NUM="$1"
        shift
        ;;
    --build-type=*)
        OPTARG=$1
        BUILD_TYPE="${OPTARG#*=}"
        shift
        ;;
    --ccache)
        CCACHE_PROGRAM="$2"
        shift 2
        ;;
    -p|--package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    --nlohmann_path)
        third_party_nlohmann_path="$2"
        shift 2
        ;;
    --pkg)
        # 跳过 --pkg，不做处理
        shift
        ;;
    --cann_3rd_lib_path=*)
        OPTARG=$1
        CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
        shift
        ;;
    -u|--ut)
        ENABLE_TEST="on"
        ENABLE_UT="on"
        shift
        ;;
    -s|--st)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        shift
        ;;
    -t|--test)
        TEST="all"
        shift
        ;;
    --open_hccl_test)
        TEST="partial"
        TEST_TASK_NAME="open_hccl_test"
        shift
        ;;
    --executor_hccl_test)
        TEST="partial"
        TEST_TASK_NAME="executor_hccl_test"
        shift
        ;;
    --executor_reduce_hccl_test)
        TEST="partial"
        TEST_TASK_NAME="executor_reduce_hccl_test"
        shift
        ;;
    --executor_pipeline_hccl_test)
        TEST="partial"
        TEST_TASK_NAME="executor_pipeline_hccl_test"
        shift
        ;;
    --aicpu)  # 新增选项，用于只编译 scatter_aicpu_kernel
        KERNEL="true"
        shift
        ;;
    --full)
        FULL_MODE="true"
        shift
        ;;
    --static)
        STATIC_MODE="true"
        shift
        ;;
    --build_aarch)
        BUILD_AARCH="true"
        shift
        ;;
    --asan)
        ASAN="true"
        shift
        ;;
    --cov)
        COV="true"
        shift
        ;;
    --sign-script=*)
        shift
        ;;
    --cb_test_verify)
        BUILD_CB_TEST="true"
        shift
        ;;
    --enable-sign)
        ENABLE_SIGN="true"
        shift
        ;;
    --sign-script)
        CUSTOM_SIGN_SCRIPT="$(realpath $2)"
        shift 2
        ;;
    --version)
        VERSION_INFO="$2"
        shift 2
        ;;
    --custom_ops_path=*)
        OPTARG=$1
        CUSTOM_OPS_PATH="$(realpath ${OPTARG#*=})"
        ENABLE_CUSTOM="on"
        shift
        ;;
    --ops=*)
        OPTARG=$1
        CUSTOM_OPS_NAME="${OPTARG#*=}"
        ENABLE_CUSTOM="on"
        shift
        ;;
    --vendor=*)
        OPTARG=$1
        CUSTOM_OPS_VENDOR="${OPTARG#*=}"
        ENABLE_CUSTOM="on"
        shift
        ;;
    *)
        log "Error: Undefined option: $1"
        usage
        exit 1
        ;;
    esac
done

if [ -n "${TEST}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=ON"
fi

if [ "${KERNEL}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DKERNEL_MODE=ON -DDEVICE_MODE=ON"
fi

if [ "${FULL_MODE}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DFULL_MODE=ON"
fi

if [ "${STATIC_MODE}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DSTATIC_MODE=ON"
fi

if [ "${BUILD_AARCH}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DAARCH_MODE=ON"
fi

if [ "${ASAN}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ASAN=true"
fi

if [ "${COV}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=true"
fi

if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    log "Error: Please set the toolkit package installation directory through parameter -p|--package-path."
    exit 1
fi

if [ -n "${third_party_nlohmann_path}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTHIRD_PARTY_NLOHMANN_PATH=${third_party_nlohmann_path}"
fi

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"
CUSTOM_OPTION="$CUSTOM_OPTION -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
CUSTOM_OPTION="$CUSTOM_OPTION -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

set_env

clean

cd ${BUILD_DIR}

if [ "${ENABLE_UT}" == "on" ]; then
    build_ut
    run_ut
elif [ "${ENABLE_ST}" == "on" ]; then
    run_st
elif [ -n "${TEST}" ];then
    build_test
elif [ "${KERNEL}" == "true" ]; then
    build_kernel
elif [ "${ENABLE_CUSTOM}" == "on" ]; then
    build_custom
elif [ "${BUILD_CB_TEST}" == "true" ]; then
    log "Info: Building cb_test_verify"
    build_cb_test_verify
    if grep -q "Make Failure" ${BUILD_DIR}/build.log || grep -q "Make test Failure" ${BUILD_DIR}/build.log; then
        log "Info: Building cb_test_verify failed"
        exit 1
    else
        log "Info: Building cb_test_verify success"
    fi
elif [ "${FULL_MODE}" == "true" ]; then
    cd ..
    mkdir -p ${BUILD_DEVICE_DIR}
    cd ${BUILD_DEVICE_DIR}
    CURRENT_CUSTOM_OPTION="${CUSTOM_OPTION}"
    CUSTOM_OPTION="${CURRENT_CUSTOM_OPTION} -DFULL_MODE=ON -DDEVICE_MODE=ON -DKERNEL_MODE=ON -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT} -DENABLE_SIGN=${ENABLE_SIGN} -DVERSION_INFO=${VERSION_INFO}"
    build_device
    cd .. & cd ${BUILD_DIR}
    CUSTOM_OPTION="${CURRENT_CUSTOM_OPTION} -DDEVICE_MODE=OFF"
    build_package
    [ -n "${BUILD_DEVICE_DIR}" ] && rm -rf ${BUILD_DEVICE_DIR}
elif [ "${STATIC_MODE}" == "true" ]; then
    build_static
    package_static_tar
else
    CUSTOM_OPTION="${CUSTOM_OPTION} -DDEVICE_MODE=OFF"
    build_package
fi
