# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/bin/bash
set -e
trap 'echo "❌ Error occurred in build.sh at line $LINENO"; exit 1' ERR

PERF_ONLY=0
GTEST_FILTER="${HCCL_GTEST_FILTER:-}"
CPU_LIST="${HCCL_CPU_LIST:-}"
LOG_FILE="${HCCL_LOG_FILE:-}"
PIN_RANK_THREADS="${HCCL_ST_PERF_PIN_THREADS:-0}"
RANK_DETAIL="${HCCL_ST_PERF_RANK_DETAIL:-0}"
ST_LOG_LEVEL="${HCCL_ST_LOG_LEVEL:-ERROR}"
PERF_SIZE_KIB="${HCCL_ST_PERF_SIZE_KIB:-}"
PERF_REPEATS="${HCCL_ST_PERF_REPEATS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf-rs)
            PERF_ONLY=1
            shift
            ;;
        --gtest_filter)
            GTEST_FILTER="$2"
            shift 2
            ;;
                --cpu-list)
                        CPU_LIST="$2"
                        shift 2
                        ;;
                --log-file)
                        LOG_FILE="$2"
                        shift 2
                        ;;
                --log-level)
                    ST_LOG_LEVEL="$2"
                    shift 2
                    ;;
                --pin-rank-threads)
                    PIN_RANK_THREADS=1
                    shift
                    ;;
                --rank-detail)
                    RANK_DETAIL=1
                    shift
                    ;;
                --perf-size)
                    PERF_SIZE_KIB="$2"
                    shift 2
                    ;;
                --perf-repeats)
                    PERF_REPEATS="$2"
                    shift 2
                    ;;
        --help|-h)
            cat <<EOF
        Usage: $(basename "$0") [--perf-rs] [--gtest_filter <expr>] [--cpu-list <list>] [--log-file <path>] [--log-level <level>] [--pin-rank-threads] [--rank-detail] [--perf-size <KiB>] [--perf-repeats <N>]

Options:
  --perf-rs                 Only run reduce_scatter CCU perf testcases.
  --gtest_filter <expr>     Pass a custom gtest filter expression.
    --cpu-list <list>         Bind test process to CPUs via taskset, e.g. 0-7 or 0,2,4,6.
    --log-file <path>         Save full test output to file while printing to console.
        --log-level <level>       ST log level: DEBUG, INFO, WARNING, or ERROR.
    --pin-rank-threads        Pin each rank thread to one CPU within the allowed CPU set.
    --rank-detail             Print per-rank total/average timing lines.
    --perf-size <KiB>         Only measure this single message size (in KiB, e.g. 32).
    --perf-repeats <N>        Override repeat count (default 5).

Env:
  HCCL_GTEST_FILTER         Alternative way to set gtest filter.
    HCCL_CPU_LIST             Alternative way to set CPU affinity.
    HCCL_LOG_FILE             Alternative way to set output log file.
        HCCL_ST_LOG_LEVEL         Alternative way to set ST log level. Defaults to ERROR.
    HCCL_ST_PERF_PIN_THREADS  Alternative way to enable per-rank thread pinning.
    HCCL_ST_PERF_RANK_DETAIL  Alternative way to print per-rank timing details.
    HCCL_ST_PERF_SIZE_KIB     Alternative way to set single-size filter (in KiB).
    HCCL_ST_PERF_REPEATS      Alternative way to set repeat count.
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

if [[ ${PERF_ONLY} -eq 1 ]]; then
    GTEST_FILTER="ST_REDUCE_SCATTER_TEST.test_ccu_sched_reducescatter_perf_001:ST_REDUCE_SCATTER_TEST.test_ccu_ms_reducescatter_perf_001"
fi

# 获取shell脚本目录作为根目录
SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [[ -n "${LOG_FILE}" && "${LOG_FILE}" != /* ]]; then
    LOG_FILE="${SHELL_DIR}/${LOG_FILE}"
fi

# 创建build编译目录
cd $SHELL_DIR
mkdir -p ./build && cd ./build/ && rm -rf ../build/*

# 自动加载 CANN 环境（若存在）
ASCEND_ENV_SH="/usr/local/Ascend/cann/set_env.sh"
if [[ -f "${ASCEND_ENV_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_ENV_SH}"
else
    echo "[build.sh] warning: ${ASCEND_ENV_SH} not found, continue without sourcing."
fi

# 编译用例工程，配置执行条件并执行
cmake .. -DBUILD_OPEN_PROJECT=ON && make -j8
LIBRARY_DIR="${SHELL_DIR}/build/utils/src/hccl_depends_stub:"

RUN_CMD=("${SHELL_DIR}/build/testcase/hccl_checker_ops_stest")

if [[ -n "${GTEST_FILTER}" ]]; then
    echo "[build.sh] Running with gtest_filter=${GTEST_FILTER}"
    RUN_CMD+=("--gtest_filter=${GTEST_FILTER}" "--gtest_brief=1")
fi

if [[ -n "${CPU_LIST}" ]]; then
    if ! command -v taskset >/dev/null 2>&1; then
        echo "[build.sh] error: taskset not found but --cpu-list was provided."
        exit 1
    fi
    echo "[build.sh] Binding test process to CPUs: ${CPU_LIST}"
    RUN_CMD=(taskset -c "${CPU_LIST}" "${RUN_CMD[@]}")
fi

export LD_LIBRARY_PATH=${LIBRARY_DIR}${LD_LIBRARY_PATH}
export HCCL_ST_LOG_LEVEL=${ST_LOG_LEVEL}
export HCCL_ST_PERF_PIN_THREADS=${PIN_RANK_THREADS}
export HCCL_ST_PERF_RANK_DETAIL=${RANK_DETAIL}
if [[ -n "${PERF_SIZE_KIB}" ]]; then
    export HCCL_ST_PERF_SIZE_KIB=${PERF_SIZE_KIB}
    echo "[build.sh] Single-size mode: only measuring ${PERF_SIZE_KIB} KiB."
fi
if [[ -n "${PERF_REPEATS}" ]]; then
    export HCCL_ST_PERF_REPEATS=${PERF_REPEATS}
    echo "[build.sh] Repeat count overridden to ${PERF_REPEATS}."
fi

if [[ -n "${LOG_FILE}" ]]; then
    mkdir -p "$(dirname "${LOG_FILE}")"
    echo "[build.sh] Writing output log to ${LOG_FILE}"
    "${RUN_CMD[@]}" | tee "${LOG_FILE}"
else
    "${RUN_CMD[@]}"
fi

exit 0