#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-default}"
shift || true

TOKEN_BYTES=327680
SCALE_COUNT=128
DEVICES=8
PRINT_COUNT=8
WARMUP=1
ITERS=1
EXTRA_ARGS=()

case "${SCENARIO}" in
  default)
    ;;
  fast)
    TOKEN_BYTES=65536
    SCALE_COUNT=0
    DEVICES=4
    WARMUP=1
    ITERS=5
    ;;
  single-item)
    TOKEN_BYTES=327680
    SCALE_COUNT=0
    DEVICES=8
    WARMUP=1
    ITERS=5
    ;;
  help|-h|--help)
    cat <<EOF
Usage: ./run.sh [default|fast|single-item] [extra testcase args...]

Examples:
  ./run.sh
  ./run.sh fast
  ./run.sh single-item --no-verify
  ./run.sh default --token-bytes 65536 --iters 10
EOF
    exit 0
    ;;
  *)
    echo "unknown scenario: ${SCENARIO}" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/allgatherbatch_testcase"

if [[ ! -x "${TARGET}" ]]; then
  echo "${TARGET} not found, build it first with CMake or make" >&2
  exit 1
fi

exec "${TARGET}" \
  --token-bytes "${TOKEN_BYTES}" \
  --scale-count "${SCALE_COUNT}" \
  --devices "${DEVICES}" \
  --print-count "${PRINT_COUNT}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  "$@"
