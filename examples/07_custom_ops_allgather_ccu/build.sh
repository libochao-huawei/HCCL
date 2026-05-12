#!/bin/bash

set -eux

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HCCL_DIR="${SCRIPT_PATH}/../../"

cd ${HCCL_DIR}
bash build.sh --vendor=cust --ops=allgather --custom_ops_path=./examples/07_custom_ops_allgather_ccu
