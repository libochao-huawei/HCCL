#!/bin/bash

set -eux

HCCL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash build.sh \
     --vendor=cust \
     --ops=allgather \
     --custom_ops_path=./examples/07_custom_ops_allgather_ccu \
     2>&1 | tee compile.log
