#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ASCEND_HOME=${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}
NP=${NP:-2}
COUNT0=${COUNT0:-1024}
COUNT1=${COUNT1:-2048}
WARMUP=${WARMUP:-20}
ITERS=${ITERS:-100}

export LD_LIBRARY_PATH="$SCRIPT_DIR/../build/op_host:$ASCEND_HOME/lib64:$LD_LIBRARY_PATH"

cd "$SCRIPT_DIR"
make
mpirun -n "$NP" ./main --count0="$COUNT0" --count1="$COUNT1" --warmup="$WARMUP" --iters="$ITERS"
