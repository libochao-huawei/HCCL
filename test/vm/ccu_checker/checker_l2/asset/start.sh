#!/bin/bash

remove_files_by_prefix() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: remove_files_by_prefix <prefix>" >&2
    return 2
  fi

  local prefix="$1"
  if [ -z "$prefix" ]; then
    return 0
  fi

  shopt -s nullglob
  local any_deleted=0
  for f in "${prefix}"*; do
    if [ -f "$f" ]; then
      rm -f -- "$f" && any_deleted=1
    fi
  done
  shopt -u nullglob

  # 无论是否删除了文件，均返回 0，确保脚本继续执行
  return 0
}

set -e
SERVER_COUNT=1  # server个数
PROCESSES_PER_SERVER=2  # 一个server的rank数


echo "step 1: config hostfile..."
rm -f hostfile
for i in $(seq 1 $SERVER_COUNT); do
    echo "127.0.0.1:$PROCESSES_PER_SERVER" >> hostfile
done

echo "hostfile success:"
cat hostfile


echo "step 2:create wrapper bash..."
rm -f hccl_wrapper.sh
cat > hccl_wrapper.sh << EOF
#!/bin/bash

# MPI global rank env
if [ -n "\$OMPI_COMM_WORLD_RANK" ]; then
    RANK=\$OMPI_COMM_WORLD_RANK  # OpenMPI
elif [ -n "\$PMI_RANK" ]; then
    RANK=\$PMI_RANK              # MPICH
else
    echo "Error: Cannot detect RANK"
    exit 1
fi

# 根据server数调整
# 根据 rank 选择 IP 地址

SERVER_COUNT=$SERVER_COUNT
PROCESSES_PER_SERVER=$PROCESSES_PER_SERVER

# 计算当前 Rank 属于哪个虚拟节点(Node ID)
NODE_ID=\$(( RANK / PROCESSES_PER_SERVER ))

if [ \$NODE_ID -lt \$SERVER_COUNT ]; then
    TARGET_IP="192.\$((NODE_ID + 1)).5.5"
    export HCCL_VM_HOST_IP=\$TARGET_IP
    echo "[Wrapper] Rank: \$RANK (PID: \$$, PPID: \$PPID) -> Node: \$NODE_ID -> Binding HostIP: \$TARGET_IP  HCCL_VM_HOST_IP is \$HCCL_VM_HOST_IP" 
else
    echo "Error: Rank \$RANK exceeds simulated cluster size"
fi

# 执行实际的测试程序
exec "\$@"
EOF

chmod +x hccl_wrapper.sh

echo "step 2: start hccl_test..."
TOTAL_PROCS=$((SERVER_COUNT * PROCESSES_PER_SERVER))

# 清理执行目录的冗余文件
remove_files_by_prefix "sqe_info_rank_"
remove_files_by_prefix "mc_instr_info_rank_"

mpirun -np $TOTAL_PROCS \
        -f hostfile \
        -launcher fork \
        ./hccl_wrapper.sh \
        /srv/workspace/ccu_213_newhccp/work_code/hccl_test/bin/reduce_scatter_test -p 2 -d int32 -f 2 -o sum -w 0 -n 1 -a ccu_sched

echo "[Wrapper] FINISH"