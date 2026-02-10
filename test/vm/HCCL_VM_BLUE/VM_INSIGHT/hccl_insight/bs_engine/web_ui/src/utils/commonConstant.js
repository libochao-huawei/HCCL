export const Colors = ['#5470c6', '#91cc75', '#f2b705', '#5755D9', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#40a9ff'];
export const ErrorType = ['CHECK_SUCCESS', 'MEMORY_CONFLICT', 'MEMORY_SEMANTIC_ERROR'];

export const MemOp = ['READ', 'WRITE', 'WRITE_WITH_REDUCE'];

export const HcclReduceOp = [
  {
    label: 'SUM',
    value: 'HCCL_REDUCE_SUM'
  },
  {
    label: 'PROD',
    value: 'HCCL_REDUCE_PROD'
  },
  {
    label: 'MAX',
    value: 'HCCL_REDUCE_MAX'
  },
  {
    label: 'MIN',
    value: 'HCCL_REDUCE_MIN'
  }
  //'HCCL_REDUCE_RESERVED'
];

export const BufferType = ['INPUT', 'OUTPUT', 'INPUT_CCL', 'OUTPUT_CCL', 'SCRATCH', 'BUFFER_RESERVED'];

export const HcclCMDType = [
  // 'HCCL_CMD_INVALID',
  {
    label: 'Broadcast',
    value: 'HCCL_CMD_BROADCAST'
  },
  {
    label: 'AllReduce',
    value: 'HCCL_CMD_ALLREDUCE'
  },
  {
    label: 'Reduce',
    value: 'HCCL_CMD_REDUCE'
  },
  {
    label: 'SendRecv',
    value: 'HCCL_CMD_SEND'
  },
  {
    label: 'AllGather',
    value: 'HCCL_CMD_ALLGATHER'
  },
  {
    label: 'ReduceScatter',
    value: 'HCCL_CMD_REDUCE_SCATTER'
  },
  {
    label: 'AllToAll',
    value: 'HCCL_CMD_ALLTOALL'
  },
  {
    label: 'AllToAllV',
    value: 'HCCL_CMD_ALLTOALLV'
  },
  {
    label: 'AllToAllVC',
    value: 'HCCL_CMD_ALLTOALLVC'
  },
  {
    label: 'Scatter',
    value: 'HCCL_CMD_SCATTER'
  },
  {
    label: 'BatchSendRecv',
    value: 'HCCL_CMD_BATCH_SEND_RECV'
  }
  // 'HCCL_CMD_GATHER'
  //   'HCCL_CMD_FINALIZE',
  //   'HCCL_CMD_INTER_GROUP_SYNC',
  //   'HCCL_CMD_MAX'
];

export const OpMode = [
  {
    label: 'OPBASE（单算子）',
    value: 'OPBASE'
  },
  {
    label: 'OFFLOAD（图模式）',
    value: 'OFFLOAD'
  }
];

//芯片类型
export const DevType = [
  {
    label: '910',
    value: 'DEV_TYPE_910',
    npu: 8
  },
  {
    label: '310P3_V',
    value: 'DEV_TYPE_310P3_V', //v Duo
    npu: 8
  },
  {
    label: '310P3_Duo',
    value: 'DEV_TYPE_310P3_Duo', //v Duo
    npu: 8
  },
  {
    label: '910A2',
    value: 'DEV_TYPE_910B',
    npu: 16
  },
  {
    label: '910A3',
    value: 'DEV_TYPE_910_93',
    npu: 16
  }
];

export const HcclDataType = [
  {
    label: 'INT8',
    value: 'HCCL_DATA_TYPE_INT8'
  },
  {
    label: 'INT16',
    value: 'HCCL_DATA_TYPE_INT16'
  },
  {
    label: 'INT32',
    value: 'HCCL_DATA_TYPE_INT32'
  },
  {
    label: 'INT64',
    value: 'HCCL_DATA_TYPE_INT64'
  },
  {
    label: 'FP16',
    value: 'HCCL_DATA_TYPE_FP16'
  },
  {
    label: 'FP32',
    value: 'HCCL_DATA_TYPE_FP32'
  },
  {
    label: 'FP64',
    value: 'HCCL_DATA_TYPE_FP64'
  },
  {
    label: 'UINT8',
    value: 'HCCL_DATA_TYPE_UINT8'
  },
  {
    label: 'UINT16',
    value: 'HCCL_DATA_TYPE_UINT16'
  },
  {
    label: 'UINT32',
    value: 'HCCL_DATA_TYPE_UINT32'
  },
  {
    label: 'UINT64',
    value: 'HCCL_DATA_TYPE_UINT64'
  },
  {
    label: 'BFP16',
    value: 'HCCL_DATA_TYPE_BFP16'
  },
  {
    label: 'RESERVED',
    value: 'HCCL_DATA_TYPE_RESERVED'
  }
];

export const ResultStatus = {
  ORCHESTRA_TASK_FAILED: 0,
  // 资源相关的错误
  INVALID_STREAM_NUM: 10,
  INVALID_NOTIFY_NUM: 11,
  INVALID_SCRATCH_MEM_SIZE: 12,
  TRANSPORT_MATCH_FAILRD: 13,
  // 单task校验相关的报错
  INVALID_SINGLE_TASK: 20,
  // 成图相关的报错
  GEN_GRAPH_FAILED: 30,
  // 内存冲突相关的报错
  MEMORY_CONFLICT: 40,
  // 语义校验相关的报错
  GEN_FAILED_INCOMPLETE_SLICE: 50, // slice中的语义块不完整
  GEN_FAILED_MODIFY_SEMANTIC_FAILED: 51, // 修改语义块失败
  CHECK_FAILED_MISSING_SEMANTIC: 52, // 校验过程中，语义块有缺失
  CHECK_FAILED_UNEXPECTED_SEMANTIC: 53, // 校验过程中，出现了不期望的语义块
  // 校验成功，没有任何报错
  CHECK_SUCCESS: 100
};

export const Category = [
  {
    value: 'DUMMY_START',
    color: Colors[0]
  },
  {
    value: 'LOCAL_POST_TO_AND_SHADOW1',
    color: Colors[1]
  },
  {
    value: 'LOCAL_WAIT_FROM_AND_SHADOW',
    color: Colors[2]
  },
  {
    value: 'READ_PAIR',
    color: Colors[3]
  },
  {
    value: 'READ_REDUCE_PAIR',
    color: Colors[4]
  },
  {
    value: 'WRITE_PAIR',
    color: Colors[5]
  },
  {
    value: 'WRITE_REDUCE_PAIR',
    color: Colors[6]
  },
  {
    value: 'LOCAL_COPY',
    color: Colors[7]
  },
  {
    value: 'LOCAL_REDUCE',
    color: Colors[8]
  },
  {
    value: 'POST_WAIT',
    color: Colors[9]
  }
];

export const Envs = [
  {
    value: 'HCCL_HIGH_PERF_ENABLE'
  },
  {
    value: 'HCCL_DETERMINISTIC'
  },
  {
    value: 'HCCL_INTRA_PCIE_ENABLE'
  },
  {
    value: 'HCCL_INTRA_ROCE_ENABLE'
  },
  {
    value: 'HCCL_ALGO'
  },
  {
    value: 'HCCL_BUFFSIZE'
  },
  {
    value: 'HCCL_INTER_HCCS_DISABLE'
  },
  {
    value: 'HCCL_OP_EXPANSION_MODE'
  },
  {
    value: 'HCCL_CONCURRENT_ENABLE'
  },
  {
    value: 'HCCL_OP_BASE_FFTS_MODE_ENABLE'
  }
];

export const TopoConfigMode = [
  { value: 'SYMMETRIC', label: '对称' },
  { value: 'ASYMMETRIC', label: '非对称' }
];
