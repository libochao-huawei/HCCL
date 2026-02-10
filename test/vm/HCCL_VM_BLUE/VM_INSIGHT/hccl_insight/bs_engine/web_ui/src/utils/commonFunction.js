import { ResultStatus } from './commonConstant.js';

export function canGetSemanticResult(errorType) {
  return [
    ResultStatus.CHECK_SUCCESS,
    ResultStatus.GEN_FAILED_INCOMPLETE_SLICE,
    ResultStatus.GEN_FAILED_MODIFY_SEMANTIC_FAILED,
    ResultStatus.CHECK_FAILED_MISSING_SEMANTIC,
    ResultStatus.CHECK_FAILED_UNEXPECTED_SEMANTIC
  ].includes(errorType);
}

export function getRank(options, id) {
  return options.find((item) => item.value === id);
}
export function getNode(options, id) {
  return options.find((item) => item.nodeId === id);
}

export function needFromLogToSemantic(errorType) {
  return [ResultStatus.CHECK_FAILED_MISSING_SEMANTIC, ResultStatus.CHECK_FAILED_UNEXPECTED_SEMANTIC].includes(errorType);
}

export const showRootRank = (opType) => {
  return ['HCCL_CMD_REDUCE', 'HCCL_CMD_BROADCAST', 'HCCL_CMD_SCATTER'].includes(opType);
};

export const showSrcAndDstRank = (opType) => {
  return ['HCCL_CMD_SEND'].includes(opType);
};

export const showReduceType = (opType) => {
  return ['HCCL_CMD_ALLREDUCE', 'HCCL_CMD_REDUCE', 'HCCL_CMD_REDUCE_SCATTER'].includes(opType);
};
