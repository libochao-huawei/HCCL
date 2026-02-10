function Node(id, queueId, rankId, type, step, children, parents, info, genSemanticError) {
  this.nodeId = id;
  this.type = type;
  this.nodeName = nodeType2Name.get(type);
  this.x = 0;
  this.y = 0;
  this.category = GetCategory(type);
  this.localStep = step;
  this.children = children;
  this.parent = parents;
  this.info = info;
  this.queueId = queueId;
  this.rankId = rankId;
  this.level = -1;
  this.isLoop = false;
  this.index = 0; // tarjan算法节点index
  this.lowLink = 0; // tarjan算法节点lowLink值
  this.genSemanticError = genSemanticError;
}

function NodeData(node) {
  this.nodeId = node.nodeId;
  this.nodeName = nodeType2Name.get(node.type);
  this.x = node.x;
  this.y = node.y;
  this.category = GetCategory(node.type);
  this.localStep = node.localStep;
  this.info = node.info;
  this.queueId = node.queueId;
  this.isLoop = node.isLoop;
  this.genSemanticError = node.genSemanticError;
}

function RankTopology(rankId, nodes) {
  this.rankId = rankId;
  this.nodes = nodes;
  this.rankName = 'rank' + rankId;
}

function TopologyInfo(rankNodes, links) {
  this.rankNodes = rankNodes;
  this.links = links;
}

function NodeLink(source, target) {
  this.source = source;
  this.target = target;
}

const NodeType = {
  LOCAL_COPY: 0,
  LOCAL_REDUCE: 1,
  LOCAL_POST: 2,
  LOCAL_WAIT: 3,
  POST: 4,
  WAIT: 5,
  READ: 6,
  READ_REDUCE: 7,
  WRITE: 8,
  WRITE_REDUCE: 9,
  BEING_READ: 10,
  BEING_READ_REDUCE: 11,
  BEING_WRITTEN: 12,
  BEING_WRITTEN_REDUCE: 13,
  LOCAL_POST_SHADOW: 14,
  LOCAL_WAIT_FROM_SHADOW: 15,
  RESERVED: 16
};

const nodeType2Name = new Map([
  [NodeType.LOCAL_COPY, 'L_COPY'],
  [NodeType.LOCAL_REDUCE, 'L_REDUCE'],
  [NodeType.LOCAL_POST, 'L_POST'],
  [NodeType.LOCAL_WAIT, 'L_WAIT'],
  [NodeType.POST, 'POST'],
  [NodeType.WAIT, 'WAIT'],
  [NodeType.READ, 'READ'],
  [NodeType.READ_REDUCE, 'READ_REDUCE'],
  [NodeType.WRITE, 'WRITE'],
  [NodeType.WRITE_REDUCE, 'WRITE_REDUCE'],
  [NodeType.BEING_READ, 'BE_READ'],
  [NodeType.BEING_READ_REDUCE, 'BE_READ_REDUCE'],
  [NodeType.BEING_WRITTEN, 'BE_WRITTEN'],
  [NodeType.BEING_WRITTEN_REDUCE, 'BE_WRITTEN_REDUCE'],
  [NodeType.LOCAL_POST_SHADOW, 'L_POST_S'],
  [NodeType.LOCAL_WAIT_FROM_SHADOW, 'L_WAIT_FROM_S'],
  [NodeType.RESERVED, 'Dummy_Start']
]);

const Category = {
  DUMMY_START: 0,
  LOCAL_POST_TO_AND_SHADOW: 1,
  LOCAL_WAIT_FROM_AND_SHADOW: 2,
  READ_PAIR: 3,
  READ_REDUCE_PAIR: 4,
  WRITE_PAIR: 5,
  WRITE_REDUCE_PAIR: 6,
  LOCAL_COPY: 7,
  LOCAL_REDUCE: 8,
  POST_WAIT: 9
};

function GetCategory(nodeType) {
  switch (nodeType) {
    case NodeType.LOCAL_COPY:
      return Category.LOCAL_COPY;
    case NodeType.LOCAL_REDUCE:
      return Category.LOCAL_REDUCE;
    case NodeType.LOCAL_POST:
    case NodeType.LOCAL_POST_SHADOW:
      return Category.LOCAL_POST_TO_AND_SHADOW;
    case NodeType.LOCAL_WAIT:
    case NodeType.LOCAL_WAIT_FROM_SHADOW:
      return Category.LOCAL_WAIT_FROM_AND_SHADOW;
    case NodeType.POST:
    case NodeType.WAIT:
      return Category.POST_WAIT;
    case NodeType.READ:
    case NodeType.BEING_READ:
      return Category.READ_PAIR;
    case NodeType.WRITE:
    case NodeType.BEING_WRITTEN:
      return Category.WRITE_PAIR;
    case NodeType.READ_REDUCE:
    case NodeType.BEING_READ_REDUCE:
      return Category.READ_REDUCE_PAIR;
    case NodeType.WRITE_REDUCE:
    case NodeType.BEING_WRITTEN_REDUCE:
      return Category.WRITE_REDUCE_PAIR;
    default:
      return Category.DUMMY_START;
  }
}

const ResultStatus = {
  ORCHESTRA_TASK_FAILED: 0,
  // 资源相关的错误
  INVALID_STREAM_NUM: 10,
  INVALID_NOTIFY_NUM: 11,
  INVALID_SCRATCH_MEM_SIZE: 12,
  TRANSPORT_MATCH_FAILED: 13,
  INVALID_SINGLE_TASK: 20, // 单task校验失败
  GEN_GRAPH_FAILED: 30, // 成图失败
  MEMORY_CONFLICT: 40, // 检测出内存冲突
  GEN_FAILED_INCOMPLETE_SLICE: 50, // slice中的语义块不完整
  GEN_FAILED_MODIFY_SEMANTIC_FAILED: 51, // 修改语义块失败
  CHECK_FAILED_MISSING_SEMANTIC: 52, // 校验过程中，语义块有缺失
  CHECK_FAILED_UNEXPECTED_SEMANTIC: 53, // 校验过程中，出现了不期望的语义块
  CHECK_SUCCESS: 100 // 校验成功
};

function genErrorInfo(errSemantic, errMemory, errMessage) {
  return {
    errorSemantic: errSemantic,
    errorMemory: errMemory,
    errorMessage: errMessage
  };
}

function genErrorSemantic(rankId, nodeId, localStep, bufferType, startAddr) {
  return {
    rankId: rankId,
    nodeId: nodeId,
    localStep: localStep,
    bufferType: bufferType,
    startAddr: Number(startAddr)
  };
}

module.exports = {
  NodeInfo: Node,
  NodeData: NodeData,
  genErrorInfo: genErrorInfo,
  genErrorSemantic: genErrorSemantic,
  TopologyInfo: TopologyInfo,
  RankTopology: RankTopology,
  NodeType: NodeType,
  NodeLink: NodeLink,
  ResultStatus: ResultStatus
};
