// Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
const nodeInfo = require('./node_info');
const fs = require('fs');
const common = require('./common');

module.exports = class TopologyLayout {
  constructor(rankGraph, userName, tcName, rankSize, isBilateral) {
    this.rankGraph = rankGraph;
    this.startNode = {};
    this.userName = userName;
    this.tcName = tcName;
    // virtual2RealMap: 虚拟流归属的主/从流queueId映射关系{start node, queueId}
    this.virtual2RealMap = new Map();
    // realStreamList: 主/从流queueId列表
    this.realStreamList = new Map();
    this.xStep = 150;
    this.yStep = 100;
    this.level = 0;
    this.rankId = -1;
    this.rankSize = rankSize;
    this.rankNodeMap = new Map();
    this.connectMap = new Map();
    this.firstRealNodeStep = 0;
    this.loopNodes = [];
    this.tarjanIndex = 0; // tarjan算法的全局index
    this.tarjanStack = []; // tarjan算法中节点栈
    this.tarjanLoops = new Set(); // tarjan算法存储所有的环节点
    this.isBilateral = isBilateral;
  }

  /*
   * preProcess: 预处理
   * 功能说明：遍历虚拟流分支，统计每个虚拟流对应连接的主/从流个数，根据个数多少确定该条虚拟流归属与哪个主/从流。
   * 后续拓扑布局时，根据主/从流id顺序依次按组进行遍历布局设计。
   * 入参：rankGraph -- 该rank的拓扑数据
   * */
  preProcess() {
    let stack = [this.startNode];
    let searchVirtual = true;
    while (stack.length > 0) {
      let node = stack.pop();
      if (node && node.children.length === 0) {
        console.log('skip node:', node.nodeId);
        continue;
      }
      node.children.forEach((child) => {
        if (child.queueId === 0) {
          // 主流
          stack.push(child);
          return true;
        }
        if (searchVirtual && node.type === nodeInfo.NodeType.LOCAL_POST_SHADOW && child.type === nodeInfo.NodeType.LOCAL_WAIT_FROM_SHADOW) {
          console.log('virtual begin node:', child.queueId, ', ', child.nodeId);
          this.virtualStreamPreProcess(child);
          console.log('virtual end');
        } else if (node.type !== nodeInfo.NodeType.LOCAL_POST_SHADOW) {
          // 记录第一个非虚拟的节点
          if (this.firstRealNodeStep === 0) {
            this.firstRealNodeStep = node.localStep;
          }
          // 遍历搜集所有主/从流queueId
          searchVirtual = false;
          this.realStreamPreProcess(node);
        }
      });
    }
    // 重新确认从流起始节点
    this.resetSubStreamStartNode();
    // realStreamList按照queueId(从小到大)排序
    this.realStreamList = new Map(
      [...this.realStreamList].sort((a, b) => {
        return a[0] - b[0];
      })
    );
  }

  /* 获取与本节点所属相同流的父节点，流首节点->返回undefined */
  getParentOfSameStream(node)
  {
    let tarParent;
    node.parent.forEach((parent) => {
      if (parent.queueId === node.queueId) {
        tarParent = parent;
      }
    });
    return tarParent;
  }

  /* 重新寻找每个子流的起始节点 */
  resetSubStreamStartNode()
  {
    for (let [queueId, node] of this.realStreamList) {
      if (node.parent.length == 0) {
        continue;
      }
      let tarParent = this.getParentOfSameStream(node);
      // 没有同属于一条流的父节点：即本节点为流起始节点
      while (tarParent !== undefined) {
        node = tarParent;
        tarParent = this.getParentOfSameStream(node);
      }
      this.realStreamList.set(node.queueId, node);
    }
  }

  /* 查找所有从流节点（可能不是首节点，需要后续更新） */
  realStreamPreProcess(start) {
    let stack = [start];
    const visitedNodes = new Set();
    while (stack.length > 0) {
      let node = stack.pop();
      if (visitedNodes.has(node)) {
        continue;
      }
      visitedNodes.add(node);
      if (node && node.children.length === 0) {
        continue;
      }

      for (let i = 0; i < node.children.length; i++) {
        let child = node.children[i];
        if (node.type === nodeInfo.NodeType.LOCAL_POST_SHADOW && child.type === nodeInfo.NodeType.LOCAL_WAIT_FROM_SHADOW) {
          return;
        }
        // 遍历到虚拟流与主/从流的交叉节点
        if (node.type !== nodeInfo.NodeType.LOCAL_POST_SHADOW && !this.realStreamList.has(child.queueId)) {
          this.realStreamList.set(child.queueId, child);
        }
        stack.push(child);
      }
    }
  }

  /*
   * virtualStreamPreProcess(): 遍历虚拟流预处理
   * 功能说明：遍历虚拟流分支，统计虚拟流与主/从流的连接个数，以便对虚拟流和主/从流分组。
   * 业务侧布局先验知识：一般情况下虚拟流大多数只会与其中主/从流其中一个流交互较多。因此，按交互频繁度进行分组，有利于拓扑布局效果。
   * 分组方式：若某个虚拟流与主流连接更多，则将其与主流分为一组，后续布局设计时，也与主流一起布局。
   *         若与多个真实流连接相同，则以queueId大小进行排序。
   * */
  virtualStreamPreProcess(start) {
    let stack = [start];
    let interactCount = new Map();
    const visitedNodes = new Set();
    while (stack.length > 0) {
      let node = stack.pop();
      if (visitedNodes.has(node)) {
        continue;
      }
      visitedNodes.add(node);
      if (node && node.children.length === 0) {
        continue;
      }
      node.children.forEach((child) => {
        // 遍历到虚拟流与主/从流的交叉节点
        if (child.queueId !== node.queueId && child.type === nodeInfo.NodeType.LOCAL_WAIT_FROM_SHADOW) {
          if (!interactCount.has(child.queueId)) {
            interactCount[child.queueId] = 1;
          } else {
            interactCount[child.queueId] += 1;
          }
        }
        stack.push(child);
      });
      node.parent.forEach((parent) => {
        // 遍历到虚拟流与主/从流的交叉节点
        if (parent.queueId !== node.queueId) {
          interactCount[parent.queueId] += 1;
        }
      });
    }
    console.log('交叉点个数统计：', interactCount);
    if (interactCount.length === 0) {
      this.virtual2RealMap.set(start.nodeId, 0);
    } else {
      let count = 0;
      let queueId = 0;
      interactCount.forEach((key, value) => {
        if (value > count) {
          count = value;
          queueId = key.queueId;
        } else if (value === count) {
          queueId = queueId > key.queueId ? key.queueId : queueId;
        }
      });
      this.virtual2RealMap.set(start.nodeId, queueId);
    }
  }

  updateNodeXCoordinate(node, xFather, isFirst) {
    if (isFirst) {
      node.x = xFather;
    } else if (node.x === 0) {
      node.x = xFather + this.xStep;
    } else {
      let xNew = xFather + this.xStep;
      // x冲突，取较大值
      node.x = node.x > xNew ? node.x : xNew;
    }
  }

  updateNodeInfo(node, xFather, isFirst, curLevel) {
    if (node && node.level === -1) {
      node.level = curLevel;
    } else {
      // level冲突，取较小值
      node.level = node.level < curLevel ? node.level : curLevel;
    }
    this.updateNodeXCoordinate(node, xFather, isFirst);
  }

  /*
   * traversalVirtualStream: 遍历和布局虚拟流分支
   * */
  traversalVirtualStream(start, realQueueId, fatherX) {
    if (this.virtual2RealMap.get(start.nodeId) !== realQueueId) {
      // 非目标虚拟流，退出遍历
      return;
    }
    this.level += 1;
    this.updateNodeInfo(start, fatherX, true, this.level);
    let stack = [start];
    while (stack.length > 0) {
      let node = stack.pop();
      if (!node.children || node.children.length === 0) {
        continue;
      }
      node.children.forEach((child) => {
        this.updateNodeInfo(child, node.x, false, this.level);
        // 父子节点queueId相同，代表子节点仍然属于虚拟流分支
        if (child.queueId === node.queueId) {
          stack.push(child);
        } else {
          this.connectMap.set(node.queueId, child);
        }
      });
    }
  }

  /*
   * traversalVirtualStreamProc: 遍历和布局主/从流所属的虚拟分支。
   * */
  traversalVirtualStreamProc(realQueueId) {
    let stack = [this.startNode];
    let visitedVirtualStreams = new Set();
    while (stack.length > 0) {
      let node = stack.pop();
      if (node && node.type === nodeInfo.NodeType.RESERVED) {
        node.level = 0;
      }
      if (node && node.children.length === 0) {
        continue;
      }
      if (node.localStep >= this.firstRealNodeStep.localStep) {
        continue;
      }
      node.children.forEach((child) => {
        if (child.queueId === node.queueId) {
          this.updateNodeInfo(child, node.x, false, 0);
          stack.push(child);
          return;
        }
        if (
          node.type === nodeInfo.NodeType.LOCAL_POST_SHADOW &&
          child.type === nodeInfo.NodeType.LOCAL_WAIT_FROM_SHADOW &&
          this.virtual2RealMap.get(child.nodeId) === realQueueId
        ) {
          // 归属realQueueId的虚拟流分支
          if (visitedVirtualStreams.has(child.queueId)) {
            return;
          }
          visitedVirtualStreams.add(child.queueId);
          this.traversalVirtualStream(child, realQueueId, node.x);
        }
      });
    }
  }

  /*
   * traversalRealStreamProc: 遍历和布局主/从流及虚拟分支。
   * */
  traversalRealStreamProc(realQueueId, start) {
    let stack = [start];
    const usedQueueId = new Set();
    let xFather = 0;
    start.parent.forEach((parent) => {
      xFather = xFather < parent.x ? parent.x : xFather;
    });
    console.log("主/从流起始X坐标：", start.nodeId, ", ", xFather);
    this.updateNodeInfo(start, xFather, true, 0);
    while (stack.length > 0) {
      let node = stack.pop();
      if (node && node.children.length === 0) {
        continue;
      }
      node.children.forEach((child) => {
        if (child.queueId === realQueueId) {
          this.updateNodeInfo(child, node.x, false, 0);
          stack.push(child);
        }
      });
    }
  }

  // 按queueId依次遍历real+virtual流组
  traversalRealGroup() {
    // 初始节点的Y坐标
    let yBase = 35;
    let levelMax = 0;
    let queueIdOri = 0;
    const yUpdateInfo = new Map();
    this.realStreamList.forEach((node, queueId, map) => {
      this.level = 0;
      console.log('开始布局主/从流: ', queueId, ', 起始节点为： ', node.nodeId);
      // 遍历主/从流所属的虚拟流
      this.traversalVirtualStreamProc(queueId);
      // 遍历主/从流
      this.traversalRealStreamProc(queueId, node);
      levelMax = this.level + 1;
      if (queueId !== 0) {
        const lastOne = yUpdateInfo.get(queueIdOri);
        yBase = lastOne.yBase + lastOne.levelMax * this.yStep;
      }
      queueIdOri = queueId;
      console.log('coordinate info: ', queueId, ', ', yBase, ', levelMax: ' + levelMax);
      yUpdateInfo.set(queueId, { yBase: yBase, levelMax: levelMax });
    });
    return yUpdateInfo;
  }

  /*
   * setNodeYCoordinate: 节点Y坐标赋值;
   * */
  setNodeYCoordinate(node, isRealNode, yBase, levelMax) {
    if (node && isRealNode) {
      // 主/从流节点y坐标=stream yBase
      node.y = yBase;
      return;
    }
    // 虚拟流节点 y = yBase + (level_max - node_level) * yStep
    if (node.level === -1) {
      // todo:返回错误
      console.error('setNodeYCoordinate error: node id is ', node.nodeId);
    } else {
      node.y = yBase + this.yStep * (levelMax - node.level);
    }
  }

  /*
   * updateNodeYCoordinates: 更新拓扑图中所有节点Y坐标;
   * */
  updateNodeYCoordinates(yUpdateInfo) {
    let stack = [this.startNode];
    const visitedNodes = new Set();
    const visitedVirtualStream = new Set();
    while (stack.length > 0) {
      let node = stack.pop();
      //console.log('pop off stack:', node.nodeId);
      if (visitedNodes.has(node.nodeId)) {
        continue;
      }
      visitedNodes.add(node.nodeId);
      if (!yUpdateInfo.has(node.queueId)) {
        console.error('updateNodeYCoordinates error: node queue id is ', node.queueId, ', 节点Id: ', node.nodeId);
        continue;
      }
      let info = yUpdateInfo.get(node.queueId);
      let levelMax = info.levelMax;
      let yBase = info.yBase;
      this.setNodeYCoordinate(node, true, yBase, levelMax);
      if (node && node.children.length === 0) {
        console.log('skip node: ', node.nodeId);
        continue;
      }
      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];
        if (node.queueId !== child.queueId && node.type === nodeInfo.NodeType.LOCAL_POST_SHADOW && child.type === nodeInfo.NodeType.LOCAL_WAIT_FROM_SHADOW) {
          // 实际业务场景，可能从主/从流节点再回到虚拟流，新增visitedVirtualStream防止冗余遍历
          if (visitedVirtualStream.has(child.queueId)) {
            continue;
          }
          // 虚拟流分支节点更新
          visitedVirtualStream.add(child.queueId);
          this.updateVirtualNodeYCoordinates(child, yUpdateInfo);
        } else {
          //console.log('push to stack:', child.nodeId);
          stack.push(child);
        }
      }
    }
  }

  /*
   * updateVirtualNodeYCoordinates: 更新拓扑图中所有虚拟流分支的节点Y坐标;
   * */
  updateVirtualNodeYCoordinates(start, yUpdateInfo) {
    let stack = [start];
    if (!this.virtual2RealMap.has(start.nodeId)) {
      console.log('updateVirtualNodeYCoordinates error: get real stream id failed: ', start.nodeId);
      return;
    }
    let realQueueId = this.virtual2RealMap.get(start.nodeId);
    if (!yUpdateInfo.has(realQueueId)) {
      console.log('updateVirtualNodeYCoordinates error: get update y info failed: ', realQueueId);
      return;
    }
    let info = yUpdateInfo.get(realQueueId);
    while (stack.length > 0) {
      let node = stack.pop();
      this.setNodeYCoordinate(node, false, info.yBase, info.levelMax);
      if (node && node.children.length === 0) {
        continue;
      }
      node.children.forEach((child) => {
        if (child.queueId === node.queueId) {
          stack.push(child);
        }
        // end节点跳过
      });
    }
  }

  getRankInfo() {
    const rankInfo = [];
    for (let count = 0; count < this.rankSize; count++) {
      rankInfo.push({ rankId: count, name: 'rank' + count });
    }
    return rankInfo;
  }

  /*
   * dump2JsonFile: 图数据生成json文件;
   * 功能描述: 将拓扑图数据生成前端画图可用的node数据（nodes + links），并且dump为json文件。
   * */
  dump2JsonFile() {
    let nodeMaps = new Map();
    let linkMaps = new Map();
    let stack = [this.startNode];
    while (stack.length > 0) {
      let node = stack.pop();
      if (nodeMaps.has(node.nodeId)) {
        continue;
      }
      const newNode = new nodeInfo.NodeData(node);
      nodeMaps.set(node.nodeId, newNode);
      if (node.children.length === 0) {
        continue;
      }
      let childrenIds = [];
      node.children.forEach((child) => {
        let isLoop = node.isLoop && child.isLoop;
        childrenIds.push({ id: child.nodeId, isLoop: isLoop });
        stack.push(child);
      });
      if (!linkMaps.has(node.nodeId)) {
        linkMaps.set(node.nodeId, childrenIds);
      }
    }
    console.log('node num: ', nodeMaps.size, ', link num: ', linkMaps.size);
    let nodes = [];
    nodeMaps.forEach((node, nodeId, map) => {
      nodes.push(node);
    });
    let links = [];
    linkMaps.forEach((targets, source, map) => {
      targets.forEach((target) => {
        links.push({ source: source, target: target.id, isLoop: target.isLoop });
      });
    });
    let rankNodes = {
      rankId: this.startNode.children[0].rankId,
      nodes: nodes,
      rankName: 'rank' + this.startNode.children[0].rankId
    };
    let nodeDatas = {
      rankNodes: rankNodes,
      links: links,
      rankInfo: this.getRankInfo()
    };
    console.log('节点数据：', rankNodes.nodes.length, ', ', links.length);
    const jsonData = JSON.stringify(nodeDatas, null, 2);
    let filePath = common.userWorkDir(this.userName) + this.tcName + '/rank' + this.startNode.children[0].rankId + '_topo_unilateral.json';
    if (this.isBilateral) {
      filePath = common.userWorkDir(this.userName) + this.tcName + '/rank' + this.startNode.children[0].rankId + '_topo_bilateral.json';
    }
    fs.writeFileSync(filePath, jsonData, (err) => {
      if (err) {
        console.error('Error writing file', err);
        return { ret: 1, info: err };
      }
    });
  }

  /*
   * protoChange2Topology: proto数据转为拓扑布局使用的树形结构数据。
   * 功能描述：1. 将业务生成的protoBuffer数据转为拓扑布局设计使用的树形结构数据（方便遍历）；
   *         2. 记录跨rank的节点连接关系表；
   * */
  protoChange2Topology(interRankLinks, nodesMap) {
    // 生成rank_info文件
    this.rankNodeMap.clear();
    this.rankGraph.nodes.forEach((node) => {
      let nodeDescribe = node.nodeDescribe || (node.nodeDescribe = '');
      let newNode = new nodeInfo.NodeInfo(node.nodeId, node.queueId, node.rankId, node.nodeType, node.localStep, [], [], nodeDescribe, node.genSemanticError);
      this.rankNodeMap.set(node.nodeId, newNode);
      nodesMap.set(node.nodeId, node.localStep);
    });
    let rankTopo = {};
    this.rankGraph.nodes.forEach((node) => {
      let nodeNew = this.rankNodeMap.get(node.nodeId);
      node.parents.forEach((parentId) => {
        let parent = this.rankNodeMap.get(parentId);
        if (parent) {
          (parent.children || (parent.children = [])).push(nodeNew);
        }
      });
      node.children.forEach((childId) => {
        let child = this.rankNodeMap.get(childId);
        if (!child) {
          interRankLinks.push({ source: node.localStep, target: childId });
        } else {
          (child.parent || (child.parent = [])).push(nodeNew);
        }
      });
      if (!node.parents || (node.parents && node.parents.length === 0)) {
        rankTopo = nodeNew;
      }
    });
    this.startNode = rankTopo;
    this.startNode.x = 65; // 起始点x坐标
    this.rankId = this.startNode.children[0].rankId;
    this.realStreamList.set(this.startNode.queueId, this.startNode);
  }

  checkVirtualConflictNode(node, lastX) {
    let nextNode = null;
    let isComplete = false;
    if (!node.children || node.children.length === 0) {
      lastX = lastX > node.x ? lastX : node.x;
      isComplete = true;
    }
    node.parent.forEach((parent) => {
      if (parent.queueId !== node.queueId) {
        lastX = lastX > parent.x ? lastX : parent.x;
        isComplete = true;
      }
    });
    node.children.forEach((child) => {
      if (child.queueId !== node.queueId) {
        lastX = lastX > child.x ? lastX : child.x;
        isComplete = true;
      } else {
        nextNode = child;
      }
    });
    return { isCompleted: isComplete, lastX: lastX, child: nextNode };
  }

  checkSubRealConflictNode(node, lastX) {
    let nextNode = null;
    let isComplete = false;
    if (!node.children || node.children.length === 0) {
      lastX = lastX > node.x ? lastX : node.x;
      isComplete = true;
    }
    node.parent.forEach((parent) => {
      if (parent.queueId !== node.queueId && parent.queueId === 0) {
        lastX = lastX > parent.x ? lastX : parent.x;
        isComplete = true;
      }
    });
    node.children.forEach((child) => {
      if (child.queueId !== node.queueId) {
        if (child.queueId === 0) {
          lastX = lastX > child.x ? lastX : child.x;
          isComplete = true;
        }
      } else {
        nextNode = child;
      }
    });
    return { isCompleted: isComplete, lastX: lastX, child: nextNode };
  }

  /*
   * optimizeVirtualStreamNode：虚拟流分支节点X坐标优化
   * 1. 起始节点入栈，node.x = father.x, firstX = node.x；
   * 2. 结尾节点 ———— 父节点个数>1, 子节点个数>1，或 子节点为主/从流节点；node.x = child.x, lastX = node.x;
   * 3. 中间节点 ———— 入栈；
   * 4. 起始-结尾之间的节点 ———— node.x = firstX + (i + 1) * lastX / (num + 1)；
   * 5. 上述计算完成后，起始节点=结尾节点 ———— firstX = lastX, 继续遍历；
   * */
  optimizeVirtualStreamNode(start, optVirtual) {
    let nodeStack = [];
    let firstX = start.x;
    let lastX = 0;
    let stack = [start];
    let completeCount = 0;
    while (stack.length > 0) {
      let node = stack.pop();
      nodeStack.push(node);
      let result = optVirtual ? this.checkVirtualConflictNode(node, lastX) : this.checkSubRealConflictNode(node, lastX);
      if (result.isCompleted) {
        completeCount += 1;
      }
      if (result.child) {
        stack.push(result.child);
      }
      lastX = result.lastX;
      // 计算中间节点X坐标
      if (completeCount === 2) {
        const size = nodeStack.length - 1;
        // 两种场景：1. lastX比预期的小，则按照xStep步进计算x坐标；2. lastX比预期的大，则根据firstX-lastX平方计算x坐标
        if (lastX > firstX + this.xStep * size) {
          nodeStack[size].x = lastX;
          for (let i = 1; i < size; i++) {
            nodeStack[i].x = firstX + (i * (lastX - firstX)) / size;
          }
        } else {
          for (let i = 1; i <= size; i++) {
            nodeStack[i].x = firstX + i * this.xStep;
          }
          lastX = nodeStack[size].x;
        }
        firstX = lastX;
        completeCount = 1;
        nodeStack = [nodeStack[size]];
      }
    }
  }

  /*
   * optimizeSubRealStreamNode：从流流分支节点X坐标优化
   * 1. 起始节点入栈，node.x = father.x, firstX = node.x；
   * 2. 结尾节点 ———— 父节点个数>1, 子节点个数>1，或 子节点为主/从流节点；node.x = child.x, lastX = node.x;
   * 3. 中间节点 ———— 入栈；
   * 4. 起始-结尾之间的节点 ———— node.x = firstX + (i + 1) * lastX / (num + 1)；
   * 5. 上述计算完成后，起始节点=结尾节点 ———— firstX = lastX, 继续遍历；
   * */
  optimizeSubRealStreamNode(start) {
    let nodeStack = [];
    let firstX = start.x;
    let lastX = 0;
    let stack = [start];
    let completeCount = 0;
    while (stack.length > 0) {
      let node = stack.pop();
      nodeStack.push(node);
      let result = this.checkSubRealConflictNode(node, lastX);
      if (result.isCompleted) {
        completeCount += 1;
      }
      if (result.child) {
        stack.push(result.child);
      }
      lastX = result.lastX;
      // 计算中间节点X坐标
      if (completeCount === 2) {
        const size = nodeStack.length - 1;
        // 两种场景：1. lastX比预期的小，则按照xStep步进计算x坐标；2. lastX比预期的大，则根据firstX-lastX平方计算x坐标
        if (lastX > firstX + this.xStep * size) {
          nodeStack[size].x = lastX;
          for (let i = 1; i < size; i++) {
            nodeStack[i].x = firstX + (i * (lastX - firstX)) / size;
          }
        } else {
          for (let i = 1; i <= size; i++) {
            nodeStack[i].x = firstX + i * this.xStep;
          }
        }
        firstX = lastX;
        completeCount = 1;
        nodeStack = [nodeStack[size]];
      }
    }
  }

  /*
   * optimizeNodeXCoordinates: 优化节点X坐标
   * 功能描述：虚拟分支第一个节点与主/从流节点的x坐标相同，且其之后所有节点，若节点与主/从流节点直接相连，则将其x坐标设置相同；
   * 同时需要更新其之前的节点的所有节点x坐标（等分）。
   * */
  optimizeNodeXCoordinates(optVirtual) {
    let stack = [this.startNode];
    const visitedNodes = new Set();
    const visitedStreams = new Set();
    while (stack.length > 0) {
      let node = stack.pop();
      if (visitedNodes.has(node.nodeId)) {
        continue;
      }
      visitedNodes.add(node.nodeId);
      if (node && node.children.length === 0) {
        continue;
      }
      if (node.localStep >= this.firstRealNodeStep.localStep) {
        continue;
      }
      node.children.forEach((child) => {
        if (child.queueId === node.queueId) {
          stack.push(child);
          return;
        }
        if (
          optVirtual &&
          node.type === nodeInfo.NodeType.LOCAL_POST_SHADOW &&
          child.type === nodeInfo.NodeType.LOCAL_WAIT_FROM_SHADOW &&
          !visitedStreams.has(child.queueId)
        ) {
          this.optimizeVirtualStreamNode(child, optVirtual);
          visitedStreams.add(child.queueId);
          console.log('虚拟流' + child.queueId + ' 优化完毕。');
        }

        if (!optVirtual && node.type !== nodeInfo.NodeType.LOCAL_POST_SHADOW && !visitedStreams.has(child.queueId)) {
          this.optimizeSubRealStreamNode(child);
          visitedStreams.add(child.queueId);
          console.log('从流' + child.queueId + ' 优化完毕, start节点：' + child.nodeId);
        }
      });
    }
    console.log('X坐标优化完毕');
  }

  /*
   * tarjanAlg: Tarjan算法检测和提取有向图中的环。
   * 1. 初始化每个节点的遍历状态status=0；（状态值分别为0->未访问，1->已访问）
   * 2. 从start节点开始DFS遍历，每个节点随着遍历初始化Dfn和Low值 = Dfn=Low=++index;
   * 3. 同时节点依次入栈stack，标记节点状态status=1
   * 4. 遍历节点的所有children，进行如下操作：
   *   4.1 若遍历到叶子节点（即节点无children），则进行如下操作：
   *     4.1.1 若节点的Dfn === Low，则节点出栈，并且记录该节点为一个强连通分量的起点；
   *     4.1.2 返回父节点处理流程...
   *   4.2 若child status==0，则进行如下操作：
   *     4.2.1 更新当前节点的Low值：Low_cur = min(Low_cur, Low_child);
   *     4.2.2 以child为节点，继续递归进行tarjanAlg遍历操作...
   *   5.1 若child status!=0，但child已在栈中，同样更新当前节点Low值：Low_cur = min(Low_cur, Low_child);
   * */
  tarjanAlg(node) {
    // 初始化index和lowLink值
    node.index = node.lowLink = ++this.tarjanIndex;
    this.tarjanStack.push(node);
    // 遍历node子节点
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      // index和lowLink合法值>0，若其值===0，则代表该节点未被访问过
      if (child.index === 0 && child.lowLink === 0) {
        this.tarjanAlg(child);
        node.lowLink = Math.min(node.lowLink, child.lowLink);
      } else if (this.tarjanStack.indexOf(child) !== -1) {
        node.lowLink = Math.min(node.lowLink, child.index);
      }
    }
    if (node.index === node.lowLink) {
      let loopNodes = [];
      while (this.tarjanStack.length > 0) {
        const ele = this.tarjanStack.pop();
        loopNodes.push(ele);
        if (ele.nodeId === node.nodeId) {
          break;
        }
      }
      // 仅记录size>1的环（即环最少组成节点为2）
      if (loopNodes.length > 1) {
        loopNodes.forEach((ele) => {
          ele.isLoop = true;
        });
        this.tarjanLoops.add(loopNodes);
      }
    }
  }

  printRankLoopInfo() {
    if (this.tarjanLoops.size === 0) {
      console.log('there is no loop in topology of rank' + this.rankId);
    } else {
      console.log('there is/are ' + this.tarjanLoops.size + ' loop(s) in topology of rank' + this.rankId);
      console.log('loop nodes in topology of rank' + this.rankId + ' as follow:');
      let count = 0;
      for (let loop of this.tarjanLoops) {
        count += 1;
        let info = '第' + count + '个loop包含的节点有: ';
        loop.forEach((node) => {
          info += node.nodeId + ' ';
        });
        console.log(info);
      }
    }
  }

  /*
   * topologyLayoutRun: 拓扑布局设计主函数。拓扑布局设计主要步骤如下：
   * step1: preProcess预处理 ———— 虚拟流和主/从流进行分组，并且收集所有主/从流queueId；
   * step2: traversalGroupStream分组遍历 ———— 按照queueId顺序遍历一组流分支;
   * step3: updateNodeYCoordinates ———— 遍历拓扑图，计算&更新y坐标;
   * step4: dump2JsonFile ———— 重新遍历拓扑图，生成前端所需画图拓扑信息，且保存为json文件；
   * */
  async topologyLayoutRun(interRankLinks, nodesMap) {
    // step0: 数据转换
    this.protoChange2Topology(interRankLinks, nodesMap);
    // step1: 拓扑图判环
    this.tarjanAlg(this.startNode);
    this.printRankLoopInfo();
    console.log('开始布局rank' + this.rankId + '拓扑图');
    // step2: 预处理
    this.preProcess();
    // step3: 分组流遍历布局
    const yUpdateInfo = this.traversalRealGroup();
    // step4: 优化x坐标
    this.optimizeNodeXCoordinates(false);
    this.optimizeNodeXCoordinates(true);
    // step5: 更新y坐标
    this.updateNodeYCoordinates(yUpdateInfo);
    console.log('Y坐标更新完毕');
    // step6: 生成前端画图所需数据，且保存为json文件。
    this.dump2JsonFile();
  }
};
