const express = require('express');
const proto = require('./protobuf_parse'); // 上面我们封装好的proto.js
const nodeInfo = require('./node_info');
const path = require('path');
const fs = require('fs');
const common = require('../service/common');
const errorCode = require('../helper/errorCode');
const TopologyLayout = require('./topology_layout');

module.exports = class CheckerTCResultParser {
  constructor(userName) {
    this.userName = userName;
    this.tcName = '';
    this.resultStatus = nodeInfo.ResultStatus.CHECK_SUCCESS;
    this.memConflict = null;
    this.workDir = common.userWorkDir(userName);
    // interRankLinks: 记录rank之前node的连接关系
    this.interRankLinks = new Map();
    this.analysisResults = {};
  }

  initConfig(userName, tcName) {
    this.userName = userName;
    this.tcName = tcName;
  }

  /*
   * 解析protoBuffer初始化，加载proto格式文件；
   * */
  init(tcName) {
    this.tcName = tcName;
    console.log('__dirname: ', __dirname);
    // 使用proto前先loadProtoDir
    proto.loadProtoDir([path.join(__dirname, '../data/proto/analysis_result.proto')]);
  }

  /*
   * isTcResultExist: 判断用例结果是否存在
   * */
  isTcResultExist(tcName) {
    let filePath = this.workDir + tcName + '/rank0_topo_unilateral.json';
    if (this.resultStatus === nodeInfo.ResultStatus.MEMORY_CONFLICT) {
      filePath = this.workDir + tcName + '/rank0_topo_bilateral.json';
    }
    return fs.existsSync(filePath);
  }

  getDataFromFile(filePath, errMsg) {
    let data = null;
    console.log('file:', filePath);
    if (fs.existsSync(filePath)) {
      let ret = common.readJsonFile(filePath, errMsg);
      if (ret.code !== errorCode.CE000) {
        console.error(ret.message);
        return data;
      }
      data = ret.data;
    }
    return data;
  }

  /*
   * getRunTcResult: 获取用例运行结果 —— rank0的拓扑数据 + dummy start节点semantic数据.
   * */
  getRunTcResult(tcName) {
    // 若用例结果已存在，则直接读取文件返回
    const resFilePath = this.workDir + tcName + '/tc_analysis_result.json';
    console.log('res file: ', resFilePath);
    if (fs.existsSync(resFilePath)) {
      return common.readJsonFile(resFilePath, '读取用例执行结果分析数据错误');
    }
    // 第一次执行用例，需要组装用例结果，并且将其保存为文件
    // 拓扑信息
    let topoFilePath = this.workDir + tcName + '/rank0_topo_unilateral.json';
    let errMsg = '读取rank0原始拓扑数据错误';
    if (this.resultStatus === nodeInfo.ResultStatus.MEMORY_CONFLICT) {
      topoFilePath = this.workDir + tcName + '/rank0_topo_bilateral.json';
      errMsg = '读取rank0双边拓扑数据错误';
    }
    let topoInfo = this.getDataFromFile(topoFilePath, errMsg);
    // 错误节点和错误语义信息
    const errInfoFilePath = this.workDir + this.tcName + '/error_info.json';
    let errInfo = this.getDataFromFile(errInfoFilePath, '读取用例错误结果数据失败');

    const result = {
      errorType: this.resultStatus,
      globalStep2LocalStep: [],
      topologyInfo: topoInfo,
      memStates: null,
      errorInfo: errInfo
    };
    // 将用例结果相关数据保存：防止二次执行后丢失
    const jsonData = JSON.stringify(result, null, 2);
    fs.writeFile(resFilePath, jsonData, (err) => {
      if (err) {
        console.error('Error writing file', err);
        return common.genRespFormat(errorCode.CE102);
      }
    });
    return common.genRespFormat(errorCode.CE000, '获取用例结果成功', result);
  }

  /*
   * getOtherResults: 获取其他用例运行错误结果
   * */
  getOtherResults(tcName) {
    const filePath = this.workDir + tcName + '/others_error.json';
    return common.readJsonFile(filePath, '读取用例执行结果其他类型错误失败');
  }

  /*
   * getNodeSemantic: 获取指定rank，指定节点的内存语义信息
   * */
  getNodeSemantic(rankId, localStep) {
    if (this.resultStatus === nodeInfo.ResultStatus.MEMORY_CONFLICT) {
      return common.genRespFormat(errorCode.CE000);
    }
    // 读取rank的语义信息文件
    const filePath = this.workDir + this.tcName + '/rank' + rankId + '_semantic.json';
    let ret = common.readJsonFile(filePath, '读取rank' + rankId + '节点语义数据错误');
    if (ret.code !== errorCode.CE000) {
      console.error('read file ' + filePath + ' failed.');
      return ret;
    }
    const rankSemantic = ret.data;
    console.log('filepath: ', filePath, ', ', rankSemantic.memStates.length);
    if (localStep > rankSemantic.memStates.length || localStep < 1) {
      console.error('Get node semantic failed. The local step ' + localStep + ' is out of range.');
      return common.genRespFormat(errorCode.CE018);
    }
    // 读取指定node语义数据
    let nodeSemantic = rankSemantic.memStates[localStep - 1];
    if (nodeSemantic.stateNoChange && nodeSemantic.stateNoChange === true) {
      console.log('the semantic of step ' + (localStep - 1) + ' is same as step ' + (nodeSemantic.localStep - 1));
      nodeSemantic = rankSemantic.memStates[nodeSemantic.localStep - 1];
    }
    console.log('get semantic result: ', localStep);
    return common.genRespFormat(errorCode.CE000, '', common.getSemanticData(rankSemantic, nodeSemantic));
  }

  /*
   * getStepStates: 根据local step获取其对应的内存语义信息；
   * */
  getStepStates(rankId, localStep) {
    const filePath = this.workDir + 'rank' + rankId + '_semantic.json';
    let ret = common.readJsonFile(filePath, '读取rank' + rankId + '节点语义数据错误');
    if (ret.code !== errorCode.CE000) {
      return ret;
    }
    const tarRankSemantic = ret.data;
    let stepState = tarRankSemantic.memStates[localStep - 1];
    if (stepState.stateNoChange && stepState.stateNoChange === true) {
      // 无改变，则寻找实际的local step的状态
      console.log('the semantic of step ' + (localStep - 1) + ' is same as step ' + (stepState.localStep - 1));
      localStep = stepState.localStep;
      stepState = tarRankSemantic.memStates[localStep - 1];
    }
    return common.genRespFormat(errorCode.CE000, '', common.getSemanticData(tarRankSemantic, stepState));
  }

  /*
   * switchRankTopoView: 用户切换拓扑视图；
   * */
  switchRankTopoView(rankId, topoView) {
    const tarRankTopoFile = this.workDir + this.tcName + '/rank' + rankId + '_topo_' + topoView + '.json';
    if (!fs.existsSync(tarRankTopoFile)) {
      return common.genRespFormat(errorCode.CE206);
    }
    const tarRankTopo = JSON.parse(require('fs').readFileSync(tarRankTopoFile, 'utf8'));

    const result = {
      errorType: this.resultStatus,
      globalStep2LocalStep: [],
      topologyInfo: tarRankTopo,
      memStates: null
    };
    return common.genRespFormat(errorCode.CE000, '', result);
  }

  /*
   * 切换rank，返回切换后rank节点的内存语义信息；
   * 功能描述：从当前rank(oriRankId，已选择节点oriNodeId(默认为dummyStart))，切换到目标rank(tarRankId);
   * 1. 若oriNodeId在tarRank存在配对节点（即存在跨rank相连的节点），则切换后，直接定位到该节点，且返回该节点semantic；
   * 2. 若oriNodeId在tarRank不存在配对节点，则切换后，默认显示dummy start节点semantic；
   * */
  switchRankNodeSemantic(oriRankId, tarRankId, oriLocalStep) {
    const isMemConflict = this.resultStatus === nodeInfo.ResultStatus.MEMORY_CONFLICT;
    let tarRankTopoFile = this.workDir + this.tcName + '/rank' + tarRankId + '_topo_unilateral.json';
    if (isMemConflict) {
      tarRankTopoFile = this.workDir + this.tcName + '/rank' + tarRankId + '_topo_bilateral.json';
    }
    console.log('tar file:', tarRankTopoFile);
    if (!fs.existsSync(tarRankTopoFile)) {
      return common.genRespFormat(isMemConflict ? errorCode.CE206 : errorCode.CE205);
    }
    const tarRankTopo = JSON.parse(require('fs').readFileSync(tarRankTopoFile, 'utf8'));

    const result = {
      errorType: this.resultStatus,
      globalStep2LocalStep: [],
      topologyInfo: tarRankTopo,
      memStates: null
    };
    // 内存冲突错误，只有拓扑信息，没有语义信息
    if (isMemConflict) {
      return common.genRespFormat(errorCode.CE000, '', result);
    }

    // 读取oriRank和tarRank的语义信息文件
    const oriFilePath = this.workDir + this.tcName + '/rank' + oriRankId + '_semantic.json';
    let ret = common.readJsonFile(oriFilePath, '读取rank' + oriRankId + '节点语义数据错误');
    if (ret.code !== errorCode.CE000) {
      return ret;
    }
    const oriRankSemantic = ret.data;
    if (oriLocalStep > oriRankSemantic.localStep2GlobalStep.length) {
      console.error('Switch rank node failed. The local step ' + oriLocalStep + ' is out of range.');
      return common.genRespFormat(errorCode.CE018);
    }

    // 跨rank节点映射，判断是否存在配对节点
    let tarLocalStep = 0;
    const oriRankLinks = this.interRankLinks.get(oriRankId);
    if (oriRankLinks) {
      oriRankLinks.forEach((linkPair) => {
        if (linkPair.source === oriLocalStep) {
          tarLocalStep = linkPair.target;
        }
      });
    }

    console.log('target local step: ', tarLocalStep);
    if (tarLocalStep !== 0) {
      let ret = this.getStepStates(tarRankId, tarLocalStep);
      if (ret.code !== errorCode.CE000) {
        return ret;
      }
      result.memStates = ret.data;
    }
    return common.genRespFormat(errorCode.CE000, '', result);
  }

  getInvalidNode(rankId, memState, errSemantics) {
    memState.inputBufferSemantics.forEach((inputBufferSemantic) => {
      if (!inputBufferSemantic.invalid) {
        return;
      }
      errSemantics.push(nodeInfo.genErrorSemantic(rankId, -1, memState.localStep));
    });
    memState.outputBufferSemantics.forEach((outputBufferSemantic) => {
      if (!outputBufferSemantic.invalid) {
        return;
      }
      console.log('output mem: ', rankId, -1, memState.outputBufferSemantics.localStep);
      errSemantics.push(nodeInfo.genErrorSemantic(rankId, -1, memState.localStep));
    });
    memState.inputCCLBufferSemantics.forEach((inputCCLBufferSemantic) => {
      if (!inputCCLBufferSemantic.invalid) {
        return;
      }
      errSemantics.push(nodeInfo.genErrorSemantic(rankId, -1, memState.localStep));
    });
    memState.inputCCLBufferSemantics.forEach((inputCCLBufferSemantic) => {
      if (!inputCCLBufferSemantic.invalid) {
        return;
      }
      errSemantics.push(nodeInfo.genErrorSemantic(memState.rankId, -1, memState.localStep));
    });
    memState.scratchBufferSemantics.forEach((scratchBufferSemantic) => {
      if (!scratchBufferSemantic.invalid) {
        return;
      }
      errSemantics.push(nodeInfo.genErrorSemantic(rankId, -1, memState.localStep));
    });
    return errSemantics;
  }

  getErrorSemanticInfo(resStatus, missingSemantics) {
    const errorSemantics = [];
    const rankGraphs = this.analysisResults.wholeGraph.rankGraphs;
    // 生成语义块错误: 遍历拓扑节点获取错误节点信息
    if (resStatus === nodeInfo.ResultStatus.GEN_FAILED_INCOMPLETE_SLICE || resStatus === nodeInfo.ResultStatus.GEN_FAILED_MODIFY_SEMANTIC_FAILED) {
      rankGraphs.forEach((rankGraph) => {
        rankGraph.nodes.forEach((node) => {
          if (!node.genSemanticError) {
            return;
          }
          errorSemantics.push(nodeInfo.genErrorSemantic(node.localStep.rank, node.nodeId, node.localStep.localStep));
        });
      });
    } else if (resStatus === nodeInfo.ResultStatus.CHECK_FAILED_UNEXPECTED_SEMANTIC) {
      // 语义块错误：遍历所有语义块获取localStep信息 --> 节点信息
      this.analysisResults.rankStates.forEach((rankState) => {
        rankState.memStates.forEach((memState) => {
          if (memState.stateNoChange) {
            return;
          }
          this.getInvalidNode(rankState.rankId, memState, errorSemantics);
        });
      });
    } else if (resStatus === nodeInfo.ResultStatus.CHECK_FAILED_MISSING_SEMANTIC) {
      missingSemantics.forEach((missEle) => {
        // 语义块缺失错误CHECK_FAILED_MISSING_SEMANTIC: 错误节点为rank最后一个节点
        rankGraphs.forEach((rankGraph) => {
          const rankId = rankGraph.nodes[1].rankId;
          if (rankId !== missEle.rankId) {
            return;
          }
          let maxLocalStep = 0;
          let lastNode = {};
          rankGraph.nodes.forEach((node) => {
            if (maxLocalStep > node.localStep) {
              return;
            }
            lastNode = node;
          });
          errorSemantics.push(nodeInfo.genErrorSemantic(rankId, lastNode.nodeId, lastNode.localStep.localStep, missEle.bufferType, missEle.startAddr));
        });
      });
    }
    return errorSemantics;
  }

  // 语义错误的场景：需要获取错误节点信息和错误语义信息 —— 用于前端进行节点定位
  saveErrorInfo() {
    let errInfo = {};
    // 校验语义错误：语义缺失
    if (this.resultStatus === nodeInfo.ResultStatus.CHECK_FAILED_MISSING_SEMANTIC) {
      if (!this.analysisResults.missingSemantic) {
        return common.genRespFormat(errorCode.CE203);
      }
      const errSemantics = this.getErrorSemanticInfo(this.resultStatus, this.analysisResults.missingSemantic);
      errInfo = nodeInfo.genErrorInfo(errSemantics, null, this.analysisResults.errorMessage);
    } else if (this.resultStatus === nodeInfo.ResultStatus.CHECK_FAILED_UNEXPECTED_SEMANTIC) {
      // 校验语义错误: 非预期语义
      const errSemantics = this.getErrorSemanticInfo(this.resultStatus, null);
      errInfo = nodeInfo.genErrorInfo(errSemantics, null, this.analysisResults.errorMessage);
    } else if (
      // 生成语义错误
      this.resultStatus === nodeInfo.ResultStatus.GEN_FAILED_INCOMPLETE_SLICE ||
      this.resultStatus === nodeInfo.ResultStatus.GEN_FAILED_MODIFY_SEMANTIC_FAILED
    ) {
      // 遍历节点或语义块，找出错误的节点信息
      const errSemantics = this.getErrorSemanticInfo(this.resultStatus, null);
      errInfo = nodeInfo.genErrorInfo(errSemantics, null, this.analysisResults.errorMessage);
    } else if (this.resultStatus === nodeInfo.ResultStatus.MEMORY_CONFLICT) {
      // 其他错误类型：如内存冲突
      errInfo = nodeInfo.genErrorInfo(null, this.analysisResults.memConflict, this.analysisResults.errorMessage);
    } else if (this.resultStatus !== nodeInfo.ResultStatus.CHECK_SUCCESS) {
      // 其他错误类型：如内存冲突
      errInfo = nodeInfo.genErrorInfo(null, null, this.analysisResults.errorMessage);
    }
    if (JSON.stringify(errInfo) === '{}') {
      return common.genRespFormat(errorCode.CE000);
    }
    // 存json文件
    const filePathStr = this.workDir + this.tcName + '/error_info.json';
    const missEleStr = JSON.stringify(errInfo, null, 2);
    let ret = common.writeObj2File(missEleStr, filePathStr);
    if (ret.code !== errorCode.CE000) {
      return ret;
    }
    return common.genRespFormat(errorCode.CE000);
  }

  /*
   * 解析rank语义信息
   * */
  saveRankStates(rankState) {
    if (!rankState) {
      return common.genRespFormat(errorCode.CE200);
    }
    console.log('rank semantic:', rankState.inputSize);
    console.log('work dir: ', this.workDir);

    const jsonData = JSON.stringify(rankState, null, 2);
    const filePath = this.workDir + this.tcName + '/rank' + rankState.rankId + '_semantic.json';
    return common.writeObj2File(jsonData, filePath);
  }

  /*
   * 拓扑布局
   * */
  parseAllTopology(wholeGraph, tcName, isBilateral) {
    const nodesMap = new Map();
    const interRankLinksOld = new Map();
    if (!wholeGraph || !wholeGraph.rankGraphs) {
      // 内存冲突报错：不能缺少双边语义拓扑图数据；其他类型：不能缺少原始拓扑图数据
      if (!isBilateral && this.analysisResults !== nodeInfo.ResultStatus.MEMORY_CONFLICT) {
        return common.genRespFormat(errorCode.CE205);
      } else if (isBilateral && this.analysisResults === nodeInfo.ResultStatus.MEMORY_CONFLICT) {
        return common.genRespFormat(errorCode.CE206);
      } else {
        console.warn(tcName + "用例执行结果缺失双边拓扑数据");
        return common.genRespFormat(errorCode.CE000);
      }
    }
    const rankSize = wholeGraph.rankGraphs.length;
    wholeGraph.rankGraphs.forEach((rankGraph) => {
      let topologyLayout = new TopologyLayout(rankGraph, this.userName, tcName, rankSize, isBilateral);
      let interRankLinksTmp = [];
      topologyLayout.topologyLayoutRun(interRankLinksTmp, nodesMap).then(() => {
        interRankLinksOld.set(topologyLayout.rankId, interRankLinksTmp);
      });
    });
    // 2. 重新设置rank间节点link
    interRankLinksOld.forEach((rankLinks, rankId, map) => {
      let newLinks = [];
      rankLinks.forEach((link) => {
        const child = nodesMap.get(link.target);
        newLinks.push({ source: link.source, target: child.localStep });
      });
      this.interRankLinks.set(rankId, newLinks);
    });
    return common.genRespFormat(errorCode.CE000);
  }

  /*
   * parseAnalysisResult: 读取并解析用例执行结果。
   * 1. 读取并解析业务输出的protoBuffer格式的用例结果文件.
   *    A. 拓扑图结构数据，并设计拓扑结构布局；
   *    B. 所有rank的内存语义数据、内存冲突错误数据、或其他类型错误数据；
   * 返回：1. 拓扑结构数据； 2. rank0的dummy start节点语义状态数据（固定为空，因此可以不返回）；
   * */
  async parseRun(tcName) {
    // 初始化
    this.init(tcName);
    // 解析protoBuffer数据
    const tcProtoBufferFile = this.workDir + this.tcName + '/analysis_result_binary.txt';
    if (!fs.existsSync(tcProtoBufferFile)) {
      return common.genRespFormat(errorCode.CE207);
    }
    console.log('tc proto buffer file: ', tcProtoBufferFile);
    try {
      // 注：此处不能加‘utf-8'，实为二进制文件
      const data = fs.readFileSync(tcProtoBufferFile);
      this.analysisResults = proto.read('AnalysisResult', data);
      // 1. 解析拓扑图数据
      let ret = this.parseAllTopology(this.analysisResults.wholeGraph, tcName, false);
      if (ret.code !== errorCode.CE000) {
        return ret;
      }
      console.log('单边拓扑图解析完成...');
      ret = this.parseAllTopology(this.analysisResults.bilateralGraph, tcName, true);
      if (ret.code !== errorCode.CE000) {
        console.error('双边拓扑解析失败...', ret.code);
        return ret;
      }
      console.log('双边拓扑图解析完成...');
      // 3. 解析语义数据
      this.resultStatus = this.analysisResults.resultStatus;
      console.log('tc status:', this.resultStatus, ', rank size:', this.analysisResults.rankSize);
      //console.log('semantic data: ', analysisResults.rankStates);
      // 4. 保存错误节点和错误语义信息
      ret = this.saveErrorInfo();
      if (ret.code !== errorCode.CE000) {
        return ret;
      }
      // 5. 解析保存语义信息
      if (this.resultStatus && Object.values(nodeInfo.ResultStatus).includes(this.resultStatus)) {
        if (!this.analysisResults.rankStates) {
          return common.genRespFormat(errorCode.CE200);
        }
        // 保存每个rank的语义信息
        this.analysisResults.rankStates.forEach((rankState) => {
          ret = this.saveRankStates(rankState);
        });
      } else {
        console.warn('未处理的用例错误类型: ' + this.resultStatus);
        return common.genRespFormat(errorCode.CE204);
      }
      if (ret.code !== errorCode.CE000) {
        return ret;
      }
    } catch (err) {
      console.error('parse tc proto-buffer result failed: ', err);
      return common.genRespFormat(errorCode.CE201);
    }
    return common.genRespFormat(errorCode.CE000);
  }

  getTcResult(tcName) {
    switch (this.resultStatus) {
      case nodeInfo.ResultStatus.CHECK_SUCCESS:
      case nodeInfo.ResultStatus.CHECK_FAILED_MISSING_SEMANTIC:
      case nodeInfo.ResultStatus.CHECK_FAILED_UNEXPECTED_SEMANTIC:
      case nodeInfo.ResultStatus.GEN_FAILED_INCOMPLETE_SLICE:
      case nodeInfo.ResultStatus.GEN_FAILED_MODIFY_SEMANTIC_FAILED:
      case nodeInfo.ResultStatus.MEMORY_CONFLICT:
        return this.getRunTcResult(tcName);
      default:
        return this.getOtherResults(tcName);
    }
  }
};
