const fs = require('fs');
const errorCodeMap = require('../helper/errorCodeMap');
const errorCode = require('../helper/errorCode');
const replaceConsole = require('../../app/lib/logger');
const fsExtra = require('fs-extra');

function initWorkspace() {
  // 环境变量
  if (!process.env.WORKSPACE_DIR) {
    // 默认工作空间为当前目录
    process.env.WORKSPACE_DIR = process.cwd() + '/work_space';
    console.log('用户未配置工作空间，默认路径为:', process.env.WORKSPACE_DIR);
  } else {
    console.log('用户已配置了工作空间，路径为:', process.env.WORKSPACE_DIR);
  }
  return process.env.WORKSPACE_DIR;
}

/*
 * walk: 遍历文件夹，寻找目标文件
 * 入参：level —— 限制文件夹的层数
 * */
function walk(path, level, levelMax, callback) {
  if (level > levelMax) {
    return;
  }
  const files = fs.readdirSync(path);
  files.forEach(function (fileName) {
    if (fs.statSync(path + '/' + fileName).isFile()) {
      if (!fileName.endsWith('.json')) {
        return true;
      }
      callback(path + '/' + fileName);
    } else {
      walk(path + '/' + fileName, callback, level++);
    }
  });
}

// 读取预置用例列表配置文件
function getPreSetTcList(configDir) {
  const preSetList = new Set();
  walk(configDir, 1, 2, (filePath) => {
    let ret = readJsonFile(filePath, '读取预置用例列表错误');
    if (ret.code !== errorCode.CE000) {
      console.error('读取预置用例配置列表文件失败');
      return preSetList;
    }
    const tcConfigs = ret.data;
    if (!tcConfigs.caselist) {
      tcConfigs.caselist = [];
    }
    tcConfigs.caselist.forEach((tc) => {
      preSetList.add({ testSuite: tc.testSuite, testCase: tc.testCase });
    });
  });
  return preSetList;
}

function checkTestCase(tcList, tcData) {
  tcList.forEach((tc) => {
    if (tc.testCase === tcData.testCase) {
      console.error('The testcase ' + tcData.testCase + ' already exists.');
      return genRespFormat(errorCode.CE011);
    }
  });
  return genRespFormat(errorCode.CE000);
}

function parseTcConfigsFile(filePath) {
  const configData = JSON.parse(require('fs').readFileSync(filePath, 'utf8'));
  console.log('user upload tc config file path: ', filePath);
  const tcConfigs = new Map();
  for (let i = 0; i < configData.caselist.length; i++) {
    const item = configData.caselist[i];
    if (tcConfigs.has(item.testSuite)) {
      const tcList = tcConfigs.get(item.testSuite);
      let ret = checkTestCase(tcList, item);
      if (ret.code !== errorCode.CE000) {
        return ret;
      }
      tcList.push(item);
      tcConfigs.set(item.testSuite, tcList);
    } else {
      tcConfigs.set(item.testSuite, [item]);
    }
  }
  console.log('parse tc config file success.');
  return genRespFormat(errorCode.CE000, '', tcConfigs);
}

function getUserConfigDir(userName) {
  return process.env.WORKSPACE_DIR + '/' + userName + '/config/';
}

function getCommonConfigDir() {
  const comPath = process.env.WORKSPACE_DIR + '/common/';
  if (!fs.existsSync(comPath)) {
    console.log('创建目录：', comPath);
    fsExtra.mkdirsSync(comPath);
  }
  return comPath;
}

function getUserWorkDir(userName) {
  return process.env.WORKSPACE_DIR + '/' + userName + '/';
}

function splitTcName(tcName) {
  if (!tcName || tcName.indexOf('.') === -1 || tcName.indexOf('.') === 0 || tcName.indexOf('.') === tcName.length) {
    return genRespFormat(errorCode.CE005);
  }
  const splitIdx = tcName.indexOf('.');
  const testSuite = tcName.substring(0, splitIdx);
  const testCase = tcName.substring(splitIdx + 1);
  return genRespFormat(errorCode.CE000, '', { testSuite: testSuite, testCase: testCase });
}

// 清除用例结果: 删除用例或修改用例后，需要清除该用例之前的运行结果
function clearTcResult(userName, testSuite, testCase) {
  const fullTcName = testSuite + '.' + testCase;
  const resPath = getUserWorkDir(userName) + fullTcName;
  if (fs.existsSync(resPath)) {
    fsExtra.removeSync(resPath);
  }
}

function checkAndModifyTestName(userName, testSuite, testCase, newConfig) {
  const configPath = getUserConfigDir(userName) + testSuite + '.json';
  if (!fs.existsSync(configPath)) {
    console.error('test suite file path:', configPath);
    return genRespFormat(errorCode.CE006);
  }
  try {
    const newCaseList = [];
    let found = false;
    const configData = JSON.parse(require('fs').readFileSync(configPath, 'utf8'));
    configData.caselist.forEach((item) => {
      if (item.testCase === testCase) {
        found = true;
        if (!newConfig) {
          // 删除用例：此处跳过
          console.log('delete testcase success: ', testCase);
          return;
        }
        console.log('modify testcase success:', testCase);
        item = newConfig; // 修改用例，此处修改
      }
      newCaseList.push(item);
    });
    if (!found) {
      console.error('There is no such test case: ' + testSuite + '.' + testCase);
      return genRespFormat(errorCode.CE007);
    }
    fs.writeFileSync(configPath, JSON.stringify({caselist: newCaseList}, null, 2), (err) => {
      if (err) {
        console.error('Error writing file: ', err);
        return genRespFormat(errorCode.CE008);
      }
    });
    clearTcResult(userName, testSuite, testCase);
  } catch (err) {
    console.error('modify testcase config fail: ', err);
    return genRespFormat(newConfig ? errorCode.CE009 : errorCode.CE010);
  }
  console.log(newConfig ? 'modify' : 'delete' + 'testcase config success.');
  return genRespFormat(errorCode.CE000);
}

function readJsonFile(filePath, errMsg) {
  try {
    const result = JSON.parse(require('fs').readFileSync(filePath, 'utf8'));
    return genRespFormat(errorCode.CE000, '', result);
  } catch (err) {
    console.error('read file ' + filePath.substring(filePath.lastIndexOf('/')) + ' fail: ', err);
    return genRespFormat(errorCode.CE101, errMsg);
  }
}

function getBufSemantic(semantic) {
  return {
    startAddr: Number(semantic.startAddr),
    size: Number(semantic.size),
    isReduce: semantic.isReduce,
    reduceType: semantic.reduceType,
    srcBufs: semantic.srcBufs,
    affectedGlobalSteps: semantic.affectedGlobalSteps,
    invalid: semantic.invalid
  };
}

function getSemanticData(rankSemantic, nodeSemantic) {
  const semanticData = {
    inputBufferSemantics: null,
    outputBufferSemantics: null,
    inputCCLBufferSemantics: null,
    outputCCLBufferSemantics: null,
    scratchBufferSemantics: null
  };

  if (nodeSemantic.inputBufferSemantics) {
    semanticData.inputBufferSemantics = [];
    nodeSemantic.inputBufferSemantics.forEach((item) => {
      semanticData.inputBufferSemantics.push(getBufSemantic(item));
    });
  }
  if (nodeSemantic.outputBufferSemantics) {
    semanticData.outputBufferSemantics = [];
    nodeSemantic.outputBufferSemantics.forEach((item) => {
      semanticData.outputBufferSemantics.push(getBufSemantic(item));
    });
  }
  if (nodeSemantic.inputCCLBufferSemantics) {
    semanticData.inputCCLBufferSemantics = [];
    nodeSemantic.inputCCLBufferSemantics.forEach((item) => {
      semanticData.inputCCLBufferSemantics.push(getBufSemantic(item));
    });
  }
  if (nodeSemantic.outputCCLBufferSemantics) {
    semanticData.outputCCLBufferSemantics = [];
    nodeSemantic.outputCCLBufferSemantics.forEach((item) => {
      semanticData.outputCCLBufferSemantics.push(getBufSemantic(item));
    });
  }
  if (nodeSemantic.scratchBufferSemantics) {
    semanticData.scratchBufferSemantics = [];
    nodeSemantic.scratchBufferSemantics.forEach((item) => {
      semanticData.scratchBufferSemantics.push(getBufSemantic(item));
    });
  }
  return {
    inputSize: Number(rankSemantic.inputSize),
    outputSize: Number(rankSemantic.outputSize),
    inputCCLSize: Number(rankSemantic.inputCCLSize),
    outputCCLSize: Number(rankSemantic.outputCCLSize),
    scratchSize: Number(rankSemantic.scratchSize),
    semantic: semanticData
  };
}
const genRespFormat = (code, msg, resData, status) => {
  let message = '';
  if (msg && msg !== '') {
    message = msg;
  } else {
    message = errorCodeMap.get(code);
  }

  return {
    code: code,
    message: message,
    data: resData,
    status: status
  };
};

function configUserLog(userName, isUploadFile) {
  if (!userName || userName.trim() === '') {
    console.error('用户名：', userName);
    return genRespFormat(errorCode.CE014);
  }
  replaceConsole(userName);
  return genRespFormat(errorCode.CE000);
}

function checkTcConfig(userName, config, isUploadFile) {
  let ret = configUserLog(userName);
  if (!config || config.testSuite === '' || config.testSuite.trim() === '') {
    console.error('wrong testsuite name:', config.testSuite);
    return genRespFormat(errorCode.CE016);
  }
  if (config.testCase === '' || config.testCase.trim() === '') {
    console.error('wrong testcase name:', config.testCase);
    return genRespFormat(errorCode.CE016);
  }
  return genRespFormat(errorCode.CE000);
}

function checkGetParams(req) {
  if (!req.query.user_name || req.query.user_name.trim() === '') {
    console.error('用户名：', req.query.user_name);
    return genRespFormat(errorCode.CE014);
  }
  replaceConsole(req.query.user_name);
  if (!req.query.cmd || req.query.cmd.trim() === '') {
    console.error('cmd name：', req.query.cmd);
    return genRespFormat(errorCode.CE015);
  }
  if (req.query.cmd !== 'start' && req.query.cmd !== 'get_msg' && req.query.cmd !== 'quit' && req.query.cmd !== 'register') {
    if (!req.query.tc_name || req.query.tc_name.trim() === '') {
      console.error('tc name：', req.query.tc_name);
      return genRespFormat(errorCode.CE016);
    }
  }
  if (req.query.cmd === 'get_semantic') {
    if (!req.query.rank_id || req.query.rank_id.trim() === '') {
      return genRespFormat(errorCode.CE017);
    }
    if (!req.query.local_step || req.query.local_step.trim() === '') {
      return genRespFormat(errorCode.CE018);
    }
  } else if (req.query.cmd === 'switch_rank') {
    if (!req.query.origin_rank || req.query.origin_rank.trim() === '') {
      return genRespFormat(errorCode.CE017);
    }
    if (!req.query.target_rank || req.query.target_rank.trim() === '') {
      return genRespFormat(errorCode.CE017);
    }
    if (!req.query.origin_local_step || req.query.origin_local_step.trim() === '') {
      return genRespFormat(errorCode.CE018);
    }
  }
  return genRespFormat(errorCode.CE000);
}

function writeObj2File(jsonData, filePath) {
  fs.writeFileSync(filePath, jsonData, (err) => {
    if (err) {
      console.error('Error writing file', err);
      return genRespFormat(errorCode.CE102);
    }
  });
  return genRespFormat(errorCode.CE000);
}

module.exports = {
  initWorkspace: initWorkspace,
  walk: walk,
  writeObj2File: writeObj2File,
  getPreSetTcList: getPreSetTcList,
  checkTestCase: checkTestCase,
  parseTcConfigsFile: parseTcConfigsFile,
  userConfigDir: getUserConfigDir,
  comCfgDir: getCommonConfigDir,
  userWorkDir: getUserWorkDir,
  splitTcName: splitTcName,
  checkAndModifyTestName: checkAndModifyTestName,
  readJsonFile: readJsonFile,
  getSemanticData: getSemanticData,
  genRespFormat: genRespFormat,
  checkGetParams: checkGetParams,
  checkTcConfig: checkTcConfig,
  configUserLog: configUserLog
};
