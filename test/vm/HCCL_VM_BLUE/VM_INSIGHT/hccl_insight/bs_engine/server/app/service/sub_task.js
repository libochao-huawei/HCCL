const CheckerTCResultParser = require('./read_proto_data');
const { execSync, exec, fork } = require('child_process');
const common = require('./common');
const fsExtra = require('fs-extra');
const errorCode = require('../helper/errorCode');
const fs = require('fs');
const os = require('os');
const replaceConsole = require('../../app/lib/logger');
const path = require('path')

const isWin = os.platform().toLowerCase().includes('win32');
let checkerTCResultParser = {};

function findTestCaseFromTestSuite(testSuiteFile, tcName) {
  let ret = common.readJsonFile(testSuiteFile);
  if (ret.code !== errorCode.CE000) {
    return ret;
  }
  const tcConfigs = ret.data;
  for (let i = 0; i < tcConfigs.caselist.length; i += 1) {
    console.log('testcase:', tcConfigs.caselist[i].testCase, ', tcName:', tcName);
    if (tcConfigs.caselist[i].testCase.toString() === tcName) {
      return common.genRespFormat(errorCode.CE000, tcConfigs.caselist[i].testCase);
    }
  }
  console.error('cannot find the testcase ' + tcName + ' in test suite ' + testSuiteFile);
  return common.genRespFormat(errorCode.CE103);
}

function findTestcaseConfigDir(userName, tcName, testSuite) {
  const comCfgDir = common.comCfgDir();
  const userCfgDir = common.userConfigDir(userName);
  let ret;
  try {
    console.log('find common config:', comCfgDir + testSuite + '.json');
    if (fs.existsSync(comCfgDir + testSuite + '.json')) {
      ret = findTestCaseFromTestSuite(comCfgDir + testSuite + '.json', tcName);
      if (ret.code === errorCode.CE000) {
        return common.genRespFormat(errorCode.CE000, comCfgDir);
      }
    }
    console.log('find user config:', userCfgDir + testSuite + '.json');
    if (fs.existsSync(userCfgDir + testSuite + '.json')) {
      ret = findTestCaseFromTestSuite(userCfgDir + testSuite + '.json', tcName);
      if (ret.code === errorCode.CE000) {
        return common.genRespFormat(errorCode.CE000, userCfgDir);
      }
    }
  } catch (e) {
    console.error('find common config failed: ', e);
    return common.genRespFormat(errorCode.CE103);
  }
  return ret;
}

/*
 * 专门为内存冲突和内存语义功能demo展示构造的假数据
 * */
function createFakeData(tcName, userTcDir) {
  const path = common.comCfgDir();
  const tcResultFilePath = path + 'analysis_result_binary_' + tcName + '.txt';
  console.log('寻找用例的结果二进制文件: ', tcResultFilePath);
  if (fs.existsSync(tcResultFilePath)) {
    fsExtra.copySync(tcResultFilePath, userTcDir + '/analysis_result_binary.txt');
    console.log('copy binary result: \n dstPath: ', userTcDir + '/analysis_result_binary.txt', ',\n oriPath: ', tcResultFilePath);
    return common.genRespFormat(errorCode.CE000);
  }
  return common.genRespFormat(errorCode.CE104);
}

/*
 * runTc: 调用HCCL运行用例
 * */
function runTc(userName, tcName, testSuite) {
  console.log('current dir:', process.cwd());
  let ret = findTestcaseConfigDir(userName, tcName, testSuite);
  console.log(ret);
  if (ret.code !== errorCode.CE000) {
    console.error('can not find tc config.');
    return ret;
  }
  const fullTcName = testSuite + '.' + tcName;
  const tcConfigDir = ret.message;
  const userTcDir = common.userWorkDir(userName) + fullTcName;
  console.log('run tc, current dir: ', userTcDir);
  fsExtra.removeSync(userTcDir);
  fsExtra.mkdirsSync(userTcDir);
  // 内存冲突和语义用例，不直接跑用例，仅将proto buffer文件拷贝到用户目录
  ret = createFakeData(tcName, userTcDir);
  if (ret.code === errorCode.CE000) {
    return ret;
  }
  if (!isWin) {
    console.log('exec path:' + process.cwd());
    if (!process.env.HCCL_BIN_PATH) {
      process.env.HCCL_BIN_PATH = process.env.WORKSPACE_DIR + '/' + userName;
      console.log('用户未配置hccl binary路径，采用默认路径为：', process.env.HCCL_BIN_PATH);
    }
  
    const binPath = path.join(process.env.HCCL_BIN_PATH, 'insight_adapter');
    if (!fs.existsSync(binPath)) {
      console.error('用户缺少算法可执行文件，无法执行用例。(' + userTcDir + '../insight_adapter');
      return common.genRespFormat(errorCode.CE021);
    }
    const cmdStr = 'cd ' + userTcDir + ' && ' + binPath + ' -run ' + tcConfigDir + ' ' + fullTcName + ' && cd ../../../';
    console.log('cmdStr: ', cmdStr);
    try {
      process.env.LD_LIBRARY_PATH = process.env.HCCL_BIN_PATH + ":" + process.env.WORKSPACE_DIR + '/common:' + process.env.LD_LIBRARY_PATH;
      console.log('env var LD_LIBRARY_PATH= ', process.env.LD_LIBRARY_PATH);
      const stdout = execSync(cmdStr).toString();
      console.log('stdout:', stdout);
    } catch (err) {
      console.error('tc execute failed:', err);
      return common.genRespFormat(errorCode.CE202);
    }
  }
  return common.genRespFormat(errorCode.CE000);
}

function getPreSetTcConfig(configDir) {
  const preSetList = new Set();
  common.walk(configDir, 1, 2, (filePath) => {
    const tcConfigs = JSON.parse(require('fs').readFileSync(filePath, 'utf8'));
    if (!tcConfigs.caselist) {
      tcConfigs.caselist = [];
    }
    tcConfigs.caselist.forEach((tc) => {
      preSetList.add(tc);
    });
  });
  return preSetList;
}

/*
 * initTcList: 初始化预置用例列表
 * 功能描述：解析预置目录中存放的用例列表配置参数文件，返回给前端。
 * */
async function initTcList(userName) {
  // 预置用例列表
  const preSetTCDir = common.comCfgDir();
  const preSetList = getPreSetTcConfig(preSetTCDir);
  const userTcDir = common.userConfigDir(userName);
  let userTcList = [];
  if (fs.existsSync(userTcDir)) {
    userTcList = getPreSetTcConfig(userTcDir);
  }
  return { code: errorCode.CE000, message: '', data: { defaultTcList: [...preSetList], userTcList: [...userTcList] } };
}

// 不同命令参数初始化
function initCmd(msg) {
  if (msg.cmd === 'start') {
    checkerTCResultParser = new CheckerTCResultParser(msg.userName);
  } else {
    const fullTcName = msg.testSuite + '.' + msg.tcName;
    checkerTCResultParser.initConfig(msg.userName, fullTcName);
  }
}

process.on('message', async (msg) => {
  replaceConsole(msg.userName);
  console.log('Received message', msg);
  const cmd = msg.cmd;
  initCmd(msg);
  if (cmd === 'start') {
    const tcList = initTcList(msg.userName);
    tcList.then((result) => {
      process.send(result);
    });
  } else if (cmd === 'run_tc') {
    console.log('suite:', msg.testSuite, ', tcName:', msg.tcName);
    const fullTcName = msg.testSuite + '.' + msg.tcName;
    if (checkerTCResultParser.isTcResultExist(fullTcName)) {
      // 若用例结果已存在，则直接返回结果
      process.send(checkerTCResultParser.getTcResult(fullTcName));
      return;
    }
    let ret = runTc(msg.userName, msg.tcName, msg.testSuite);
    if (ret.code !== errorCode.CE000) {
      process.send(ret);
      return;
    }
    if (msg.testSuite === 'AllReduceTest' && msg.tcName === 'memConflict') {
      const filePath = common.userWorkDir(msg.userName) + fullTcName + '/memory_conflict.json';
      const data = common.readJsonFile(filePath);
      process.send(data);
      return;
    }
    checkerTCResultParser.parseRun(fullTcName).then((result) => {
      if (result.code !== errorCode.CE000) {
        process.send(result);
        return;
      }
      const tcResult = checkerTCResultParser.getTcResult(fullTcName);
      process.send(tcResult);
    });
  } else if (cmd === 'switch_rank') {
    const rankInfo = checkerTCResultParser.switchRankNodeSemantic(msg.oriRankId, msg.tarRankId, msg.oriLocalStep);
    process.send(rankInfo);
  } else if (cmd === 'get_semantic') {
    const nodeSemantic = checkerTCResultParser.getNodeSemantic(msg.rankId, msg.localStep);
    process.send(nodeSemantic);
  } else if (cmd === 'switch_topo_view') {
    const topoInfo = checkerTCResultParser.switchRankTopoView(msg.rankId, msg.topoView);
    process.send(topoInfo);
  } else {
    process.send(common.genRespFormat(errorCode.CE004));
  }
});

process.on('error', async (err) => {
  console.error('child error:', err);
  process.send(common.genRespFormat(errorCode.CE105), err);
});

process.on('exit', async (msg) => {
  console.log('child exit:', msg);
  process.send(common.genRespFormat(errorCode.CE106, msg));
});

process.on('uncaughtException', async (err) => {
  console.error('child uncaughtException:', err);
  process.send(common.genRespFormat(errorCode.CE107, err));
});
