const express = require('express');
const router = express.Router();
const userTaskInfo = require('../service/user_process.js');
const multer = require('connect-multiparty');
const middle = multer({});
const errorCode = require('../helper/errorCode');
const fs = require('fs');
const common = require('../service/common');
const tcOperator = require('../service/test_case_operation');

process.on('SIGINT', () => {
  setImmediate(() => {
    process.exit(0); // 确保所有事件处理完成后再退出
  });
  console.log('退出hccl insight服务.');
});

// 用户上传用例配置文件
router.post('/upload_tc_config', middle, async function (req, res) {
  console.log('file data:', req.body, ', ', req.files);
  // 检查参数
  let ret = common.configUserLog(req.body.user_name);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  if (!req.files.file_info) {
    return res.send(common.genRespFormat(errorCode.CE019));
  }
  // 检查用户
  if (!userTaskInfo.subProcessList.has(req.body.user_name)) {
    console.error(req.body.user_name + '用户不存在');
    // 用户进程不存在，或已终止
    console.log('upload_tc_config check user: ', req.body.user_name);
    return res.send(common.genRespFormat(errorCode.CE003));
  }
  const configPath = common.userConfigDir(req.body.user_name);
  ret = tcOperator.initUserConfigDir(req.body.user_name);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  ret = common.parseTcConfigsFile(req.files.file_info.path);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  console.log('test case config data:', ret.data);
  return res.send(tcOperator.saveUploadTcConfigs(ret.data, configPath));
});

// 用户新建用例
router.post('/create_new_tc', async function (req, res) {
  console.log('config1:', req.body);
  // 检查参数
  let ret = common.checkTcConfig(req.body.user_name, req.body.config);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  // 检查用户
  if (!userTaskInfo.subProcessList.has(req.body.user_name)) {
    console.error('用户不存在');
    // 用户进程不存在，或已终止
    console.error('create_new_tc check user failed: ', req.body.user_name);
    return res.send(common.genRespFormat(errorCode.CE003));
  }
  try {
    tcOperator.initUserConfigDir(req.body.user_name);
    const tcConfigPath = process.env.WORKSPACE_DIR + '/' + req.body.user_name + '/config/' + req.body.config.testSuite + '.json';
    console.log('create new tc, test suite config file path:', tcConfigPath);
    if (fs.existsSync(tcConfigPath)) {
      const configData = JSON.parse(require('fs').readFileSync(tcConfigPath, 'utf8'));
      let ret = common.checkTestCase(configData.caselist, req.body.config);
      if (ret.code !== errorCode.CE000) {
        return res.send(ret);
      }
      configData.caselist.push(req.body.config);
      fs.writeFileSync(tcConfigPath, JSON.stringify(configData, null, 2));
    } else {
      fs.writeFileSync(tcConfigPath, JSON.stringify({caselist: [req.body.config]}, null, 2));
    }
  } catch (error) {
    console.error('create new tc failed: ', error);
    return res.send(common.genRespFormat(errorCode.CE012));
  }

  return res.send(common.genRespFormat(errorCode.CE000));
});

// 用户修改用例参数
router.post('/modify_tc_config', async function (req, res) {
  console.log('modify config:', req.body);
  // 检查参数
  let ret = common.checkTcConfig(req.body.user_name, req.body.config);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  // 检查用户
  if (!userTaskInfo.subProcessList.has(req.body.user_name)) {
    console.log('用户不存在');
    // 用户进程不存在，或已终止
    console.log('modify_tc_config check user: ', req.body.user_name);
    return res.send(common.genRespFormat(errorCode.CE003));
  }
  return res.send(common.checkAndModifyTestName(req.body.user_name, req.body.config.testSuite, req.body.config.testCase, req.body.config));
});

// 用户删除用例
router.post('/delete_tc', async function (req, res) {
  // 检查参数
  let ret = common.configUserLog(req.body.user_name);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  if (!req.body.tc_name || req.body.tc_name.trim() === '') {
    console.error('wrong testcase name:', req.body.tc_name);
    return res.send(common.genRespFormat(errorCode.CE016));
  }
  console.log('delete tc: ', req.body.tc_name);
  // 检查用户
  if (!userTaskInfo.subProcessList.has(req.body.user_name)) {
    console.log('用户不存在');
    // 用户进程不存在，或已终止
    console.log('delete_tc check user: ', req.body.user_name);
    return res.send(common.genRespFormat(errorCode.CE003));
  }
  ret = common.splitTcName(req.body.tc_name);
  if (ret.code !== errorCode.CE000) {
    return res.send(ret);
  }
  return res.send(common.checkAndModifyTestName(req.body.user_name, ret.data.testSuite, ret.data.testCase, null));
});

const checkerCmdFuncList = new Map([
  ['register', checkerCmdRegister],
  ['start', checkerCmdStart],
  ['get_msg', checkerCmdGetMsg],
  ['run_tc', checkerCmdRunTc],
  ['get_semantic', checkerCmdGetSemantic],
  ['switch_rank', checkerCmdSwitchRank],
  ['switch_topo_view', checkerCmdSwitchTopoView],
  ['quit', checkerCmdQuit]
]);

// 用例列表初始化
router.get('/checker', async function (req, res) {
  let cmd = req.query.cmd;
  let userName = req.query.user_name;
  // 检查参数
  let result = common.checkGetParams(req);
  if (result.code !== errorCode.CE000) {
    console.error('参数检查失败:', result);
    return res.send(result);
  }
  // 检查用户
  if (cmd === 'quit' && !userTaskInfo.subProcessList.has(userName)) {
    console.log('用户已删除');
    return res.send(common.genRespFormat(errorCode.CE000, '用户退出成功.'));
  }
  const checkRes = userTaskInfo.checkUser(cmd, userName);
  if (checkRes.code !== errorCode.CE000) {
    return res.send(checkRes);
  }
  // 执行命令
  console.log('cmd :', cmd, ' userName:', userName);
  let func = checkerCmdFuncList.get(cmd);
  if (func) {
    let subTask = {};
    if (cmd === 'register') {
      subTask = new userTaskInfo.SubTask(userName, '');
    } else {
      subTask = userTaskInfo.subProcessList.get(userName);
      if (cmd !== 'get_msg') { // cmd=get_msg，获取前一个命令的result
        subTask.resetDataStatus();
      }
    }
    return res.send(func(req, subTask));
  } else {
    return res.send(common.genRespFormat(errorCode.CE004));
  }
});

function checkerCmdRegister(req, subTask) {
  let userName = req.query.user_name;
  console.log('注册用户:', userName);
  let ret = subTask.createWorkSpace();
  if (ret.code !== errorCode.CE000) {
    console.error('创建用户工作空间失败.');
    return ret;
  }
  userTaskInfo.subProcessList.set(userName, subTask);
  console.log('用户注册成功，子进程个数: ', userTaskInfo.subProcessList.size);
  return common.genRespFormat(errorCode.CE000, '用户注册成功');
}

function checkerCmdStart(req, subTask) {
  subTask.send({ cmd: req.query.cmd, userName: req.query.user_name });
  console.log('成功获取预置用例集.');
  return common.genRespFormat(errorCode.CE000, '创建用户工作空间成功');
}

function checkerCmdGetMsg(req, subTask) {
  console.log('成功获取子进程结果: ', subTask.result);
  return subTask.result;
}

function checkerCmdRunTc(req, subTask) {
  let cmd = req.query.cmd;
  let userName = req.query.user_name;
  // 解析用例名称
  let ret = common.splitTcName(req.query.tc_name);
  if (ret.code !== errorCode.CE000) {
    return ret;
  }
  console.log('执行用例，testSuite:', ret.data.testSuite, ', testCase:', ret.data.testCase);
  subTask.send({ cmd: cmd, userName: userName, tcName: ret.data.testCase, testSuite: ret.data.testSuite });
  console.log('用例执行成功...');
  return common.genRespFormat(errorCode.CE000, '用户执行成功', null, subTask.status);
}

function checkerCmdGetSemantic(req, subTask) {
  // 解析用例名称
  let ret = common.splitTcName(req.query.tc_name);
  if (ret.code !== errorCode.CE000) {
    return ret;
  }
  subTask.resetDataStatus();
  subTask.send({
    cmd: req.query.cmd,
    userName: req.query.user_name,
    tcName: ret.data.testCase,
    testSuite: ret.data.testSuite,
    rankId: req.query.rank_id,
    localStep: req.query.local_step
  });
  console.log('成功获取节点语义信息');
  return common.genRespFormat(errorCode.CE000, '获取节点语义信息成功', null, subTask.status);
}

function checkerCmdSwitchRank(req, subTask) {
  // 解析用例名称
  let ret = common.splitTcName(req.query.tc_name);
  if (ret.code !== errorCode.CE000) {
    return ret;
  }
  subTask.resetDataStatus();
  subTask.send({
    cmd: req.query.cmd,
    userName: req.query.user_name,
    tcName: ret.data.testCase,
    testSuite: ret.data.testSuite,
    oriRankId: req.query.origin_rank,
    oriLocalStep: req.query.origin_local_step,
    tarRankId: req.query.target_rank
  });
  console.log('切换rank成功');
  return common.genRespFormat(errorCode.CE000, '切换rank成功', null, subTask.status);
}

// 当前rank下切换topo视图：单边/双边拓扑
function checkerCmdSwitchTopoView(req, subTask) {
  // 解析用例名称
  let ret = common.splitTcName(req.query.tc_name);
  if (ret.code !== errorCode.CE000) {
    return ret;
  }
  subTask.resetDataStatus();
  subTask.send({
    cmd: req.query.cmd,
    userName: req.query.user_name,
    tcName: ret.data.testCase,
    testSuite: ret.data.testSuite,
    rankId: req.query.rank_id,
    topoView: req.query.topo_view
  });
  console.log('切换rank成功');
  return common.genRespFormat(errorCode.CE000, '切换rank成功', null, subTask.status);
}

function checkerCmdQuit(req, subTask) {
  let userName = req.query.user_name;
  subTask.quit();
  userTaskInfo.subProcessList.delete(userName);
  console.log('用户' + userName + '注销成功');
  return common.genRespFormat(errorCode.CE000, '用户' + userName + '退出成功', null);
}

module.exports = router.use('/test', router);
