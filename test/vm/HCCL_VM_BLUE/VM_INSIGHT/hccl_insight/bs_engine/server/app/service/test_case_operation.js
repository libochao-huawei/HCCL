const userTaskInfo = require('./user_process');
const fs = require('fs');
const fsExtra = require('fs-extra');
const common = require('./common');
const errorCode = require('../helper/errorCode');
const errCode = require('../helper/errorCode');

function initUserConfigDir(userName) {
  try {
    const configPath = common.userConfigDir(userName);
    if (!fs.existsSync(configPath)) {
      fsExtra.mkdirsSync(configPath);
    }
  } catch (error) {
    console.error('create user workspace config directory failed: ', error);
    return common.genRespFormat(errorCode.CE013);
  }
  return common.genRespFormat(errorCode.CE000);
}

// 检查用户导入用例是否已存在：testSuite.testCase作为名称查重
function checkTcConfigRepeatability(tcGroups, configPath) {
  const preSetList = common.getPreSetTcList(common.comCfgDir());
  for (let [testSuite, tcData] of preSetList.entries()) {
    // 先检查预置用例列表
    if (preSetList.has({ testSuite: tcData.testSuite, testCase: tcData.testCase })) {
      console.error('预置用例列表中已存在该用例：' + tcData.testSuite + '.' + tcData.testCase);
      return common.genRespFormat(errorCode.CE011);
    }
    // 非预置用例列表检查
    const dstPath = configPath + '/' + testSuite + '.json';
    if (!fs.existsSync(dstPath)) {
      continue;
    }
    const userTcList = common.getPreSetTcList(configPath);
    if (userTcList.has({ testSuite: tcData.testSuite, testCase: tcData.testCase })) {
      console.error('用户私有用例列表中已存在该用例：' + tcData.testSuite + '.' + tcData.testCase);
      return common.genRespFormat(errorCode.CE011);
    }
  }
  return common.genRespFormat(errorCode.CE000);
}

function saveUploadTcConfigs(tcGroups, configPath) {
  let ret = checkTcConfigRepeatability(tcGroups, configPath);
  if (ret.code !== errCode.CE000) {
    console.error('checkTcConfigRepeatability failed:', ret);
    return ret;
  }
  try {
    tcGroups.forEach((tcData, testSuite, map) => {
      // 使用testSuite名称作为文件名称（需要做校验）
      const dstPath = configPath + testSuite + '.json';
      console.log('save test case config data to file:', dstPath);
      fs.writeFileSync(dstPath, JSON.stringify({ caselist: tcData }, null, 2), (err) => {
        if (err) {
          console.error('Error writing file', err);
          return common.genRespFormat(errorCode.CE102);
        }
      });
    });
  } catch (err) {
    console.error('save test case config data to file failed: ', err);
    return common.genRespFormat(errorCode.CE013);
  }
  return common.genRespFormat(errorCode.CE000);
}

module.exports = {
  initUserConfigDir,
  saveUploadTcConfigs
};
