const { exec, fork, execSync } = require('child_process');
const os = require('os');
const fsExtra = require('fs-extra');
const common = require('../service/common');
const subProcessList = new Map();
const errorCode = require('../../app/helper/errorCode');
const replaceConsole = require('../../app/lib/logger');
const fs = require('fs');
const path = require('path');
const CheckerTCResultParser = require('./read_proto_data');

class SubTask {
  constructor(userName, result) {
    replaceConsole(userName);
    this.userName = userName;
    this.result = result;
    this.child = fork(path.join(__dirname, 'sub_task.js'));
    this.workDir = common.userWorkDir(userName);
    this.isWin = os.platform().toLowerCase().includes('win32');
    // 数据状态：true代表数据已准备完毕；false代表数据准备中。
    this.status = false;

    this.child.on('message', function (msg) {
      subProcessList.get(userName).setMessage(msg);
      // 19行和20~21行，差异在哪？
      // this.result = msg;
      // console.log('result:', this.result, ', userName:', userName, ', this=', this);
    });
    this.child.on('exit', function (code, signal) {
      console.log(`Child process exited with code ${code} and signal ${signal}`);
    });
    this.child.on('error', function (err) {
      console.error('father get child error:', err);
    });
  }

  resetDataStatus() {
    this.result = { code: errorCode.CE000, data: null, message: '', status: false };
  }

  createWorkSpace() {
    console.log('current dir: ', process.cwd(), ', create dir: ', this.workDir);
    try {
      // 重置工作空间
      this.resetUserWorkSpace();
      if (!this.isWin) {
        const userBinFile = this.workDir + 'insight_adapter';
        if (!fs.existsSync(userBinFile)) {
          console.warn('用户工作空间下' + userBinFile + '文件不存在。计划从取公共工作空间二进制文件运行。');
          const commBinFile = common.comCfgDir() + 'insight_adapter';
          if (!fs.existsSync(commBinFile)) {
            console.warn('公共工作空间下' + commBinFile + '文件不存在');
            return common.genRespFormat(errorCode.CE000);
          } else {
            // 将workspace/common中二进制文件拷贝到用户目录下
            fs.copyFileSync(commBinFile, userBinFile);
          }
        }
        const stdout = execSync('chmod +x ' + userBinFile).toString();
        console.log('stdout:', stdout);
      } else {
        console.log('window系统环境，跳过...');
      }
    } catch (err) {
      console.log('未知错误：', err);
      return common.genRespFormat(errorCode.CE020);
    }
    return common.genRespFormat(errorCode.CE000);
  }

  send(msg) {
    return this.child.send(msg);
  }

  setMessage(msg) {
    this.result = { code: msg.code, data: msg.data, message: msg.message, status: true };
  }

  resetUserWorkSpace() {
    if (!fs.existsSync(this.workDir)) {
      fsExtra.mkdirsSync(this.workDir);
      return;
    }
    const files = fs.readdirSync(this.workDir);
    files.forEach((fileName) => {
      if (fs.statSync(this.workDir + '/' + fileName).isDirectory()) {
        fsExtra.removeSync(this.workDir + '/' + fileName);
      }
    });
  }

  quit() {
    console.log('user ' + this.userName + ' quit, delete work dir : ', this.workDir);
    this.resetUserWorkSpace();
    this.child.kill();
  }
}

function checkUser(cmd, userName) {
  if (subProcessList.size >= 20) {
    console.log('用户数限制');
    // 用户数量限制
    return common.genRespFormat(errorCode.CE001);
  }
  if (cmd === 'register' && subProcessList.has(userName)) {
    console.log(userName + '用户重名');
    // 用户重复登录限制
    return common.genRespFormat(errorCode.CE002);
  }
  if (cmd !== 'register' && !subProcessList.has(userName)) {
    console.log(userName + '用户不存在');
    // 用户进程不存在，或已终止
    return common.genRespFormat(errorCode.CE003);
  }
  return common.genRespFormat(errorCode.CE000, '用户 ' + userName + '合法.');
}

module.exports = {
  subProcessList: subProcessList,
  checkUser: checkUser,
  SubTask: SubTask
};
