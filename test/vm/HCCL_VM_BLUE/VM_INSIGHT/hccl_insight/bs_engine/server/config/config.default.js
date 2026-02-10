const path = require('path');

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
const workspaceDir = initWorkspace();
const log4jsConfig = {
  appenders: {
    console: {
      type: 'console',
      layout: {
        type: 'colored'
      }
    },
    default: {
      type: 'dateFile',
      filename: path.resolve(workspaceDir, './logs/service.log'),
      backups: 3
    },
    normal: {
      type: 'dateFile',
      filename: path.resolve(workspaceDir, './logs/service.log')
    },
    error: {
      type: 'dateFile',
      filename: path.resolve(workspaceDir, './logs/service.log')
    }
  },
  categories: {
    console: { appenders: ['console'], level: 'trace' },
    default: { appenders: ['normal'], level: 'info' },
    error: { appenders: ['error'], level: 'error' }
  }
};

module.exports = {
  jwtSecretKey: 'express template',
  jwtexpired: '30d',
  log4jsConfig
};
