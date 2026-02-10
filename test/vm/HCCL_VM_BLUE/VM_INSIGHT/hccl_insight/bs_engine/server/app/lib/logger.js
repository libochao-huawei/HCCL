const log4js = require('log4js');
const config = require('../../config');
const path = require('path');

log4js.configure(config.log4jsConfig);

const defaultLogger = log4js.getLogger('default');
const errorLogger = log4js.getLogger('error');
const consoleLogger = log4js.getLogger('console');

/*
 * 为每个用户配置专属的category，且分配专属日志文件路径
 * */
function getUserConfig(category, logPath, level) {
  let categoryCfg = {
    type: 'console'
  };
  if (category) {
    categoryCfg = {
      type: 'dateFile',
      filename: path.resolve(logPath),
      alwaysIncludePattern: true,
      encoding: 'utf8'
    };
  }
  if (level) {
    categoryCfg.level = level;
  }
  return categoryCfg;
}

// 为不同的用户配置不同的logger
const getLoggerForUser = (userName) => {
  const loggerCategory = `user-${userName}`;

  if (!config.log4jsConfig.appenders[loggerCategory]) {
    const logPath = process.env.WORKSPACE_DIR + `/logs/${userName}/` + loggerCategory + '.log';
    config.log4jsConfig.appenders[loggerCategory] = getUserConfig(loggerCategory, logPath, false);
    config.log4jsConfig.categories[loggerCategory] = {
      appenders: [loggerCategory],
      level: 'all'
    };
    log4js.configure(config.log4jsConfig);
  }
  return log4js.getLogger(loggerCategory);
};

const replaceConsole = (userName) => {
  if (!userName || userName === '') {
    return;
  }
  const userLogger = getLoggerForUser(userName);
  console.debug = consoleLogger.debug.bind(userLogger);
  console.log = defaultLogger.info.bind(userLogger);
  console.warn = defaultLogger.warn.bind(userLogger);
  console.info = defaultLogger.info.bind(userLogger);
  console.trace = defaultLogger.trace.bind(userLogger);
  console.error = errorLogger.error.bind(userLogger);
};

module.exports = replaceConsole;
