const { env } = process;

const envMap = {
  dev: 'dev',
  sit: 'sit',
  uat: 'uat',
  prod: 'prod'
};

const defaultConfig = require('./config.default');

module.exports = {
  ...defaultConfig
};
