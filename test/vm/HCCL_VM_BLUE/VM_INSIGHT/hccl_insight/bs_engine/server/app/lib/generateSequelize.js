const AdsConfigCenter = require('@panshi/ads-config-center').default;
const { Sequelize, ConnectionError } = require('sequelize');
module.exports = (dbname, host) => {
  return new Sequelize(dbname, undefined, undefined, {
    host: host,
    dialect: 'mysql',
    timezone: '+08:00',
    pool: {
      max: 5,
      min: 0,
      idle: 30000
    },
    retry: {
      max: 5, // 最大重试次数
      match: [ConnectionError] // 需要重试的错误类型
    },
    define: {
      timestamps: false
    },
    logging: false,
    hooks: {
      beforeConnect: async function (cfg) {
        const dataSourceCredential = await AdsConfigCenter.getDataSourceCredential();
        cfg.username = dataSourceCredential[0];
        cfg.password = dataSourceCredential[1];
      }
    }
  });
};
