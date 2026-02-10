const AdsConfigCenter = require('@panshi/ads-config-center').default;
const config = require('./config');
const dotenv = require('dotenv');
const path = require('path');
dotenv.config({ path: path.resolve(__dirname, './.env'), override: false });
const Automate = require('sequelize-automate');

async function initModel() {
  const dataSourceCredential = await AdsConfigCenter.getDataSourceCredential();
  const username = dataSourceCredential[0];
  const password = dataSourceCredential[1];
  const dbOptions = {
    database: config.database,
    username: username,
    password: password,
    dialect: 'mysql',
    host: config.host,
    port: config.port,
    logging: false
  };
  const options = {
    type: 'js',
    dir: 'app/models'
  };
  const automate = new Automate(dbOptions, options);
  await automate.run();
}

initModel().then(() => {
  console.log('Init Model finished');
});
