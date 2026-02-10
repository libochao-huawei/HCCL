const path = require('path');
const express = require('express');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
require('express-async-errors');
const routes = require('./app/routes');
const mountMiddleware = require('./app/lib/useMiddleware');
const replaceConsole = require('./app/lib/logger');
const cors = require('./app/middleware/cors');
const addCommon = require('./app/middleware/addCommon');
const catchError = require('./app/middleware/catchError');

const app = express();
// 应用设置
app.set('trust proxy', 'loopback');
app.set('view engine', 'ejs');
// 替换console
replaceConsole();

// 配置中间件
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));
app.use(cookieParser());
// 路由配置
mountMiddleware(app, ['cors', 'addCommon']);
app.use('/hccl/demo', routes);
const history = require('connect-history-api-fallback');
app.use(express.static(path.join(__dirname, 'dist')));
app.use(history());
app.use('/static', express.static('static'));
// if error is not an instanceOf APIError, convert it.
// 中间件加载
mountMiddleware(app, ['catchError']);

module.exports = app;
