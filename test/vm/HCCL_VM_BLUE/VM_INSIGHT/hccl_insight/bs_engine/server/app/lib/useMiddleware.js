const path = require('path');
module.exports = (app, middlewares) => {
  const middlewaresPath = path.resolve(__dirname, '../middleware');
  const midModules = middlewares.map((name) => require(path.join(middlewaresPath, name)));
  for (const middleware of midModules) {
    app.use(middleware);
  }
};
