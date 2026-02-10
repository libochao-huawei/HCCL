module.exports = (req, res, next) => {
  if (/^\/panshi\/cic\/.*/.test(req.path)) {
    res.header('Access-Control-Allow-Origin', req.headers.origin);
    res.header('Access-Control-Allow-Credentials', true);
    res.header('Access-Control-Allow-Headers', 'X-Requested-With, Accept, Authorization');
    res.header('Access-Control-Allow-Methods', 'DELETE,POST,GET,PUT,PATCH,OPTIONS');
    res.header('Content-Type', 'application/json;charset=utf-8');

    if (req.method === 'OPTIONS') {
      return res.status(200).end();
    }
  }
  next();
};
