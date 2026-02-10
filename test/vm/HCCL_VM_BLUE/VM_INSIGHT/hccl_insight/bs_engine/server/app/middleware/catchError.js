const httpStatus = require('http-status');
const errorCode = require('../helper/errorCode');
const errorCodeMap = require('../helper/errorCodeMap');

const handleError = (err, req, res, next) => {
  if (err) {
    const status = err?.status || httpStatus.INTERNAL_SERVER_ERROR;
    const code = err?.code || errorCode.CM001;
    const message = err?.message || err || errorCodeMap[code] || '';
    console.error(req.origin, err);
    res?.commonJson(null, code, message, status);
  }
};

module.exports = handleError;
