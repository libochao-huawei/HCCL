const httpStatus = require('http-status');
const errorCodeMap = require('../helper/errorCodeMap');
const errorCode = require('../helper/errorCode');

module.exports = (req, res, next) => {
  res.commonJson = function (data, errCode = errorCode.CE000, errMsg = '', httpStatusCode = httpStatus.OK) {
    const hasError = Boolean(errCode !== errorCode.CE000 || httpStatusCode !== httpStatus.OK);
    const responseBody = {
      data: data || null,
      statusCode: hasError ? 1 : 0
    };
    if (errCode !== errorCode.CE000) {
      responseBody.error = {
        code: errCode,
        message: errMsg || errorCodeMap[errCode] || ''
      };
    }
    this.status(httpStatusCode).json(responseBody);
  };

  next();
};
