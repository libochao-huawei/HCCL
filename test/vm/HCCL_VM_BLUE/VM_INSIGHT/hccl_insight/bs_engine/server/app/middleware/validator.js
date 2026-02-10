const { captureRejectionSymbol } = require('supertest/lib/test');
const errorCode = require('../helper/errorCode');
const validator = (schema) => {
  return async (req, res, next) => {
    try {
      // 异步校验
      const value = await schema.validateAsync(req);
      if (req.body) {
        req.body = { ...value?.body };
      }
      if (req.query) {
        req.query = { ...value?.query };
      }
      next();
    } catch (error) {
      res.commonJson(null, errorCode.CM002, error.message);
    }
  };
};

module.exports = validator;
