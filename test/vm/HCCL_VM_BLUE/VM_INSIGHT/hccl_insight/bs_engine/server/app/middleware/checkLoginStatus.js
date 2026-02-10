const w3Service = require('../service/v1/w3');
const AppError = require('../helper/AppError');
module.exports = async (req, res, next) => {
  if (req.auth) {
    next();
  } else {
    const userInfo = await w3Service.getCurrentUser(req);
    if (userInfo.uuid) {
      next();
    } else {
      throw new AppError.UnauthorizedError();
    }
  }
};
