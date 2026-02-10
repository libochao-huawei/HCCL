const errorCode = require('../helper/errorCode');
const config = require('../../config');
const axios = require('axios');
const httpStatus = require('http-status');

const getUserInfo = async (req, res, next) => {
  const url = `${config.foundation}/services/v1/current/user`;
  const userResponse = await axios.get(url, {
    headers: {
      Cookie: req.headers.cookie
    }
  });
  if (userResponse.status !== httpStatus.OK || !userResponse?.data?.data) {
    res.commonJson(null, errorCode.CM004, '', httpStatus.UNAUTHORIZED);
    return;
  }
  req.userInfo = userResponse.data.data;
  next();
};

module.exports = getUserInfo;
