const { expressjwt } = require('express-jwt');
const config = require('../../config');
module.exports = expressjwt({
  secret: config.jwtSecretKey,
  algorithms: ['HS256']
}).unless({
  path: [/login$/]
});
