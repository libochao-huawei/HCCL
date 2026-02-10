const httpStatus = require('http-status');
const errorCode = require('./errorCode');
const errorCodeMap = require('./errorCodeMap');

/**
 * @extends Error
 */
class ExtendableError extends Error {
  constructor(message, status, isPublic, code) {
    super(message);
    this.message = message;
    this.name = this.constructor.name;
    this.status = status;
    this.isPublic = isPublic;
    this.code = code;
    this.isOperrational = true;
    Error.captureStackTrace(this, this.constructor.name);
  }
}

/**
 * Class representing an API error.
 * @extends ExtendableError
 */
class APIError extends ExtendableError {
  /**
   * Creates an API error.
   * @param {string} message - Error message.
   * @param {number} status - HTTP status code of error.
   * @param {boolean} isPublic - Whether the message should be visible to user or not.
   */
  constructor(message = errorCodeMap[errorCode.CM001], status = httpStatus.INTERNAL_SERVER_ERROR, isPublic = false, code = errorCode.CM001) {
    super(message, status, isPublic, code);
    this.name = 'APIError';
  }
}

/**
 * Class representing an MySQL error.
 * @extends ExtendableError
 */
class MySQLError extends ExtendableError {
  /**
   * Creates an API error.
   * @param {string} message - Error message.
   * @param {number} status - HTTP status code of error.
   * @param {boolean} isPublic - Whether the message should be visible to user or not.
   */
  constructor(message = errorCodeMap[errorCode.CM008], status = httpStatus.INTERNAL_SERVER_ERROR, isPublic = true, code = errorCode.CM008) {
    super(message, status, isPublic, code);
    this.name = 'MySQLError';
  }
}

/**
 * Class representing an Unauthorized error.
 * @extends ExtendableError
 */
class UnauthorizedError extends ExtendableError {
  /**
   * Creates an API error.
   * @param {string} message - Error message.
   * @param {number} status - HTTP status code of error.
   * @param {boolean} isPublic - Whether the message should be visible to user or not.
   */
  constructor(message = errorCodeMap[errorCode.CM006], status = httpStatus.UNAUTHORIZED, isPublic = true, code = errorCode.CM006) {
    super(message, status, isPublic, code);
    this.name = 'UnauthorizedError';
  }
}

module.exports = {
  APIError,
  MySQLError,
  UnauthorizedError
};
