const fs = require('fs');
const path = require('path');
const _ = require('lodash');
const protobufjs = require('protobufjs');

// 所有的proto文件
let protoBuf = null;

// 加上protoBuf数据结构模板
function loadProtoDir(filesPath) {
  protoBuf = protobufjs.loadSync(filesPath).nested;
  return protoBuf;
}

function lookup(typeName) {
  if (!_.isString(typeName)) {
    throw new TypeError('typeName must be a string');
  }
  if (!protoBuf) {
    throw new TypeError('Please load proto before lookup');
  }
  return _.get(protoBuf, typeName);
}

function read(protoName, bufData) {
  const model = lookup(protoName);
  if (!model) {
    throw new TypeError(`${protoName} not found, please check it again`);
  }
  return model.decode(bufData);
}

module.exports = {
  read,
  loadProtoDir // 在调用create前，先调用该方法把所有的.proto放到内存中
};
