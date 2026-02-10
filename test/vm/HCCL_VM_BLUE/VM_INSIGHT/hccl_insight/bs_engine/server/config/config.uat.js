module.exports = {
  database: 'cic',
  host: '7.203.235.60',
  port: 3306,
  foundation: 'http://haekwe-uat10.huawei.com/panshi/foundation',
  common: 'http://haekwe-uat10.huawei.com/panshi/common',
  w3Service: 'http://w3-beta.huawei.com',
  efsService: 'http://efs-beta.huawei.com',
  bpmUrlPrefix: 'http://kweuat.huawei.com/celonbpm/gateway/com.huawei.hic.celon.bpm:runtimepguat2/celonbpm/runtimepg2/services',
  s3Service: 'https://s3-kp-kwe.his-beta.huawei.com',
  bucketName: 'panshi-common',
  callbackUrl: { Process_Data_import: 'http://haekwe-uat10.huawei.com/panshi/common/process/review/process' }
};
