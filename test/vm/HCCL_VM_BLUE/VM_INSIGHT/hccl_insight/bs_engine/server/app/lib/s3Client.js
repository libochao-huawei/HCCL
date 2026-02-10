const Lock = require('async-lock');
// 引入obs库
// 使用npm安装
const ObsClient = require('esdk-obs-nodejs');
const config = require('../../config');
const AdsConfigCenter = require('@panshi/ads-config-center').default;
const { APIError } = require('../helper/AppError');

const lock = new Lock();

class S3ObsClient {
  constructor() {
    this.bucketName = config.bucketName;
    this.getObsClient().then((s3c) => console.log('S3初始化成功'));
  }

  async getObsClient() {
    const s3key = await AdsConfigCenter.gets3Credential();
    if (!this.s3ObsClient) {
      await lock.acquire('singleS3Lock', async () => {
        if (!this.s3ObsClient) {
          this.s3ObsClient = new ObsClient({
            access_key_id: s3key[0],
            secret_access_key: s3key[1],
            server: config.s3Service,
            path_style: true
          });
        }
      });
    }
    return this.s3ObsClient;
  }

  /**
   * 获取文件下载链接
   *
   * @param path 文件路径
   * @param options 额外配置
   * @return 下载链接
   */
  async getSingedDownloadUrl(path, options) {
    const s3ObsClient = await this.getObsClient();
    const res = s3ObsClient.createSignedUrlSync({
      Method: 'GET',
      Bucket: this.bucketName,
      Key: path,
      Expires: 600,
      ...options
    });
    return res.SignedUrl;
  }

  /**
   * 上传文件
   *
   * @param key 文件key
   * @param options 文件路径或文件流
   */
  async putObject(key, options) {
    const s3ObsClient = await this.getObsClient();
    return new Promise((resolve, reject) => {
      s3ObsClient
        .putObject({
          Bucket: this.bucketName,
          Key: key,
          ...options
        })
        .then((result) => {
          let success = false;
          if (result.CommonMsg.Status < 300) {
            if (result.InterfaceResult) {
              success = true;
              console.log('ObsService Operation Succeed');
            }
            resolve({ success, result });
          } else {
            console.log('ObsService Code-->' + result.CommonMsg.Code);
            console.log('ObsService Message-->' + result.CommonMsg.Message);
            console.log('ObsService HostId-->' + result.CommonMsg.HostId);
            console.log('ObsService RequestId-->' + result.CommonMsg.RequestId);
            reject(new APIError(`文件上传OBS异常：${result.CommonMsg.Message}`));
          }
        })
        .catch((err) => {
          console.error('Error-->' + err);
          reject(new APIError(`文件上传OBS异常：${err}`));
        });
    });
  }

  async getObject(key, options) {
    const s3ObsClient = await this.getObsClient();
    return new Promise((resolve, reject) => {
      s3ObsClient.getObject(
        {
          Bucket: this.bucketName,
          Key: key,
          // 使用 SaveAsStream : true 指定使用流式下载。
          // 使用 SaveAsFile 参数指定文件下载的路径。
          ...options
        },
        (err, result) => {
          if (err) {
            console.error('S3Obs downloadObject Error-->' + err);
            reject(new APIError(`下载OBS文件异常：${err}`));
          } else {
            if (result.CommonMsg.Status < 300) {
              resolve(result);
            } else {
              console.error('S3Obs downloadObject Error-->' + result.CommonMsg.Message);
              reject(new APIError(`下载OBS文件异常：${result.CommonMsg.Message}`));
            }
          }
        }
      );
    });
  }

  /**
   * 删除文件 （对象）
   *
   * @param {Array<Object>} objects 要删除的对象列表，例：[{Key: 'objectname1'}, {Key: 'objectname2'}, {Key : 'objectname3'}]
   * @returns {Promise<*>}
   */
  async deleteObjects(objects) {
    const s3ObsClient = await this.getObsClient();
    return new Promise((resolve, reject) => {
      s3ObsClient.deleteObjects(
        {
          Bucket: this.bucketName,
          Quiet: false,
          Objects: objects
        },
        (err, result) => {
          if (err) {
            console.error('S3Obs deleteObjects Error-->' + err);
            reject(new APIError(`删除OBS文件异常：${err}`));
          } else {
            console.log('S3Obs deleteObjects Status-->' + result.CommonMsg.Status);
            if (result.CommonMsg.Status < 300 && result.InterfaceResult) {
              // 获取删除成功的对象
              console.log('S3Obs deleteObjects Deleteds:');
              for (let i = 0; i < result.InterfaceResult.Deleteds.length; i++) {
                console.log('Deleted[' + i + ']:');
                console.log('Key-->' + result.InterfaceResult.Deleteds[i].Key);
                console.log('VersionId-->' + result.InterfaceResult.Deleteds[i].VersionId);
              }
              // 获取删除失败的对象
              console.log('S3Obs deleteObjects Errors:');
              for (let i = 0; i < result.InterfaceResult.Errors.length; i++) {
                console.log('Error[' + i + ']:');
                console.log('Key-->' + result.InterfaceResult.Errors[i].Key);
                console.log('VersionId-->' + result.InterfaceResult.Errors[i].VersionId);
              }
              resolve(result);
            } else {
              console.error('S3Obs deleteObjects Error-->' + result.CommonMsg.Message);
              reject(new APIError(`删除OBS文件异常：${result.CommonMsg.Message}`));
            }
          }
        }
      );
    });
  }
}

const s3ObsClient = new S3ObsClient();

module.exports = {
  s3ObsClient
};
