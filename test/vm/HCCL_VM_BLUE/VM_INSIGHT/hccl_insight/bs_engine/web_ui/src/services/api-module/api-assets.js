/* eslint-disable */
import service from '../index';
const prefix = '/hccl/demo/';
export default {
  fetchDataByCmd(data) {
    const options = {
      url: `${prefix}checker`,
      type: 'get',
      params: data
    };
    return service.getData(options);
  },
  uploadCase(data) {
    const options = {
      url: `${prefix}upload_tc_config`,
      type: 'post',
      data
    };
    return service.getData(options);
  },
  createCase(data) {
    const options = {
      url: `${prefix}create_new_tc`,
      type: 'post',
      data
    };
    return service.getData(options);
  },
  modifyCase(data) {
    const options = {
      url: `${prefix}modify_tc_config`,
      type: 'post',
      data
    };
    return service.getData(options);
  },
  deleteCase(data) {
    const options = {
      url: `${prefix}delete_tc`,
      type: 'post',
      data
    };
    return service.getData(options);
  }
};
