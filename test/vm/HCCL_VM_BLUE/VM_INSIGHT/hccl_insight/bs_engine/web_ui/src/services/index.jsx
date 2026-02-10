import apiAssets from './api-module/api-assets';
import { Modal } from '@aurora/vue3';
import axios from 'axios';
let errorModal;
let $service = axios.create({
  baseURL: '/',
  timeout: 50000
});

const getData = ({ url, type, data, params, responseType, headers }) => {
  const method = type.toLowerCase();
  const option = {
    method,
    url,
    responseType,
    headers: {},
    params: null,
    data: null
  };
  option.params = params;
  option.data = data;
  if (headers) {
    option.headers = Object.assign({}, option.headers, headers);
  }
  return new Promise((resolve, reject) => {
    // 解决登录失效未弹出登录框，AUI开发人员提供的临时方案：登录失效全屏登出跳珠到登录页面，登录成功后返回原来的页面;
    $service.interceptors.request.use(
      function (config) {
        //在发送请求之前做某事
        config.handleError = true;
        // 因为打包后报错提示弹框不出现，这里关闭aui自带的错误提示，统一用拦截器处理接口返回不为 200 的错误提示
        config.hideErr = true;
        config.timeout = 28800000;
        return config;
      },
      function (error) {
        //请求错误时做些事
        return Promise.reject(error);
      }
    );
    $service(option)
      .then((response) => {
        if (response.data.code === 'CE000') {
          resolve(response);
        } else {
          return Promise.reject(response);
        }
      })
      .catch((err) => {
        console.log('err', err);
        if (!errorModal) {
          console.log('errorModal', errorModal);
          const message = err?.data?.message;
          const style = { padding: '2px 0' };
          errorModal = Modal.alert({
            message: () =>
              message ? (
                typeof message === 'string' ? (
                  message
                ) : (
                  <ul>
                    <li style={style}>code：{message.code}</li>
                    <li style={style}>errno：{message.errno}</li>
                    <li style={style}>path：{message.path}</li>
                    <li style={style}>syscall：{message.syscall}</li>
                  </ul>
                )
              ) : (
                '非常抱歉，系统处理您的请求时发生了未知错误！请稍后再试。'
              ),
            status: 'error'
          }).then(() => {
            errorModal = null;
          });
        }
        reject(err);
      });
  });
};

export default {
  install: (app) => {
    app.provide('$api', {
      apiAssets
    });
  },
  getData
};
