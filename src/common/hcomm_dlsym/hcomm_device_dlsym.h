#ifndef HCOMM_DEVICE_DLSYM_H
#define HCOMM_DEVICE_DLSYM_H

#ifdef __cplusplus
extern "C" {
#endif

// 动态库管理接口（大驼峰命名）
int HcommDeviceDlInit(void);
void HcommDeviceDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DEVICE_DLSYM_H