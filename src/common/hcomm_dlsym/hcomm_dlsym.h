#ifndef HCOMM_DLSYM_H
#define HCOMM_DLSYM_H

#ifdef __cplusplus
extern "C" {
#endif

// 动态库管理接口（大驼峰命名）
int HcommDlInit(void);
void HcommDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DLSYM_H