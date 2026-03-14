#ifndef DTYPE_COMMON_DL_H
#define DTYPE_COMMON_DL_H

#include "dtype_common.h"   // 原始头文件，包含所有 C++ 定义

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hrtGetDeviceTypePtr)(DevType &devType);

// 宏：将原始API名映射为函数指针调用
#define hrtGetDeviceType                (*hrtGetDeviceTypePtr)

// 查询函数声明
bool HcommIsSupportHrtGetDeviceType(void);

// 动态库管理接口
void DtypeCommonDlInit(void* libHcommHandle);
void DtypeCommonDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // DTYPE_COMMON_DL_H