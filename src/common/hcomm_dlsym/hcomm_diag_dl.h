#ifndef HCOMM_DIAG_DL_H
#define HCOMM_DIAG_DL_H

#include "hcomm_diag.h"   // 原始头文件，包含所有声明和类型定义

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcommRegOpInfoPtr)(const char*, void*, size_t);
extern HcclResult (*hcommRegOpTaskExceptionPtr)(const char*, HcommGetOpInfoCallback);

// 宏：将原始API名映射为函数指针调用
#define HcommRegOpInfo                (*hcommRegOpInfoPtr)
#define HcommRegOpTaskException        (*hcommRegOpTaskExceptionPtr)

// 查询函数声明
bool HcommIsSupportHcommRegOpInfo(void);
bool HcommIsSupportHcommRegOpTaskException(void);

// 动态库管理接口
void HcommDiagDlInit(void* libHcommHandle);
void HcommDiagDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DIAG_DL_H