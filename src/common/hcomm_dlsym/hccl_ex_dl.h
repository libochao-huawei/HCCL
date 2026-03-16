#ifndef HCCL_EX_DL_H
#define HCCL_EX_DL_H

#include "hccl_ex.h"   // 原始头文件，包含所有声明和类型定义

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcclCreateComResourcePtr)(const char*, u32, void**);
extern HcclResult (*hcclGetAicpuOpStreamNotifyPtr)(const char*, rtStream_t*, void**);
extern HcclResult (*hcclAllocComResourcePtr)(HcclComm, u32, void**);
extern HcclResult (*hcclAllocComResourceByTilingPtr)(HcclComm, void*, void*, void**);
extern HcclResult (*hcclGetAicpuOpStreamAndNotifyPtr)(HcclComm, rtStream_t*, u8, void**);
extern HcclResult (*hcclGetTopoDescPtr)(HcclComm, HcclTopoDescs*, uint32_t);
extern HcclResult (*hcclCommRegisterPtr)(HcclComm, void*, uint64_t, void**, uint32_t);
extern HcclResult (*hcclCommDeregisterPtr)(HcclComm, void*);
extern HcclResult (*hcclCommExchangeMemPtr)(HcclComm, void*, uint32_t*, uint32_t);

// 宏：将原始API名映射为函数指针调用
#define HcclCreateComResource                (*hcclCreateComResourcePtr)
#define HcclGetAicpuOpStreamNotify           (*hcclGetAicpuOpStreamNotifyPtr)
#define HcclAllocComResource                 (*hcclAllocComResourcePtr)
#define HcclAllocComResourceByTiling         (*hcclAllocComResourceByTilingPtr)
#define HcclGetAicpuOpStreamAndNotify        (*hcclGetAicpuOpStreamAndNotifyPtr)
#define HcclGetTopoDesc                       (*hcclGetTopoDescPtr)
#define HcclCommRegister                       (*hcclCommRegisterPtr)
#define HcclCommDeregister                     (*hcclCommDeregisterPtr)
#define HcclCommExchangeMem                    (*hcclCommExchangeMemPtr)

// 查询函数声明
bool HcommIsSupportHcclCreateComResource(void);
bool HcommIsSupportHcclGetAicpuOpStreamNotify(void);
bool HcommIsSupportHcclAllocComResource(void);
bool HcommIsSupportHcclAllocComResourceByTiling(void);
bool HcommIsSupportHcclGetAicpuOpStreamAndNotify(void);
bool HcommIsSupportHcclGetTopoDesc(void);
bool HcommIsSupportHcclCommRegister(void);
bool HcommIsSupportHcclCommDeregister(void);
bool HcommIsSupportHcclCommExchangeMem(void);

// 动态库管理接口
void HcclExDlInit(void* libHcommHandle);
void HcclExDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_EX_DL_H