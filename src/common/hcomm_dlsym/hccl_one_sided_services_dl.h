#ifndef HCCL_ONE_SIDED_SERVICES_DL_H
#define HCCL_ONE_SIDED_SERVICES_DL_H

#include "hccl_one_sided_services.h"   // 原始头文件，包含所有声明和类型定义

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcclRegisterMemPtr)(HcclComm, u32, int, void*, u64, HcclMemDesc*);
extern HcclResult (*hcclDeregisterMemPtr)(HcclComm, HcclMemDesc*);
extern HcclResult (*hcclExchangeMemDescPtr)(HcclComm, u32, HcclMemDescs*, int, HcclMemDescs*, u32*);
extern HcclResult (*hcclEnableMemAccessPtr)(HcclComm, HcclMemDesc*, HcclMem*);
extern HcclResult (*hcclDisableMemAccessPtr)(HcclComm, HcclMemDesc*);
extern HcclResult (*hcclBatchPutPtr)(HcclComm, u32, HcclOneSideOpDesc*, u32, rtStream_t);
extern HcclResult (*hcclBatchGetPtr)(HcclComm, u32, HcclOneSideOpDesc*, u32, rtStream_t);
extern HcclResult (*hcclRemapRegistedMemoryPtr)(HcclComm*, HcclMem*, u64, u64);
extern HcclResult (*hcclRegisterGlobalMemPtr)(const HcclMem*, void**);
extern HcclResult (*hcclDeregisterGlobalMemPtr)(void*);
extern HcclResult (*hcclCommBindMemPtr)(HcclComm, void*);
extern HcclResult (*hcclCommUnbindMemPtr)(HcclComm, void*);
extern HcclResult (*hcclCommPreparePtr)(HcclComm, const HcclPrepareConfig*, const int);

// 宏：将原始API名映射为函数指针调用
#define HcclRegisterMem                (*hcclRegisterMemPtr)
#define HcclDeregisterMem              (*hcclDeregisterMemPtr)
#define HcclExchangeMemDesc            (*hcclExchangeMemDescPtr)
#define HcclEnableMemAccess            (*hcclEnableMemAccessPtr)
#define HcclDisableMemAccess           (*hcclDisableMemAccessPtr)
#define HcclBatchPut                   (*hcclBatchPutPtr)
#define HcclBatchGet                   (*hcclBatchGetPtr)
#define HcclRemapRegistedMemory        (*hcclRemapRegistedMemoryPtr)
#define HcclRegisterGlobalMem           (*hcclRegisterGlobalMemPtr)
#define HcclDeregisterGlobalMem         (*hcclDeregisterGlobalMemPtr)
#define HcclCommBindMem                  (*hcclCommBindMemPtr)
#define HcclCommUnbindMem                (*hcclCommUnbindMemPtr)
#define HcclCommPrepare                  (*hcclCommPreparePtr)

// 查询函数声明
bool HcommIsSupportHcclRegisterMem(void);
bool HcommIsSupportHcclDeregisterMem(void);
bool HcommIsSupportHcclExchangeMemDesc(void);
bool HcommIsSupportHcclEnableMemAccess(void);
bool HcommIsSupportHcclDisableMemAccess(void);
bool HcommIsSupportHcclBatchPut(void);
bool HcommIsSupportHcclBatchGet(void);
bool HcommIsSupportHcclRemapRegistedMemory(void);
bool HcommIsSupportHcclRegisterGlobalMem(void);
bool HcommIsSupportHcclDeregisterGlobalMem(void);
bool HcommIsSupportHcclCommBindMem(void);
bool HcommIsSupportHcclCommUnbindMem(void);
bool HcommIsSupportHcclCommPrepare(void);

// 动态库管理接口
void HcclOneSidedServicesDlInit(void* libHcommHandle);
void HcclOneSidedServicesDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_ONE_SIDED_SERVICES_DL_H