#ifndef HCCL_RES_DL_H
#define HCCL_RES_DL_H

#include "hccl_res.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcclGetHcclBufferPtr)(HcclComm, void**, uint64_t*);
extern HcclResult (*hcclGetRemoteIpcHcclBufPtr)(HcclComm, uint64_t, void**, uint64_t*);
extern HcclResult (*hcclThreadAcquirePtr)(HcclComm, CommEngine, uint32_t, uint32_t, ThreadHandle*);
extern HcclResult (*hcclThreadAcquireWithStreamPtr)(HcclComm, CommEngine, aclrtStream, uint32_t, ThreadHandle*);
extern HcclResult (*hcclChannelAcquirePtr)(HcclComm, CommEngine, const HcclChannelDesc*, uint32_t, ChannelHandle*);
extern HcclResult (*hcclChannelGetHcclBufferPtr)(HcclComm, ChannelHandle, void**, uint64_t*);
extern HcclResult (*hcclEngineCtxCreatePtr)(HcclComm, const char*, CommEngine, uint64_t, void**);
extern HcclResult (*hcclEngineCtxGetPtr)(HcclComm, const char*, CommEngine, void**, uint64_t*);
extern HcclResult (*hcclEngineCtxCopyPtr)(HcclComm, CommEngine, const char*, const void*, uint64_t, uint64_t);
extern int32_t    (*hcclTaskRegisterPtr)(HcclComm, const char*, Callback);
extern int32_t    (*hcclTaskUnRegisterPtr)(HcclComm, const char*);
extern HcclResult (*hcclDevMemAcquirePtr)(HcclComm, const char*, uint64_t*, void**, bool*);
extern HcclResult (*hcclThreadExportToCommEnginePtr)(HcclComm, uint32_t, const ThreadHandle*, CommEngine, ThreadHandle*);
extern HcclResult (*hcclChannelGetRemoteMemsPtr)(HcclComm, ChannelHandle, uint32_t*, CommMem**, char***);
extern HcclResult (*hcclCommMemRegPtr)(HcclComm, const char*, const CommMem*, HcclMemHandle*);
extern HcclResult (*hcclEngineCtxDestroyPtr)(HcclComm, const char*, CommEngine);

// 宏：将原始API名映射为函数指针调用（保持API名大驼峰）
#define HcclGetHcclBuffer               (*hcclGetHcclBufferPtr)
#define HcclGetRemoteIpcHcclBuf         (*hcclGetRemoteIpcHcclBufPtr)
#define HcclThreadAcquire                (*hcclThreadAcquirePtr)
#define HcclThreadAcquireWithStream      (*hcclThreadAcquireWithStreamPtr)
#define HcclChannelAcquire                (*hcclChannelAcquirePtr)
#define HcclChannelGetHcclBuffer          (*hcclChannelGetHcclBufferPtr)
#define HcclEngineCtxCreate                (*hcclEngineCtxCreatePtr)
#define HcclEngineCtxGet                   (*hcclEngineCtxGetPtr)
#define HcclEngineCtxCopy                  (*hcclEngineCtxCopyPtr)
#define HcclTaskRegister                   (*hcclTaskRegisterPtr)
#define HcclTaskUnRegister                 (*hcclTaskUnRegisterPtr)
#define HcclDevMemAcquire                  (*hcclDevMemAcquirePtr)
#define HcclThreadExportToCommEngine       (*hcclThreadExportToCommEnginePtr)
#define HcclChannelGetRemoteMems           (*hcclChannelGetRemoteMemsPtr)
#define HcclCommMemReg                     (*hcclCommMemRegPtr)
#define HcclEngineCtxDestroy                (*hcclEngineCtxDestroyPtr)

// 动态库管理接口（大驼峰命名）
void HcclResDlInit(void* libHcommHandle);
void HcclResDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RES_DL_H