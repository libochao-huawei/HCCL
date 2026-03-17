#ifndef HCCL_EX_DL_H
#define HCCL_EX_DL_H

#include "hccl_ex.h"   // 原始头文件

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 对外 API 的包装函数声明
HcclResult HcclCreateComResource(const char* commName, u32 streamMode, void** commContext);
HcclResult HcclGetAicpuOpStreamNotify(const char* commName, rtStream_t* Opstream, void** aicpuNotify);
HcclResult HcclAllocComResource(HcclComm comm, u32 streamMode, void** commContext);
HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* Mc2Tiling, void** commContext);
HcclResult HcclGetAicpuOpStreamAndNotify(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify);
HcclResult HcclGetTopoDesc(HcclComm comm, HcclTopoDescs *topoDescs, uint32_t topoSize);
HcclResult HcclCommRegister(HcclComm comm, void* addr, uint64_t size, void **handle, uint32_t flag);
HcclResult HcclCommDeregister(HcclComm comm, void* handle);
HcclResult HcclCommExchangeMem(HcclComm comm, void* windowHandle, uint32_t* peerRanks, uint32_t peerRankNum);

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