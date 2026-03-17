#ifndef HCCL_MC2_EX_DL_H
#define HCCL_MC2_EX_DL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 对外 API 包装函数声明
HcclResult HcclGetCommHandleByCtx(void *ctx, void **opHandle);
HcclResult HcclReleaseComm(void* opHandle);
HcclResult HcclGetTaskStatus(void* opHandle, void *status);
HcclResult HcclCheckFinishByStream(void* opHandle);
HcclResult HcclPrintTaskExceptionAllComm(void* opHandle);
HcclResult HcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast);
HcclResult HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr);
HcclResult HcclLaunchOp(void* opHandle, void* data);

// 查询函数声明
bool HcommIsSupportHcclGetCommHandleByCtx(void);
bool HcommIsSupportHcclReleaseComm(void);
bool HcommIsSupportHcclGetTaskStatus(void);
bool HcommIsSupportHcclCheckFinishByStream(void);
bool HcommIsSupportHcclPrintTaskExceptionAllComm(void);
bool HcommIsSupportHcclLaunchCcoreWait(void);
bool HcommIsSupportHcclLaunchCcorePost(void);
bool HcommIsSupportHcclLaunchOp(void);

// 动态库管理接口
void HcclMc2ExDlInit(void* libHcommHandle);
void HcclMc2ExDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_MC2_EX_DL_H