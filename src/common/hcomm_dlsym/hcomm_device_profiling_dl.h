#ifndef HCOMM_DEVICE_PROFILING_DL_H
#define HCOMM_DEVICE_PROFILING_DL_H

#include "hccl_res_dl.h"
#include "alg_param.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcommProfilingReportMainStreamAndFirstTaskPtr)(ThreadHandle);
extern HcclResult (*hcommProfilingReportMainStreamAndLastTaskPtr)(ThreadHandle);
extern HcclResult (*hcommProfilingReportDeviceHcclOpInfoPtr)(HcomProInfo);
extern HcclResult (*hcommProfilingInitPtr)(ThreadHandle*, uint32_t);
extern HcclResult (*hcommProfilingEndPtr)(ThreadHandle*, uint32_t);

// 宏：将原始API名映射为函数指针调用
#define HcommProfilingReportMainStreamAndFirstTask                (*hcommProfilingReportMainStreamAndFirstTaskPtr)
#define HcommProfilingReportMainStreamAndLastTask                 (*hcommProfilingReportMainStreamAndLastTaskPtr)
#define HcommProfilingReportDeviceHcclOpInfo                       (*hcommProfilingReportDeviceHcclOpInfoPtr)
#define HcommProfilingInit                                          (*hcommProfilingInitPtr)
#define HcommProfilingEnd                                           (*hcommProfilingEndPtr)

// 查询函数声明
bool HcommIsSupportHcommProfilingReportMainStreamAndFirstTask(void);
bool HcommIsSupportHcommProfilingReportMainStreamAndLastTask(void);
bool HcommIsSupportHcommProfilingReportDeviceHcclOpInfo(void);
bool HcommIsSupportHcommProfilingInit(void);
bool HcommIsSupportHcommProfilingEnd(void);

// 动态库管理接口
void HcommDeviceProfilingDlInit(void* libHcommHandle);
void HcommDeviceProfilingDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DEVICE_PROFILING_DL_H