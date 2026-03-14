#ifndef HCOMM_DEVICE_PROFILING_DL_H
#define HCOMM_DEVICE_PROFILING_DL_H

#include "hccl_res_dl.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

constexpr u32 MAX_LENGTH = 128;

typedef struct HcomProInfo {
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
}HcomProInfo;

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