#ifndef HCOM_DL_H
#define HCOM_DL_H

#include "hcom.h"   // 原始头文件，包含所有声明和类型定义

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 对外 API 的包装函数声明
HcclResult HcomGetCommCCLBufferSize(const char *group, uint64_t &size);
HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag);
HcclResult HcomGetRankSizeEx(const char *group, uint32_t *rankSize, uint32_t flag);
HcclResult HcomGetCommHandleByGroup(const char *group, void **commHandle);
HcclResult HcomGetRankSize(const char *group, uint32_t *rankSize);

// 查询函数声明
bool HcommIsSupportHcomGetCommCCLBufferSize(void);
bool HcommIsSupportHcomGetL0TopoTypeEx(void);
bool HcommIsSupportHcomGetRankSizeEx(void);
bool HcommIsSupportHcomGetCommHandleByGroup(void);
bool HcommIsSupportHcomGetRankSize(void);

// 动态库管理接口
void HcomDlInit(void* libHcommHandle);
void HcomDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOM_DL_H