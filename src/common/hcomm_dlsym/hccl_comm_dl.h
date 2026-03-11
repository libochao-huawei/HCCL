#ifndef HCCL_COMM_DL_H
#define HCCL_COMM_DL_H

#include "hccl/hccl_comm.h"   // 原头文件，包含所有类型和 inline 函数
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
// 注意：重复的函数（如 HcclGetRankSize、HcclGetRankId）已在 hccl_rank_graph_dl 中处理，此处不再重复
extern HcclResult (*hcclCommInitClusterInfoPtr)(const char*, uint32_t, HcclComm*);
extern HcclResult (*hcclCommInitClusterInfoConfigPtr)(const char*, uint32_t, HcclCommConfig*, HcclComm*);
extern HcclResult (*hcclCreateSubCommConfigPtr)(HcclComm*, uint32_t, uint32_t*, uint64_t, uint32_t, HcclCommConfig*, HcclComm*);
extern HcclResult (*hcclGetRootInfoPtr)(HcclRootInfo*);
extern HcclResult (*hcclCommInitRootInfoPtr)(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*);
extern HcclResult (*hcclCommInitRootInfoConfigPtr)(uint32_t, const HcclRootInfo*, uint32_t, const HcclCommConfig*, HcclComm*);
extern HcclResult (*hcclSetConfigPtr)(HcclConfig, HcclConfigValue);
extern HcclResult (*hcclGetConfigPtr)(HcclConfig, HcclConfigValue*);
extern HcclResult (*hcclGetCommNamePtr)(HcclComm, char*);
extern HcclResult (*hcclCommGetHandleWithNamePtr)(const char*, HcclComm*);
extern HcclResult (*hcclBarrierPtr)(HcclComm, aclrtStream);
extern HcclResult (*hcclCommDestroyPtr)(HcclComm);
extern HcclResult (*hcclCommInitAllPtr)(uint32_t, int32_t*, HcclComm*);
extern HcclResult (*hcclGetCommAsyncErrorPtr)(HcclComm, HcclResult*);
extern const char* (*hcclGetErrorStringPtr)(HcclResult);
extern uint32_t (*hcclGetCommConfigCapabilityPtr)(void);
extern HcclResult (*hcclCommSuspendPtr)(HcclComm);
extern HcclResult (*hcclCommResumePtr)(HcclComm);
extern HcclResult (*hcclCommSetMemoryRangePtr)(HcclComm, void*, size_t, size_t, uint64_t);
extern HcclResult (*hcclCommUnsetMemoryRangePtr)(HcclComm, void*);
extern HcclResult (*hcclCommActivateCommMemoryPtr)(HcclComm, void*, size_t, size_t, aclrtDrvMemHandle, uint64_t);
extern HcclResult (*hcclCommDeactivateCommMemoryPtr)(HcclComm, void*);
extern HcclResult (*hcclCommWorkingDevNicSetPtr)(HcclComm, uint32_t*, bool*, uint32_t);
extern HcclResult (*hcclGroupStartPtr)(void);
extern HcclResult (*hcclGroupEndPtr)(void);
extern HcclResult (*hcclCommSymWinRegisterPtr)(HcclComm, void*, uint64_t, CommSymWindow*, uint32_t);
extern HcclResult (*hcclCommSymWinDeregisterPtr)(CommSymWindow);
extern HcclResult (*hcclCommSymWinGetPtr)(HcclComm, void*, size_t, CommSymWindow*, size_t*);

// 对外 API 的函数声明（包装函数）
HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm);
HcclResult HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);
HcclResult HcclCreateSubCommConfig(HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig* config, HcclComm* subComm);
HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo);
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);
HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue);
HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue* configValue);
HcclResult HcclGetCommName(HcclComm comm, char* commName);
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm);
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream);
HcclResult HcclCommDestroy(HcclComm comm);
HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms);
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError);
const char* HcclGetErrorString(HcclResult code);
uint32_t HcclGetCommConfigCapability(void);
HcclResult HcclCommSuspend(HcclComm comm);
HcclResult HcclCommResume(HcclComm comm);
HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr);
HcclResult HcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags);
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr);
HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks);
HcclResult HcclGroupStart(void);
HcclResult HcclGroupEnd(void);
HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, CommSymWindow* winHandle, uint32_t flag);
HcclResult HcclCommSymWinDeregister(CommSymWindow winHandle);
HcclResult HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, CommSymWindow* winHandle, size_t* offset);

// 查询函数声明
bool HcommIsSupportHcclCommInitClusterInfo(void);
bool HcommIsSupportHcclCommInitClusterInfoConfig(void);
bool HcommIsSupportHcclCreateSubCommConfig(void);
bool HcommIsSupportHcclGetRootInfo(void);
bool HcommIsSupportHcclCommInitRootInfo(void);
bool HcommIsSupportHcclCommInitRootInfoConfig(void);
bool HcommIsSupportHcclSetConfig(void);
bool HcommIsSupportHcclGetConfig(void);
bool HcommIsSupportHcclGetCommName(void);
bool HcommIsSupportHcclCommGetHandleWithName(void);
bool HcommIsSupportHcclBarrier(void);
bool HcommIsSupportHcclCommDestroy(void);
bool HcommIsSupportHcclCommInitAll(void);
bool HcommIsSupportHcclGetCommAsyncError(void);
bool HcommIsSupportHcclGetErrorString(void);
bool HcommIsSupportHcclGetCommConfigCapability(void);
bool HcommIsSupportHcclCommSuspend(void);
bool HcommIsSupportHcclCommResume(void);
bool HcommIsSupportHcclCommSetMemoryRange(void);
bool HcommIsSupportHcclCommUnsetMemoryRange(void);
bool HcommIsSupportHcclCommActivateCommMemory(void);
bool HcommIsSupportHcclCommDeactivateCommMemory(void);
bool HcommIsSupportHcclCommWorkingDevNicSet(void);
bool HcommIsSupportHcclGroupStart(void);
bool HcommIsSupportHcclGroupEnd(void);
bool HcommIsSupportHcclCommSymWinRegister(void);
bool HcommIsSupportHcclCommSymWinDeregister(void);
bool HcommIsSupportHcclCommSymWinGet(void);

void HcclCommDlInit(void* libHcommHandle);        // 本模块独立初始化
void HcclCommDlFini(void);                         // 本模块独立销毁

#ifdef __cplusplus
}
#endif

#endif // HCCL_COMM_DL_H