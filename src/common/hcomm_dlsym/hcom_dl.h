#ifndef HCOM_DL_H
#define HCOM_DL_H

#include "hcom.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED ((HcclResult)(-2))
#endif

// 查询函数声明
bool HcommIsSupportHcomGetRankSize(void);
bool HcommIsSupportHcomGetLocalRankSize(void);
bool HcommIsSupportHcomGetRankId(void);
bool HcommIsSupportHcomGetLocalRankId(void);
bool HcommIsSupportHcomGetWorldRankFromGroupRank(void);
bool HcommIsSupportHcomGetGroupRankFromWorldRank(void);
bool HcommIsSupportHcomCreateGroup(void);
bool HcommIsSupportHcomDestroyGroup(void);
bool HcommIsSupportHcomSetGradFusionByIndex(void);
bool HcommIsSupportHcomSetGradFusionBySize(void);
bool HcommIsSupportHcomInitByRankTable(void);
bool HcommIsSupportHcomDestroy(void);
bool HcommIsSupportHcomGetCommHandleByGroup(void);
bool HcommIsSupportGetGroupNameByOpBaseHcom(void);
bool HcommIsSupportHcomCreateComResourceByComm(void);
bool HcommIsSupportHcomTopoInfoRegCallback(void);
bool HcommIsSupportHcomGetandClearOverFlowTasks(void);
bool HcommIsSupportHcomGetWorkflowMode(void);
bool HcommIsSupportHcomSetWorkflowMode(void);
bool HcommIsSupportHcomCalcOpOnline(void);
bool HcommIsSupportHcomCalcOpResOffline(void);
bool HcommIsSupportHcomGetMemType(void);
bool HcommIsSupportHcomGetBandWidthPerNPU(void);
bool HcommIsSupportHcomGetServerNumAndDeviceNumPerServer(void);
bool HcommIsSupportHcomGetSecAddrCopyFlag(void);
bool HcommIsSupportHcomInitByString(void);
bool HcommIsSupportHcomInitByMasterInfo(void);
bool HcommIsSupportHcomCreateCommCCLbuffer(void);
bool HcommIsSupportHcomGetInCCLbuffer(void);
bool HcommIsSupportHcomGetOutCCLbuffer(void);
bool HcommIsSupportHcomSetLaunchKernelMode(void);
bool HcommIsSupportHcomGetAicpuOpStreamNotify(void);
bool HcommIsSupportHcomMc2AiCpuStreamAllocAndGet(void);
bool HcommIsSupportHcomSetDumpDebugMode(void);
bool HcommIsSupportHcomGetAlgorithm(void);
bool HcommIsSupportHcomGetAlgExecParam(void);
bool HcommIsSupportHcomSetAutoTuneMode(void);
bool HcommIsSupportHcomGetDeviceType(void);
bool HcommIsSupportHcomSetProfilingMode(void);
bool HcommIsSupportHcomGetSplitStrategy(void);
bool HcommIsSupportHcomFindGroup(void);
bool HcommIsSupportHcomSelectAlg(void);
bool HcommIsSupportHcomCalcAivCoreNum(void);
bool HcommIsSupportHcomSetWorkspaceResource(void);
bool HcommIsSupportHcomSetGlobalWorkSpace(void);
bool HcommIsSupportHcomSetAivCoreLimit(void);
bool HcommIsSupportHcomReleaseSubComms(void);
bool HcommIsSupportHcomUnloadTask(void);
bool HcommIsSupportHcomClearAivSyncBuf(void);
bool HcommIsSupportHcomSetAttachedStream(void);
bool HcommIsSupportHcomSupportDeterministicOptim(void);
bool HcommIsSupportHcomTbeMemClean(void);
bool HcommIsSupportHcomGetInitStatus(void);
bool HcommIsSupportHcomAllGather(void);
bool HcommIsSupportHcomAllGatherV(void);
bool HcommIsSupportHcomAllReduce(void);
bool HcommIsSupportHcomReduce(void);
bool HcommIsSupportHcomBroadcast(void);
bool HcommIsSupportHcomReduceScatter(void);
bool HcommIsSupportHcomReduceScatterV(void);
bool HcommIsSupportHcomSend(void);
bool HcommIsSupportHcomReceive(void);
bool HcommIsSupportHcomAlltoAllV(void);
bool HcommIsSupportHcomAlltoAllVC(void);
bool HcommIsSupportHcomAllToAll(void);
bool HcommIsSupportHcomGetHcclComm(void);
bool HcommIsSupportHcomGenerateCclOpTag(void);
bool HcommIsSupportHcomGetCommCCLBufferSize(void);
bool HcommIsSupportHcomGetL0TopoTypeEx(void);
bool HcommIsSupportHcomGetRankSizeEx(void);
bool HcommIsSupportHcomInitByFile(void);
bool HcommIsSupportHcomGetWorkspaceSubStreamNum(void);
bool HcommIsSupportHcomGetWorkspaceMemSize(void);

// 动态库管理接口
void HcomDlInit(void* libHcommHandle);
void HcomDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOM_DL_H