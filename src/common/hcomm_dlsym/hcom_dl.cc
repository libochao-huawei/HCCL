#include "log.h"
#include "hcom_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcomGetRankSizePtr)(const char*, u32*) = NULL;
HcclResult (*hcomGetLocalRankSizePtr)(const char*, u32*) = NULL;
HcclResult (*hcomGetRankIdPtr)(const char*, u32*) = NULL;
HcclResult (*hcomGetLocalRankIdPtr)(const char*, u32*) = NULL;
HcclResult (*hcomGetWorldRankFromGroupRankPtr)(const char*, u32, u32*) = NULL;
HcclResult (*hcomGetGroupRankFromWorldRankPtr)(u32, const char*, u32*) = NULL;
HcclResult (*hcomCreateGroupPtr)(const char*, u32, u32*) = NULL;
HcclResult (*hcomDestroyGroupPtr)(const char*) = NULL;
HcclResult (*hcomSetGradFusionByIndexPtr)(const char*, u32, const u32*) = NULL;
HcclResult (*hcomSetGradFusionBySizePtr)(const char*, u32, const float*) = NULL;
HcclResult (*hcomInitByRankTablePtr)(const char*, uint32_t) = NULL;
HcclResult (*hcomDestroyPtr)() = NULL;
HcclResult (*hcomGetCommHandleByGroupPtr)(const char*, HcclComm*) = NULL;
HcclResult (*getGroupNameByOpBaseHcomPtr)(s64, char**) = NULL;
HcclResult (*hcomCreateComResourceByCommPtr)(HcclComm, u32, bool, void**, bool) = NULL;
void (*hcomTopoInfoRegCallbackPtr)(HcclResult (*)(const char *, uint32_t), void (*)(const char *)) = NULL;
HcclResult (*hcomGetandClearOverFlowTasksPtr)(const char*, hccl::HcclDumpInfo**, s32*) = NULL;
HcclWorkflowMode (*hcomGetWorkflowModePtr)() = NULL;
HcclResult (*hcomSetWorkflowModePtr)(HcclWorkflowMode) = NULL;
HcclResult (*hcomCalcOpOnlinePtr)(HcomOpParam*, HcomResResponse*) = NULL;
HcclResult (*hcomCalcOpResOfflinePtr)(HcomOpParam*, HcomResResponse*) = NULL;
HcclResult (*hcomGetMemTypePtr)(const char*, const char*, bool, u32*, bool*, bool, bool) = NULL;
HcclResult (*hcomGetBandWidthPerNPUPtr)(u32, float*) = NULL;
HcclResult (*hcomGetServerNumAndDeviceNumPerServerPtr)(u32*, u32*, u32*) = NULL;
bool (*hcomGetSecAddrCopyFlagPtr)(const char*) = NULL;
HcclResult (*hcomInitByStringPtr)(const char*, const char*, WorkMode, HcomInitConfig*) = NULL;
HcclResult (*hcomInitByMasterInfoPtr)(const char*, const char*, const char*, const char*, const char*, HcomInitConfig*) = NULL;
HcclResult (*hcomCreateCommCCLbufferPtr)(const char*) = NULL;
HcclResult (*hcomGetInCCLbufferPtr)(const char*, void**, u64*) = NULL;
HcclResult (*hcomGetOutCCLbufferPtr)(const char*, void**, u64*) = NULL;
void (*hcomSetLaunchKernelModePtr)(bool) = NULL;
HcclResult (*hcomGetAicpuOpStreamNotifyPtr)(const char*, HcclRtStream*, u8, void**) = NULL;
HcclResult (*hcomMc2AiCpuStreamAllocAndGetPtr)(const char*, u32, rtStream_t*) = NULL;
void (*hcomSetDumpDebugModePtr)(bool) = NULL;
HcclResult (*hcomGetAlgorithmPtr)(u32, char**) = NULL;
HcclResult (*hcomGetAlgExecParamPtr)(const char*, const char*, u64, void*, void*, HcclCMDType, bool, HcclDataType, HcclReduceOp, void**, u64*, u32) = NULL;
void (*hcomSetAutoTuneModePtr)(bool) = NULL;
DevType (*hcomGetDeviceTypePtr)() = NULL;
HcclResult (*hcomSetProfilingModePtr)(HcomProfilingMode, const char*) = NULL;
HcclResult (*hcomGetSplitStrategyPtr)(const char*, const struct model_feature*, u32**, u32*, bool*, GradSplitForceMode, OriginalGraphShapeType) = NULL;
bool (*hcomFindGroupPtr)(const char*) = NULL;
HcclResult (*hcomSelectAlgPtr)(s64, const char*, u64, void*, HcclDataType, HcclReduceOp, HcclCMDType, int32_t, bool&, char*) = NULL;
HcclResult (*hcomCalcAivCoreNumPtr)(const char*, HcclCMDType, u64, void*, HcclDataType, int32_t, char*, u32*) = NULL;
HcclResult (*hcomSetWorkspaceResourcePtr)(const char*, const char*, rtStream_t*, s32, void*, u64) = NULL;
HcclResult (*hcomSetGlobalWorkSpacePtr)(const char*, void**, u32) = NULL;
HcclResult (*hcomSetAivCoreLimitPtr)(const char*, u32) = NULL;
HcclResult (*hcomReleaseSubCommsPtr)() = NULL;
HcclResult (*hcomUnloadTaskPtr)(const char*, const char*) = NULL;
HcclResult (*hcomClearAivSyncBufPtr)(const char*, bool) = NULL;
HcclResult (*hcomSetAttachedStreamPtr)(const char*, u32, const rtStream_t*, s32) = NULL;
HcclResult (*hcomSupportDeterministicOptimPtr)(const char*, bool*) = NULL;
HcclResult (*hcomTbeMemCleanPtr)(int64_t[], int64_t[], uint32_t, rtStream_t, int32_t) = NULL;
HcclResult (*hcomGetInitStatusPtr)(bool*) = NULL;
HcclResult (*hcomAllGatherPtr)(const char*, void*, void*, u64, HcclDataType, const char*, rtStream_t) = NULL;
HcclResult (*hcomAllGatherVPtr)(const char*, const void*, u64, const void*, const void*, const void*, HcclDataType, const char*, rtStream_t) = NULL;
HcclResult (*hcomAllReducePtr)(const char*, void*, void*, u64, HcclDataType, HcclReduceOp, const char*, rtStream_t) = NULL;
HcclResult (*hcomReducePtr)(const char*, void*, void*, u64, HcclDataType, HcclReduceOp, u32, const char*, rtStream_t) = NULL;
HcclResult (*hcomBroadcastPtr)(const char*, void*, u64, HcclDataType, u32, const char*, rtStream_t) = NULL;
HcclResult (*hcomReduceScatterPtr)(const char*, void*, void*, u64, HcclDataType, HcclReduceOp, const char*, rtStream_t) = NULL;
HcclResult (*hcomReduceScatterVPtr)(const char*, void*, const void*, const void*, void*, u64, HcclDataType, HcclReduceOp, const char*, rtStream_t) = NULL;
HcclResult (*hcomSendPtr)(const char*, void*, u64, HcclDataType, u32, u32, const char*, rtStream_t) = NULL;
HcclResult (*hcomReceivePtr)(const char*, void*, u64, HcclDataType, u32, u32, const char*, rtStream_t) = NULL;
HcclResult (*hcomAlltoAllVPtr)(const void*, const void*, const void*, HcclDataType, const void*, const void*, const void*, HcclDataType, const char*, rtStream_t, const char*) = NULL;
HcclResult (*hcomAlltoAllVCPtr)(const void*, const void*, HcclDataType, const void*, HcclDataType, const char*, rtStream_t, const char*) = NULL;
HcclResult (*hcomAllToAllPtr)(const void*, u64, HcclDataType, const void*, u64, HcclDataType, const char*, rtStream_t, const char*) = NULL;
HcclResult (*hcomGetHcclCommPtr)(int64_t, std::string&) = NULL;
HcclResult (*hcomGenerateCclOpTagPtr)(const char*, s64, const char*, char*) = NULL;
HcclResult (*hcomGetCommCCLBufferSizePtr)(const char*, uint64_t&) = NULL;
HcclResult (*hcomGetL0TopoTypeExPtr)(const char*, CommTopo*, uint32_t) = NULL;
HcclResult (*hcomGetRankSizeExPtr)(const char*, uint32_t*, uint32_t) = NULL;

// 支持标志（静态，默认 false）
#define DEFINE_SUPPORT_FLAG(name) static bool g_##name##Supported = false

DEFINE_SUPPORT_FLAG(hcomGetRankSize);
DEFINE_SUPPORT_FLAG(hcomGetLocalRankSize);
DEFINE_SUPPORT_FLAG(hcomGetRankId);
DEFINE_SUPPORT_FLAG(hcomGetLocalRankId);
DEFINE_SUPPORT_FLAG(hcomGetWorldRankFromGroupRank);
DEFINE_SUPPORT_FLAG(hcomGetGroupRankFromWorldRank);
DEFINE_SUPPORT_FLAG(hcomCreateGroup);
DEFINE_SUPPORT_FLAG(hcomDestroyGroup);
DEFINE_SUPPORT_FLAG(hcomSetGradFusionByIndex);
DEFINE_SUPPORT_FLAG(hcomSetGradFusionBySize);
DEFINE_SUPPORT_FLAG(hcomInitByRankTable);
DEFINE_SUPPORT_FLAG(hcomDestroy);
DEFINE_SUPPORT_FLAG(hcomGetCommHandleByGroup);
DEFINE_SUPPORT_FLAG(getGroupNameByOpBaseHcom);
DEFINE_SUPPORT_FLAG(hcomCreateComResourceByComm);
DEFINE_SUPPORT_FLAG(hcomTopoInfoRegCallback);
DEFINE_SUPPORT_FLAG(hcomGetandClearOverFlowTasks);
DEFINE_SUPPORT_FLAG(hcomGetWorkflowMode);
DEFINE_SUPPORT_FLAG(hcomSetWorkflowMode);
DEFINE_SUPPORT_FLAG(hcomCalcOpOnline);
DEFINE_SUPPORT_FLAG(hcomCalcOpResOffline);
DEFINE_SUPPORT_FLAG(hcomGetMemType);
DEFINE_SUPPORT_FLAG(hcomGetBandWidthPerNPU);
DEFINE_SUPPORT_FLAG(hcomGetServerNumAndDeviceNumPerServer);
DEFINE_SUPPORT_FLAG(hcomGetSecAddrCopyFlag);
DEFINE_SUPPORT_FLAG(hcomInitByString);
DEFINE_SUPPORT_FLAG(hcomInitByMasterInfo);
DEFINE_SUPPORT_FLAG(hcomCreateCommCCLbuffer);
DEFINE_SUPPORT_FLAG(hcomGetInCCLbuffer);
DEFINE_SUPPORT_FLAG(hcomGetOutCCLbuffer);
DEFINE_SUPPORT_FLAG(hcomSetLaunchKernelMode);
DEFINE_SUPPORT_FLAG(hcomGetAicpuOpStreamNotify);
DEFINE_SUPPORT_FLAG(hcomMc2AiCpuStreamAllocAndGet);
DEFINE_SUPPORT_FLAG(hcomSetDumpDebugMode);
DEFINE_SUPPORT_FLAG(hcomGetAlgorithm);
DEFINE_SUPPORT_FLAG(hcomGetAlgExecParam);
DEFINE_SUPPORT_FLAG(hcomSetAutoTuneMode);
DEFINE_SUPPORT_FLAG(hcomGetDeviceType);
DEFINE_SUPPORT_FLAG(hcomSetProfilingMode);
DEFINE_SUPPORT_FLAG(hcomGetSplitStrategy);
DEFINE_SUPPORT_FLAG(hcomFindGroup);
DEFINE_SUPPORT_FLAG(hcomSelectAlg);
DEFINE_SUPPORT_FLAG(hcomCalcAivCoreNum);
DEFINE_SUPPORT_FLAG(hcomSetWorkspaceResource);
DEFINE_SUPPORT_FLAG(hcomSetGlobalWorkSpace);
DEFINE_SUPPORT_FLAG(hcomSetAivCoreLimit);
DEFINE_SUPPORT_FLAG(hcomReleaseSubComms);
DEFINE_SUPPORT_FLAG(hcomUnloadTask);
DEFINE_SUPPORT_FLAG(hcomClearAivSyncBuf);
DEFINE_SUPPORT_FLAG(hcomSetAttachedStream);
DEFINE_SUPPORT_FLAG(hcomSupportDeterministicOptim);
DEFINE_SUPPORT_FLAG(hcomTbeMemClean);
DEFINE_SUPPORT_FLAG(hcomGetInitStatus);
DEFINE_SUPPORT_FLAG(hcomAllGather);
DEFINE_SUPPORT_FLAG(hcomAllGatherV);
DEFINE_SUPPORT_FLAG(hcomAllReduce);
DEFINE_SUPPORT_FLAG(hcomReduce);
DEFINE_SUPPORT_FLAG(hcomBroadcast);
DEFINE_SUPPORT_FLAG(hcomReduceScatter);
DEFINE_SUPPORT_FLAG(hcomReduceScatterV);
DEFINE_SUPPORT_FLAG(hcomSend);
DEFINE_SUPPORT_FLAG(hcomReceive);
DEFINE_SUPPORT_FLAG(hcomAlltoAllV);
DEFINE_SUPPORT_FLAG(hcomAlltoAllVC);
DEFINE_SUPPORT_FLAG(hcomAllToAll);
DEFINE_SUPPORT_FLAG(hcomGetHcclComm);
DEFINE_SUPPORT_FLAG(hcomGenerateCclOpTag);
DEFINE_SUPPORT_FLAG(hcomGetCommCCLBufferSize);
DEFINE_SUPPORT_FLAG(hcomGetL0TopoTypeEx);
DEFINE_SUPPORT_FLAG(hcomGetRankSizeEx);

// ---------- 桩函数定义 ----------
static HcclResult StubHcomGetRankSize(const char* group, u32* rankSize) {
    (void)group; (void)rankSize; HCCL_ERROR("[HcclWrapper] HcomGetRankSize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetLocalRankSize(const char* group, u32* localRankSize) {
    (void)group; (void)localRankSize; HCCL_ERROR("[HcclWrapper] HcomGetLocalRankSize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetRankId(const char* group, u32* rankId) {
    (void)group; (void)rankId; HCCL_ERROR("[HcclWrapper] HcomGetRankId not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetLocalRankId(const char* group, u32* localRankId) {
    (void)group; (void)localRankId; HCCL_ERROR("[HcclWrapper] HcomGetLocalRankId not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetWorldRankFromGroupRank(const char* group, u32 groupRank, u32* worldRank) {
    (void)group; (void)groupRank; (void)worldRank; HCCL_ERROR("[HcclWrapper] HcomGetWorldRankFromGroupRank not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetGroupRankFromWorldRank(u32 worldRank, const char* group, u32* groupRank) {
    (void)worldRank; (void)group; (void)groupRank; HCCL_ERROR("[HcclWrapper] HcomGetGroupRankFromWorldRank not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCreateGroup(const char* group, u32 rankNum, u32* rankIds) {
    (void)group; (void)rankNum; (void)rankIds; HCCL_ERROR("[HcclWrapper] HcomCreateGroup not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomDestroyGroup(const char* group) {
    (void)group; HCCL_ERROR("[HcclWrapper] HcomDestroyGroup not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* inputIdxList) {
    (void)group; (void)segmentNum; (void)inputIdxList; HCCL_ERROR("[HcclWrapper] HcomSetGradFusionByIndex not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList) {
    (void)group; (void)segmentNum; (void)sizeList; HCCL_ERROR("[HcclWrapper] HcomSetGradFusionBySize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomInitByRankTable(const char* rankTable, uint32_t rankId) {
    (void)rankTable; (void)rankId; HCCL_ERROR("[HcclWrapper] HcomInitByRankTable not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomDestroy() {
    HCCL_ERROR("[HcclWrapper] HcomDestroy not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetCommHandleByGroup(const char* group, HcclComm* commHandle) {
    (void)group; (void)commHandle; HCCL_ERROR("[HcclWrapper] HcomGetCommHandleByGroup not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubGetGroupNameByOpBaseHcom(s64 opBaseHcom, char** groupname) {
    (void)opBaseHcom; (void)groupname; HCCL_ERROR("[HcclWrapper] GetGroupNameByOpBaseHcom not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2) {
    (void)comm; (void)streamMode; (void)isOpbaseMode; (void)commContext; (void)isMC2; HCCL_ERROR("[HcclWrapper] HcomCreateComResourceByComm not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomTopoInfoRegCallback(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *)) {
    (void)p1; (void)p2; HCCL_ERROR("[HcclWrapper] HcomTopoInfoRegCallback not supported");
}
static HcclResult StubHcomGetandClearOverFlowTasks(const char* group, hccl::HcclDumpInfo** hcclDumpInfoPtr, s32* len) {
    (void)group; (void)hcclDumpInfoPtr; (void)len; HCCL_ERROR("[HcclWrapper] HcomGetandClearOverFlowTasks not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclWorkflowMode StubHcomGetWorkflowMode() {
    HCCL_ERROR("[HcclWrapper] HcomGetWorkflowMode not supported"); return HcclWorkflowMode::HCCL_MODE_NORMAL;
}
static HcclResult StubHcomSetWorkflowMode(HcclWorkflowMode mode) {
    (void)mode; HCCL_ERROR("[HcclWrapper] HcomSetWorkflowMode not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCalcOpOnline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    (void)hcomOpParam; (void)hcomResResponse; HCCL_ERROR("[HcclWrapper] HcomCalcOpOnline not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCalcOpResOffline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    (void)hcomOpParam; (void)hcomResResponse; HCCL_ERROR("[HcclWrapper] HcomCalcOpResOffline not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetMemType(const char* group, const char* socVersion, bool isMalloc, u32* memType, bool* isTsMem, bool withoutImplCompile, bool level2Address) {
    (void)group; (void)socVersion; (void)isMalloc; (void)memType; (void)isTsMem; (void)withoutImplCompile; (void)level2Address; HCCL_ERROR("[HcclWrapper] HcomGetMemType not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetBandWidthPerNPU(u32 level, float* bandWidth) {
    (void)level; (void)bandWidth; HCCL_ERROR("[HcclWrapper] HcomGetBandWidthPerNPU not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetServerNumAndDeviceNumPerServer(u32* serverNum, u32* deviceNumPerServer, u32* deviceNumPerAggregation) {
    (void)serverNum; (void)deviceNumPerServer; (void)deviceNumPerAggregation; HCCL_ERROR("[HcclWrapper] HcomGetServerNumAndDeviceNumPerServer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static bool StubHcomGetSecAddrCopyFlag(const char* socVersion) {
    (void)socVersion; HCCL_ERROR("[HcclWrapper] HcomGetSecAddrCopyFlag not supported"); return false;
}
static HcclResult StubHcomInitByString(const char* rankTableM, const char* identify, WorkMode commWorkMode, HcomInitConfig* initConfig) {
    (void)rankTableM; (void)identify; (void)commWorkMode; (void)initConfig; HCCL_ERROR("[HcclWrapper] HcomInitByString not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomInitByMasterInfo(const char* masterIp, const char* masterPort, const char* masterDeviceId, const char* rankSize, const char* rankIp, HcomInitConfig* initConfig) {
    (void)masterIp; (void)masterPort; (void)masterDeviceId; (void)rankSize; (void)rankIp; (void)initConfig; HCCL_ERROR("[HcclWrapper] HcomInitByMasterInfo not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCreateCommCCLbuffer(const char* group) {
    (void)group; HCCL_ERROR("[HcclWrapper] HcomCreateCommCCLbuffer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetInCCLbuffer(const char* group, void** buffer, u64* size) {
    (void)group; (void)buffer; (void)size; HCCL_ERROR("[HcclWrapper] HcomGetInCCLbuffer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetOutCCLbuffer(const char* group, void** buffer, u64* size) {
    (void)group; (void)buffer; (void)size; HCCL_ERROR("[HcclWrapper] HcomGetOutCCLbuffer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomSetLaunchKernelMode(bool state) {
    (void)state; HCCL_ERROR("[HcclWrapper] HcomSetLaunchKernelMode not supported");
}
static HcclResult StubHcomGetAicpuOpStreamNotify(const char* group, HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify) {
    (void)group; (void)opStream; (void)aicpuNotifyNum; (void)aicpuNotify; HCCL_ERROR("[HcclWrapper] HcomGetAicpuOpStreamNotify not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomMc2AiCpuStreamAllocAndGet(const char* group, u32 streamMode, rtStream_t* aiCpuStream) {
    (void)group; (void)streamMode; (void)aiCpuStream; HCCL_ERROR("[HcclWrapper] HcomMc2AiCpuStreamAllocAndGet not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomSetDumpDebugMode(bool dumpDebug) {
    (void)dumpDebug; HCCL_ERROR("[HcclWrapper] HcomSetDumpDebugMode not supported");
}
static HcclResult StubHcomGetAlgorithm(u32 level, char** algo) {
    (void)level; (void)algo; HCCL_ERROR("[HcclWrapper] HcomGetAlgorithm not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetAlgExecParam(const char* tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType, bool clearEnable, HcclDataType dataType, HcclReduceOp op, void** commContext, u64* len, u32 aivCoreLimit) {
    (void)tag; (void)group; (void)count; (void)inputPtr; (void)outputPtr; (void)opType; (void)clearEnable; (void)dataType; (void)op; (void)commContext; (void)len; (void)aivCoreLimit; HCCL_ERROR("[HcclWrapper] HcomGetAlgExecParam not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomSetAutoTuneMode(bool autoTuneMode) {
    (void)autoTuneMode; HCCL_ERROR("[HcclWrapper] HcomSetAutoTuneMode not supported");
}
static DevType StubHcomGetDeviceType() {
    HCCL_ERROR("[HcclWrapper] HcomGetDeviceType not supported"); return DevType::DEV_TYPE_910;
}
static HcclResult StubHcomSetProfilingMode(HcomProfilingMode profilingMode, const char* profilingOption) {
    (void)profilingMode; (void)profilingOption; HCCL_ERROR("[HcclWrapper] HcomSetProfilingMode not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetSplitStrategy(const char* group, const struct model_feature* feature, u32** segmentIdxPtr, u32* len, bool* configured, GradSplitForceMode force, OriginalGraphShapeType shapeType) {
    (void)group; (void)feature; (void)segmentIdxPtr; (void)len; (void)configured; (void)force; (void)shapeType; HCCL_ERROR("[HcclWrapper] HcomGetSplitStrategy not supported"); return HCCL_E_NOT_SUPPORTED;
}
static bool StubHcomFindGroup(const char* group) {
    (void)group; HCCL_ERROR("[HcclWrapper] HcomFindGroup not supported"); return false;
}
static HcclResult StubHcomSelectAlg(s64 comm, const char* group, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, int32_t aivCoreLimit, bool& ifAiv, char* algName) {
    (void)comm; (void)group; (void)count; (void)counts; (void)dataType; (void)op; (void)opType; (void)aivCoreLimit; (void)ifAiv; (void)algName; HCCL_ERROR("[HcclWrapper] HcomSelectAlg not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCalcAivCoreNum(const char* group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, char* algName, u32* numBlocks) {
    (void)group; (void)opType; (void)count; (void)counts; (void)dataType; (void)aivCoreLimit; (void)algName; (void)numBlocks; HCCL_ERROR("[HcclWrapper] HcomCalcAivCoreNum not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetWorkspaceResource(const char* tag, const char* group, rtStream_t* stream, s32 len, void* memPtr, u64 maxSize) {
    (void)tag; (void)group; (void)stream; (void)len; (void)memPtr; (void)maxSize; HCCL_ERROR("[HcclWrapper] HcomSetWorkspaceResource not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetGlobalWorkSpace(const char* group, void** globalWorkSpaceAddr, u32 len) {
    (void)group; (void)globalWorkSpaceAddr; (void)len; HCCL_ERROR("[HcclWrapper] HcomSetGlobalWorkSpace not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetAivCoreLimit(const char* group, u32 aivCoreLimit) {
    (void)group; (void)aivCoreLimit; HCCL_ERROR("[HcclWrapper] HcomSetAivCoreLimit not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReleaseSubComms() {
    HCCL_ERROR("[HcclWrapper] HcomReleaseSubComms not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomUnloadTask(const char* group, const char* tag) {
    (void)group; (void)tag; HCCL_ERROR("[HcclWrapper] HcomUnloadTask not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomClearAivSyncBuf(const char* group, bool aivClearEnable) {
    (void)group; (void)aivClearEnable; HCCL_ERROR("[HcclWrapper] HcomClearAivSyncBuf not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len) {
    (void)group; (void)graphId; (void)stream; (void)len; HCCL_ERROR("[HcclWrapper] HcomSetAttachedStream not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSupportDeterministicOptim(const char* group, bool* isDeterministicOptim) {
    (void)group; (void)isDeterministicOptim; HCCL_ERROR("[HcclWrapper] HcomSupportDeterministicOptim not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, rtStream_t stream, int32_t deviceLogicId) {
    (void)addrList; (void)sizeList; (void)count; (void)stream; (void)deviceLogicId; HCCL_ERROR("[HcclWrapper] HcomTbeMemClean not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetInitStatus(bool* initiated) {
    (void)initiated; HCCL_ERROR("[HcclWrapper] HcomGetInitStatus not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllGather(const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)inputCount; (void)dataType; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomAllGather not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllGatherV(const char* tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType dataType, const char* group, rtStream_t stream) {
    (void)tag; (void)sendBuf; (void)sendCount; (void)recvBuf; (void)recvCounts; (void)rdispls; (void)dataType; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomAllGatherV not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)count; (void)dataType; (void)op; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomAllReduce not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)count; (void)dataType; (void)op; (void)root; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReduce not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomBroadcast(const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream) {
    (void)tag; (void)ptr; (void)count; (void)dataType; (void)root; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomBroadcast not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReduceScatter(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)count; (void)dataType; (void)op; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReduceScatter not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReduceScatterV(const char* tag, void* sendBuf, const void* sendCounts, const void* sdispls, void* recvBuf, u64 recvCount, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    (void)tag; (void)sendBuf; (void)sendCounts; (void)sdispls; (void)recvBuf; (void)recvCount; (void)dataType; (void)op; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReduceScatterV not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSend(const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)count; (void)dataType; (void)destRank; (void)srTag; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomSend not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReceive(const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group, rtStream_t stream) {
    (void)tag; (void)outputPtr; (void)count; (void)dataType; (void)srcRank; (void)srTag; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReceive not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAlltoAllV(const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    (void)sendBuf; (void)sendCounts; (void)sdispls; (void)sendType; (void)recvBuf; (void)recvCounts; (void)rdispls; (void)recvType; (void)group; (void)stream; (void)tag; HCCL_ERROR("[HcclWrapper] HcomAlltoAllV not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAlltoAllVC(const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    (void)sendBuf; (void)sendCountMatrix; (void)sendType; (void)recvBuf; (void)recvType; (void)group; (void)stream; (void)tag; HCCL_ERROR("[HcclWrapper] HcomAlltoAllVC not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllToAll(const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    (void)sendBuf; (void)sendCount; (void)sendType; (void)recvBuf; (void)recvCount; (void)recvType; (void)group; (void)stream; (void)tag; HCCL_ERROR("[HcclWrapper] HcomAllToAll not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetHcclComm(int64_t comm, std::string& group) {
    (void)comm; (void)group; HCCL_ERROR("[HcclWrapper] HcomGetHcclComm not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGenerateCclOpTag(const char* opType, s64 hcomComm, const char* group, char* sTag) {
    (void)opType; (void)hcomComm; (void)group; (void)sTag; HCCL_ERROR("[HcclWrapper] HcomGenerateCclOpTag not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetCommCCLBufferSize(const char* group, uint64_t& size) {
    (void)group; (void)size; HCCL_ERROR("[HcclWrapper] HcomGetCommCCLBufferSize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag) {
    (void)group; (void)topoType; (void)flag; HCCL_ERROR("[HcclWrapper] HcomGetL0TopoTypeEx not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag) {
    (void)group; (void)rankSize; (void)flag; HCCL_ERROR("[HcclWrapper] HcomGetRankSizeEx not supported"); return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcomDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcomGetRankSizePtr, "HcomGetRankSize", StubHcomGetRankSize, g_hcomGetRankSizeSupported);
    SET_PTR(hcomGetLocalRankSizePtr, "HcomGetLocalRankSize", StubHcomGetLocalRankSize, g_hcomGetLocalRankSizeSupported);
    SET_PTR(hcomGetRankIdPtr, "HcomGetRankId", StubHcomGetRankId, g_hcomGetRankIdSupported);
    SET_PTR(hcomGetLocalRankIdPtr, "HcomGetLocalRankId", StubHcomGetLocalRankId, g_hcomGetLocalRankIdSupported);
    SET_PTR(hcomGetWorldRankFromGroupRankPtr, "HcomGetWorldRankFromGroupRank", StubHcomGetWorldRankFromGroupRank, g_hcomGetWorldRankFromGroupRankSupported);
    SET_PTR(hcomGetGroupRankFromWorldRankPtr, "HcomGetGroupRankFromWorldRank", StubHcomGetGroupRankFromWorldRank, g_hcomGetGroupRankFromWorldRankSupported);
    SET_PTR(hcomCreateGroupPtr, "HcomCreateGroup", StubHcomCreateGroup, g_hcomCreateGroupSupported);
    SET_PTR(hcomDestroyGroupPtr, "HcomDestroyGroup", StubHcomDestroyGroup, g_hcomDestroyGroupSupported);
    SET_PTR(hcomSetGradFusionByIndexPtr, "HcomSetGradFusionByIndex", StubHcomSetGradFusionByIndex, g_hcomSetGradFusionByIndexSupported);
    SET_PTR(hcomSetGradFusionBySizePtr, "HcomSetGradFusionBySize", StubHcomSetGradFusionBySize, g_hcomSetGradFusionBySizeSupported);
    SET_PTR(hcomInitByRankTablePtr, "HcomInitByRankTable", StubHcomInitByRankTable, g_hcomInitByRankTableSupported);
    SET_PTR(hcomDestroyPtr, "HcomDestroy", StubHcomDestroy, g_hcomDestroySupported);
    SET_PTR(hcomGetCommHandleByGroupPtr, "HcomGetCommHandleByGroup", StubHcomGetCommHandleByGroup, g_hcomGetCommHandleByGroupSupported);
    SET_PTR(getGroupNameByOpBaseHcomPtr, "GetGroupNameByOpBaseHcom", StubGetGroupNameByOpBaseHcom, g_getGroupNameByOpBaseHcomSupported);
    SET_PTR(hcomCreateComResourceByCommPtr, "HcomCreateComResourceByComm", StubHcomCreateComResourceByComm, g_hcomCreateComResourceByCommSupported);
    SET_PTR(hcomTopoInfoRegCallbackPtr, "HcomTopoInfoRegCallback", StubHcomTopoInfoRegCallback, g_hcomTopoInfoRegCallbackSupported);
    SET_PTR(hcomGetandClearOverFlowTasksPtr, "HcomGetandClearOverFlowTasks", StubHcomGetandClearOverFlowTasks, g_hcomGetandClearOverFlowTasksSupported);
    SET_PTR(hcomGetWorkflowModePtr, "HcomGetWorkflowMode", StubHcomGetWorkflowMode, g_hcomGetWorkflowModeSupported);
    SET_PTR(hcomSetWorkflowModePtr, "HcomSetWorkflowMode", StubHcomSetWorkflowMode, g_hcomSetWorkflowModeSupported);
    SET_PTR(hcomCalcOpOnlinePtr, "HcomCalcOpOnline", StubHcomCalcOpOnline, g_hcomCalcOpOnlineSupported);
    SET_PTR(hcomCalcOpResOfflinePtr, "HcomCalcOpResOffline", StubHcomCalcOpResOffline, g_hcomCalcOpResOfflineSupported);
    SET_PTR(hcomGetMemTypePtr, "HcomGetMemType", StubHcomGetMemType, g_hcomGetMemTypeSupported);
    SET_PTR(hcomGetBandWidthPerNPUPtr, "HcomGetBandWidthPerNPU", StubHcomGetBandWidthPerNPU, g_hcomGetBandWidthPerNPUSupported);
    SET_PTR(hcomGetServerNumAndDeviceNumPerServerPtr, "HcomGetServerNumAndDeviceNumPerServer", StubHcomGetServerNumAndDeviceNumPerServer, g_hcomGetServerNumAndDeviceNumPerServerSupported);
    SET_PTR(hcomGetSecAddrCopyFlagPtr, "HcomGetSecAddrCopyFlag", StubHcomGetSecAddrCopyFlag, g_hcomGetSecAddrCopyFlagSupported);
    SET_PTR(hcomInitByStringPtr, "HcomInitByString", StubHcomInitByString, g_hcomInitByStringSupported);
    SET_PTR(hcomInitByMasterInfoPtr, "HcomInitByMasterInfo", StubHcomInitByMasterInfo, g_hcomInitByMasterInfoSupported);
    SET_PTR(hcomCreateCommCCLbufferPtr, "HcomCreateCommCCLbuffer", StubHcomCreateCommCCLbuffer, g_hcomCreateCommCCLbufferSupported);
    SET_PTR(hcomGetInCCLbufferPtr, "HcomGetInCCLbuffer", StubHcomGetInCCLbuffer, g_hcomGetInCCLbufferSupported);
    SET_PTR(hcomGetOutCCLbufferPtr, "HcomGetOutCCLbuffer", StubHcomGetOutCCLbuffer, g_hcomGetOutCCLbufferSupported);
    SET_PTR(hcomSetLaunchKernelModePtr, "HcomSetLaunchKernelMode", StubHcomSetLaunchKernelMode, g_hcomSetLaunchKernelModeSupported);
    SET_PTR(hcomGetAicpuOpStreamNotifyPtr, "HcomGetAicpuOpStreamNotify", StubHcomGetAicpuOpStreamNotify, g_hcomGetAicpuOpStreamNotifySupported);
    SET_PTR(hcomMc2AiCpuStreamAllocAndGetPtr, "HcomMc2AiCpuStreamAllocAndGet", StubHcomMc2AiCpuStreamAllocAndGet, g_hcomMc2AiCpuStreamAllocAndGetSupported);
    SET_PTR(hcomSetDumpDebugModePtr, "HcomSetDumpDebugMode", StubHcomSetDumpDebugMode, g_hcomSetDumpDebugModeSupported);
    SET_PTR(hcomGetAlgorithmPtr, "HcomGetAlgorithm", StubHcomGetAlgorithm, g_hcomGetAlgorithmSupported);
    SET_PTR(hcomGetAlgExecParamPtr, "HcomGetAlgExecParam", StubHcomGetAlgExecParam, g_hcomGetAlgExecParamSupported);
    SET_PTR(hcomSetAutoTuneModePtr, "HcomSetAutoTuneMode", StubHcomSetAutoTuneMode, g_hcomSetAutoTuneModeSupported);
    SET_PTR(hcomGetDeviceTypePtr, "HcomGetDeviceType", StubHcomGetDeviceType, g_hcomGetDeviceTypeSupported);
    SET_PTR(hcomSetProfilingModePtr, "HcomSetProfilingMode", StubHcomSetProfilingMode, g_hcomSetProfilingModeSupported);
    SET_PTR(hcomGetSplitStrategyPtr, "HcomGetSplitStrategy", StubHcomGetSplitStrategy, g_hcomGetSplitStrategySupported);
    SET_PTR(hcomFindGroupPtr, "HcomFindGroup", StubHcomFindGroup, g_hcomFindGroupSupported);
    SET_PTR(hcomSelectAlgPtr, "HcomSelectAlg", StubHcomSelectAlg, g_hcomSelectAlgSupported);
    SET_PTR(hcomCalcAivCoreNumPtr, "HcomCalcAivCoreNum", StubHcomCalcAivCoreNum, g_hcomCalcAivCoreNumSupported);
    SET_PTR(hcomSetWorkspaceResourcePtr, "HcomSetWorkspaceResource", StubHcomSetWorkspaceResource, g_hcomSetWorkspaceResourceSupported);
    SET_PTR(hcomSetGlobalWorkSpacePtr, "HcomSetGlobalWorkSpace", StubHcomSetGlobalWorkSpace, g_hcomSetGlobalWorkSpaceSupported);
    SET_PTR(hcomSetAivCoreLimitPtr, "HcomSetAivCoreLimit", StubHcomSetAivCoreLimit, g_hcomSetAivCoreLimitSupported);
    SET_PTR(hcomReleaseSubCommsPtr, "HcomReleaseSubComms", StubHcomReleaseSubComms, g_hcomReleaseSubCommsSupported);
    SET_PTR(hcomUnloadTaskPtr, "HcomUnloadTask", StubHcomUnloadTask, g_hcomUnloadTaskSupported);
    SET_PTR(hcomClearAivSyncBufPtr, "HcomClearAivSyncBuf", StubHcomClearAivSyncBuf, g_hcomClearAivSyncBufSupported);
    SET_PTR(hcomSetAttachedStreamPtr, "HcomSetAttachedStream", StubHcomSetAttachedStream, g_hcomSetAttachedStreamSupported);
    SET_PTR(hcomSupportDeterministicOptimPtr, "HcomSupportDeterministicOptim", StubHcomSupportDeterministicOptim, g_hcomSupportDeterministicOptimSupported);
    SET_PTR(hcomTbeMemCleanPtr, "HcomTbeMemClean", StubHcomTbeMemClean, g_hcomTbeMemCleanSupported);
    SET_PTR(hcomGetInitStatusPtr, "HcomGetInitStatus", StubHcomGetInitStatus, g_hcomGetInitStatusSupported);
    SET_PTR(hcomAllGatherPtr, "HcomAllGather", StubHcomAllGather, g_hcomAllGatherSupported);
    SET_PTR(hcomAllGatherVPtr, "HcomAllGatherV", StubHcomAllGatherV, g_hcomAllGatherVSupported);
    SET_PTR(hcomAllReducePtr, "HcomAllReduce", StubHcomAllReduce, g_hcomAllReduceSupported);
    SET_PTR(hcomReducePtr, "HcomReduce", StubHcomReduce, g_hcomReduceSupported);
    SET_PTR(hcomBroadcastPtr, "HcomBroadcast", StubHcomBroadcast, g_hcomBroadcastSupported);
    SET_PTR(hcomReduceScatterPtr, "HcomReduceScatter", StubHcomReduceScatter, g_hcomReduceScatterSupported);
    SET_PTR(hcomReduceScatterVPtr, "HcomReduceScatterV", StubHcomReduceScatterV, g_hcomReduceScatterVSupported);
    SET_PTR(hcomSendPtr, "HcomSend", StubHcomSend, g_hcomSendSupported);
    SET_PTR(hcomReceivePtr, "HcomReceive", StubHcomReceive, g_hcomReceiveSupported);
    SET_PTR(hcomAlltoAllVPtr, "HcomAlltoAllV", StubHcomAlltoAllV, g_hcomAlltoAllVSupported);
    SET_PTR(hcomAlltoAllVCPtr, "HcomAlltoAllVC", StubHcomAlltoAllVC, g_hcomAlltoAllVCSupported);
    SET_PTR(hcomAllToAllPtr, "HcomAllToAll", StubHcomAllToAll, g_hcomAllToAllSupported);
    SET_PTR(hcomGetHcclCommPtr, "HcomGetHcclComm", StubHcomGetHcclComm, g_hcomGetHcclCommSupported);
    SET_PTR(hcomGenerateCclOpTagPtr, "HcomGenerateCclOpTag", StubHcomGenerateCclOpTag, g_hcomGenerateCclOpTagSupported);
    SET_PTR(hcomGetCommCCLBufferSizePtr, "HcomGetCommCCLBufferSize", StubHcomGetCommCCLBufferSize, g_hcomGetCommCCLBufferSizeSupported);
    SET_PTR(hcomGetL0TopoTypeExPtr, "HcomGetL0TopoTypeEx", StubHcomGetL0TopoTypeEx, g_hcomGetL0TopoTypeExSupported);
    SET_PTR(hcomGetRankSizeExPtr, "HcomGetRankSizeEx", StubHcomGetRankSizeEx, g_hcomGetRankSizeExSupported);

    #undef SET_PTR
}

void HcomDlFini(void) {
    // 重置为桩函数，支持标志置 false（可选）
    #define RESET_PTR(ptr, stub, support_flag) do { ptr = stub; support_flag = false; } while(0)

    RESET_PTR(hcomGetRankSizePtr, StubHcomGetRankSize, g_hcomGetRankSizeSupported);
    RESET_PTR(hcomGetLocalRankSizePtr, StubHcomGetLocalRankSize, g_hcomGetLocalRankSizeSupported);
    RESET_PTR(hcomGetRankIdPtr, StubHcomGetRankId, g_hcomGetRankIdSupported);
    RESET_PTR(hcomGetLocalRankIdPtr, StubHcomGetLocalRankId, g_hcomGetLocalRankIdSupported);
    RESET_PTR(hcomGetWorldRankFromGroupRankPtr, StubHcomGetWorldRankFromGroupRank, g_hcomGetWorldRankFromGroupRankSupported);
    RESET_PTR(hcomGetGroupRankFromWorldRankPtr, StubHcomGetGroupRankFromWorldRank, g_hcomGetGroupRankFromWorldRankSupported);
    RESET_PTR(hcomCreateGroupPtr, StubHcomCreateGroup, g_hcomCreateGroupSupported);
    RESET_PTR(hcomDestroyGroupPtr, StubHcomDestroyGroup, g_hcomDestroyGroupSupported);
    RESET_PTR(hcomSetGradFusionByIndexPtr, StubHcomSetGradFusionByIndex, g_hcomSetGradFusionByIndexSupported);
    RESET_PTR(hcomSetGradFusionBySizePtr, StubHcomSetGradFusionBySize, g_hcomSetGradFusionBySizeSupported);
    RESET_PTR(hcomInitByRankTablePtr, StubHcomInitByRankTable, g_hcomInitByRankTableSupported);
    RESET_PTR(hcomDestroyPtr, StubHcomDestroy, g_hcomDestroySupported);
    RESET_PTR(hcomGetCommHandleByGroupPtr, StubHcomGetCommHandleByGroup, g_hcomGetCommHandleByGroupSupported);
    RESET_PTR(getGroupNameByOpBaseHcomPtr, StubGetGroupNameByOpBaseHcom, g_getGroupNameByOpBaseHcomSupported);
    RESET_PTR(hcomCreateComResourceByCommPtr, StubHcomCreateComResourceByComm, g_hcomCreateComResourceByCommSupported);
    RESET_PTR(hcomTopoInfoRegCallbackPtr, StubHcomTopoInfoRegCallback, g_hcomTopoInfoRegCallbackSupported);
    RESET_PTR(hcomGetandClearOverFlowTasksPtr, StubHcomGetandClearOverFlowTasks, g_hcomGetandClearOverFlowTasksSupported);
    RESET_PTR(hcomGetWorkflowModePtr, StubHcomGetWorkflowMode, g_hcomGetWorkflowModeSupported);
    RESET_PTR(hcomSetWorkflowModePtr, StubHcomSetWorkflowMode, g_hcomSetWorkflowModeSupported);
    RESET_PTR(hcomCalcOpOnlinePtr, StubHcomCalcOpOnline, g_hcomCalcOpOnlineSupported);
    RESET_PTR(hcomCalcOpResOfflinePtr, StubHcomCalcOpResOffline, g_hcomCalcOpResOfflineSupported);
    RESET_PTR(hcomGetMemTypePtr, StubHcomGetMemType, g_hcomGetMemTypeSupported);
    RESET_PTR(hcomGetBandWidthPerNPUPtr, StubHcomGetBandWidthPerNPU, g_hcomGetBandWidthPerNPUSupported);
    RESET_PTR(hcomGetServerNumAndDeviceNumPerServerPtr, StubHcomGetServerNumAndDeviceNumPerServer, g_hcomGetServerNumAndDeviceNumPerServerSupported);
    RESET_PTR(hcomGetSecAddrCopyFlagPtr, StubHcomGetSecAddrCopyFlag, g_hcomGetSecAddrCopyFlagSupported);
    RESET_PTR(hcomInitByStringPtr, StubHcomInitByString, g_hcomInitByStringSupported);
    RESET_PTR(hcomInitByMasterInfoPtr, StubHcomInitByMasterInfo, g_hcomInitByMasterInfoSupported);
    RESET_PTR(hcomCreateCommCCLbufferPtr, StubHcomCreateCommCCLbuffer, g_hcomCreateCommCCLbufferSupported);
    RESET_PTR(hcomGetInCCLbufferPtr, StubHcomGetInCCLbuffer, g_hcomGetInCCLbufferSupported);
    RESET_PTR(hcomGetOutCCLbufferPtr, StubHcomGetOutCCLbuffer, g_hcomGetOutCCLbufferSupported);
    RESET_PTR(hcomSetLaunchKernelModePtr, StubHcomSetLaunchKernelMode, g_hcomSetLaunchKernelModeSupported);
    RESET_PTR(hcomGetAicpuOpStreamNotifyPtr, StubHcomGetAicpuOpStreamNotify, g_hcomGetAicpuOpStreamNotifySupported);
    RESET_PTR(hcomMc2AiCpuStreamAllocAndGetPtr, StubHcomMc2AiCpuStreamAllocAndGet, g_hcomMc2AiCpuStreamAllocAndGetSupported);
    RESET_PTR(hcomSetDumpDebugModePtr, StubHcomSetDumpDebugMode, g_hcomSetDumpDebugModeSupported);
    RESET_PTR(hcomGetAlgorithmPtr, StubHcomGetAlgorithm, g_hcomGetAlgorithmSupported);
    RESET_PTR(hcomGetAlgExecParamPtr, StubHcomGetAlgExecParam, g_hcomGetAlgExecParamSupported);
    RESET_PTR(hcomSetAutoTuneModePtr, StubHcomSetAutoTuneMode, g_hcomSetAutoTuneModeSupported);
    RESET_PTR(hcomGetDeviceTypePtr, StubHcomGetDeviceType, g_hcomGetDeviceTypeSupported);
    RESET_PTR(hcomSetProfilingModePtr, StubHcomSetProfilingMode, g_hcomSetProfilingModeSupported);
    RESET_PTR(hcomGetSplitStrategyPtr, StubHcomGetSplitStrategy, g_hcomGetSplitStrategySupported);
    RESET_PTR(hcomFindGroupPtr, StubHcomFindGroup, g_hcomFindGroupSupported);
    RESET_PTR(hcomSelectAlgPtr, StubHcomSelectAlg, g_hcomSelectAlgSupported);
    RESET_PTR(hcomCalcAivCoreNumPtr, StubHcomCalcAivCoreNum, g_hcomCalcAivCoreNumSupported);
    RESET_PTR(hcomSetWorkspaceResourcePtr, StubHcomSetWorkspaceResource, g_hcomSetWorkspaceResourceSupported);
    RESET_PTR(hcomSetGlobalWorkSpacePtr, StubHcomSetGlobalWorkSpace, g_hcomSetGlobalWorkSpaceSupported);
    RESET_PTR(hcomSetAivCoreLimitPtr, StubHcomSetAivCoreLimit, g_hcomSetAivCoreLimitSupported);
    RESET_PTR(hcomReleaseSubCommsPtr, StubHcomReleaseSubComms, g_hcomReleaseSubCommsSupported);
    RESET_PTR(hcomUnloadTaskPtr, StubHcomUnloadTask, g_hcomUnloadTaskSupported);
    RESET_PTR(hcomClearAivSyncBufPtr, StubHcomClearAivSyncBuf, g_hcomClearAivSyncBufSupported);
    RESET_PTR(hcomSetAttachedStreamPtr, StubHcomSetAttachedStream, g_hcomSetAttachedStreamSupported);
    RESET_PTR(hcomSupportDeterministicOptimPtr, StubHcomSupportDeterministicOptim, g_hcomSupportDeterministicOptimSupported);
    RESET_PTR(hcomTbeMemCleanPtr, StubHcomTbeMemClean, g_hcomTbeMemCleanSupported);
    RESET_PTR(hcomGetInitStatusPtr, StubHcomGetInitStatus, g_hcomGetInitStatusSupported);
    RESET_PTR(hcomAllGatherPtr, StubHcomAllGather, g_hcomAllGatherSupported);
    RESET_PTR(hcomAllGatherVPtr, StubHcomAllGatherV, g_hcomAllGatherVSupported);
    RESET_PTR(hcomAllReducePtr, StubHcomAllReduce, g_hcomAllReduceSupported);
    RESET_PTR(hcomReducePtr, StubHcomReduce, g_hcomReduceSupported);
    RESET_PTR(hcomBroadcastPtr, StubHcomBroadcast, g_hcomBroadcastSupported);
    RESET_PTR(hcomReduceScatterPtr, StubHcomReduceScatter, g_hcomReduceScatterSupported);
    RESET_PTR(hcomReduceScatterVPtr, StubHcomReduceScatterV, g_hcomReduceScatterVSupported);
    RESET_PTR(hcomSendPtr, StubHcomSend, g_hcomSendSupported);
    RESET_PTR(hcomReceivePtr, StubHcomReceive, g_hcomReceiveSupported);
    RESET_PTR(hcomAlltoAllVPtr, StubHcomAlltoAllV, g_hcomAlltoAllVSupported);
    RESET_PTR(hcomAlltoAllVCPtr, StubHcomAlltoAllVC, g_hcomAlltoAllVCSupported);
    RESET_PTR(hcomAllToAllPtr, StubHcomAllToAll, g_hcomAllToAllSupported);
    RESET_PTR(hcomGetHcclCommPtr, StubHcomGetHcclComm, g_hcomGetHcclCommSupported);
    RESET_PTR(hcomGenerateCclOpTagPtr, StubHcomGenerateCclOpTag, g_hcomGenerateCclOpTagSupported);
    RESET_PTR(hcomGetCommCCLBufferSizePtr, StubHcomGetCommCCLBufferSize, g_hcomGetCommCCLBufferSizeSupported);
    RESET_PTR(hcomGetL0TopoTypeExPtr, StubHcomGetL0TopoTypeEx, g_hcomGetL0TopoTypeExSupported);
    RESET_PTR(hcomGetRankSizeExPtr, StubHcomGetRankSizeEx, g_hcomGetRankSizeExSupported);

    #undef RESET_PTR
}

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcomGetRankSize(const char* group, u32* rankSize) {
    return hcomGetRankSizePtr(group, rankSize);
}
HcclResult HcomGetLocalRankSize(const char* group, u32* localRankSize) {
    return hcomGetLocalRankSizePtr(group, localRankSize);
}
HcclResult HcomGetRankId(const char* group, u32* rankId) {
    return hcomGetRankIdPtr(group, rankId);
}
HcclResult HcomGetLocalRankId(const char* group, u32* localRankId) {
    return hcomGetLocalRankIdPtr(group, localRankId);
}
HcclResult HcomGetWorldRankFromGroupRank(const char* group, u32 groupRank, u32* worldRank) {
    return hcomGetWorldRankFromGroupRankPtr(group, groupRank, worldRank);
}
HcclResult HcomGetGroupRankFromWorldRank(u32 worldRank, const char* group, u32* groupRank) {
    return hcomGetGroupRankFromWorldRankPtr(worldRank, group, groupRank);
}
HcclResult HcomCreateGroup(const char* group, u32 rankNum, u32* rankIds) {
    return hcomCreateGroupPtr(group, rankNum, rankIds);
}
HcclResult HcomDestroyGroup(const char* group) {
    return hcomDestroyGroupPtr(group);
}
HcclResult HcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* inputIdxList) {
    return hcomSetGradFusionByIndexPtr(group, segmentNum, inputIdxList);
}
HcclResult HcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList) {
    return hcomSetGradFusionBySizePtr(group, segmentNum, sizeList);
}
HcclResult HcomInitByRankTable(const char* rankTable, uint32_t rankId) {
    return hcomInitByRankTablePtr(rankTable, rankId);
}
HcclResult HcomDestroy() {
    return hcomDestroyPtr();
}
HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle) {
    return hcomGetCommHandleByGroupPtr(group, commHandle);
}
HcclResult GetGroupNameByOpBaseHcom(s64 opBaseHcom, char** groupname) {
    return getGroupNameByOpBaseHcomPtr(opBaseHcom, groupname);
}
HcclResult HcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2) {
    return hcomCreateComResourceByCommPtr(comm, streamMode, isOpbaseMode, commContext, isMC2);
}
void HcomTopoInfoRegCallback(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *)) {
    hcomTopoInfoRegCallbackPtr(p1, p2);
}
HcclResult HcomGetandClearOverFlowTasks(const char* group, hccl::HcclDumpInfo** hcclDumpInfoPtr, s32* len) {
    return hcomGetandClearOverFlowTasksPtr(group, hcclDumpInfoPtr, len);
}
HcclWorkflowMode HcomGetWorkflowMode() {
    return hcomGetWorkflowModePtr();
}
HcclResult HcomSetWorkflowMode(HcclWorkflowMode mode) {
    return hcomSetWorkflowModePtr(mode);
}
HcclResult HcomCalcOpOnline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    return hcomCalcOpOnlinePtr(hcomOpParam, hcomResResponse);
}
HcclResult HcomCalcOpResOffline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    return hcomCalcOpResOfflinePtr(hcomOpParam, hcomResResponse);
}
HcclResult HcomGetMemType(const char* group, const char* socVersion, bool isMalloc, u32* memType, bool* isTsMem, bool withoutImplCompile, bool level2Address) {
    return hcomGetMemTypePtr(group, socVersion, isMalloc, memType, isTsMem, withoutImplCompile, level2Address);
}
HcclResult HcomGetBandWidthPerNPU(u32 level, float* bandWidth) {
    return hcomGetBandWidthPerNPUPtr(level, bandWidth);
}
HcclResult HcomGetServerNumAndDeviceNumPerServer(u32* serverNum, u32* deviceNumPerServer, u32* deviceNumPerAggregation) {
    return hcomGetServerNumAndDeviceNumPerServerPtr(serverNum, deviceNumPerServer, deviceNumPerAggregation);
}
bool HcomGetSecAddrCopyFlag(const char* socVersion) {
    return hcomGetSecAddrCopyFlagPtr(socVersion);
}
HcclResult HcomInitByString(const char* rankTableM, const char* identify, WorkMode commWorkMode, HcomInitConfig* initConfig) {
    return hcomInitByStringPtr(rankTableM, identify, commWorkMode, initConfig);
}
HcclResult HcomInitByMasterInfo(const char* masterIp, const char* masterPort, const char* masterDeviceId, const char* rankSize, const char* rankIp, HcomInitConfig* initConfig) {
    return hcomInitByMasterInfoPtr(masterIp, masterPort, masterDeviceId, rankSize, rankIp, initConfig);
}
HcclResult HcomCreateCommCCLbuffer(const char* group) {
    return hcomCreateCommCCLbufferPtr(group);
}
HcclResult HcomGetInCCLbuffer(const char* group, void** buffer, u64* size) {
    return hcomGetInCCLbufferPtr(group, buffer, size);
}
HcclResult HcomGetOutCCLbuffer(const char* group, void** buffer, u64* size) {
    return hcomGetOutCCLbufferPtr(group, buffer, size);
}
void HcomSetLaunchKernelMode(bool state) {
    hcomSetLaunchKernelModePtr(state);
}
HcclResult HcomGetAicpuOpStreamNotify(const char* group, HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify) {
    return hcomGetAicpuOpStreamNotifyPtr(group, opStream, aicpuNotifyNum, aicpuNotify);
}
HcclResult HcomMc2AiCpuStreamAllocAndGet(const char* group, u32 streamMode, rtStream_t* aiCpuStream) {
    return hcomMc2AiCpuStreamAllocAndGetPtr(group, streamMode, aiCpuStream);
}
void HcomSetDumpDebugMode(bool dumpDebug) {
    hcomSetDumpDebugModePtr(dumpDebug);
}
HcclResult HcomGetAlgorithm(u32 level, char** algo) {
    return hcomGetAlgorithmPtr(level, algo);
}
HcclResult HcomGetAlgExecParam(const char* tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType, bool clearEnable, HcclDataType dataType, HcclReduceOp op, void** commContext, u64* len, u32 aivCoreLimit) {
    return hcomGetAlgExecParamPtr(tag, group, count, inputPtr, outputPtr, opType, clearEnable, dataType, op, commContext, len, aivCoreLimit);
}
void HcomSetAutoTuneMode(bool autoTuneMode) {
    hcomSetAutoTuneModePtr(autoTuneMode);
}
DevType HcomGetDeviceType() {
    return hcomGetDeviceTypePtr();
}
HcclResult HcomSetProfilingMode(HcomProfilingMode profilingMode, const char* profilingOption) {
    return hcomSetProfilingModePtr(profilingMode, profilingOption);
}
HcclResult HcomGetSplitStrategy(const char* group, const struct model_feature* feature, u32** segmentIdxPtr, u32* len, bool* configured, GradSplitForceMode force, OriginalGraphShapeType shapeType) {
    return hcomGetSplitStrategyPtr(group, feature, segmentIdxPtr, len, configured, force, shapeType);
}
bool HcomFindGroup(const char* group) {
    return hcomFindGroupPtr(group);
}
HcclResult HcomSelectAlg(s64 comm, const char* group, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, int32_t aivCoreLimit, bool& ifAiv, char* algName) {
    return hcomSelectAlgPtr(comm, group, count, counts, dataType, op, opType, aivCoreLimit, ifAiv, algName);
}
HcclResult HcomCalcAivCoreNum(const char* group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, char* algName, u32* numBlocks) {
    return hcomCalcAivCoreNumPtr(group, opType, count, counts, dataType, aivCoreLimit, algName, numBlocks);
}
HcclResult HcomSetWorkspaceResource(const char* tag, const char* group, rtStream_t* stream, s32 len, void* memPtr, u64 maxSize) {
    return hcomSetWorkspaceResourcePtr(tag, group, stream, len, memPtr, maxSize);
}
HcclResult HcomSetGlobalWorkSpace(const char* group, void** globalWorkSpaceAddr, u32 len) {
    return hcomSetGlobalWorkSpacePtr(group, globalWorkSpaceAddr, len);
}
HcclResult HcomSetAivCoreLimit(const char* group, u32 aivCoreLimit) {
    return hcomSetAivCoreLimitPtr(group, aivCoreLimit);
}
HcclResult HcomReleaseSubComms() {
    return hcomReleaseSubCommsPtr();
}
HcclResult HcomUnloadTask(const char* group, const char* tag) {
    return hcomUnloadTaskPtr(group, tag);
}
HcclResult HcomClearAivSyncBuf(const char* group, bool aivClearEnable) {
    return hcomClearAivSyncBufPtr(group, aivClearEnable);
}
HcclResult HcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len) {
    return hcomSetAttachedStreamPtr(group, graphId, stream, len);
}
HcclResult HcomSupportDeterministicOptim(const char* group, bool* isDeterministicOptim) {
    return hcomSupportDeterministicOptimPtr(group, isDeterministicOptim);
}
HcclResult HcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, rtStream_t stream, int32_t deviceLogicId) {
    return hcomTbeMemCleanPtr(addrList, sizeList, count, stream, deviceLogicId);
}
HcclResult HcomGetInitStatus(bool* initiated) {
    return hcomGetInitStatusPtr(initiated);
}
HcclResult HcomAllGather(const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group, rtStream_t stream) {
    return hcomAllGatherPtr(tag, inputPtr, outputPtr, inputCount, dataType, group, stream);
}
HcclResult HcomAllGatherV(const char* tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType dataType, const char* group, rtStream_t stream) {
    return hcomAllGatherVPtr(tag, sendBuf, sendCount, recvBuf, recvCounts, rdispls, dataType, group, stream);
}
HcclResult HcomAllReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    return hcomAllReducePtr(tag, inputPtr, outputPtr, count, dataType, op, group, stream);
}
HcclResult HcomReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, const char* group, rtStream_t stream) {
    return hcomReducePtr(tag, inputPtr, outputPtr, count, dataType, op, root, group, stream);
}
HcclResult HcomBroadcast(const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream) {
    return hcomBroadcastPtr(tag, ptr, count, dataType, root, group, stream);
}
HcclResult HcomReduceScatter(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    return hcomReduceScatterPtr(tag, inputPtr, outputPtr, count, dataType, op, group, stream);
}
HcclResult HcomReduceScatterV(const char* tag, void* sendBuf, const void* sendCounts, const void* sdispls, void* recvBuf, u64 recvCount, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    return hcomReduceScatterVPtr(tag, sendBuf, sendCounts, sdispls, recvBuf, recvCount, dataType, op, group, stream);
}
HcclResult HcomSend(const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group, rtStream_t stream) {
    return hcomSendPtr(tag, inputPtr, count, dataType, destRank, srTag, group, stream);
}
HcclResult HcomReceive(const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group, rtStream_t stream) {
    return hcomReceivePtr(tag, outputPtr, count, dataType, srcRank, srTag, group, stream);
}
HcclResult HcomAlltoAllV(const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    return hcomAlltoAllVPtr(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, group, stream, tag);
}
HcclResult HcomAlltoAllVC(const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    return hcomAlltoAllVCPtr(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, group, stream, tag);
}
HcclResult HcomAllToAll(const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    return hcomAllToAllPtr(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, group, stream, tag);
}
HcclResult HcomGetHcclComm(int64_t comm, std::string& group) {
    return hcomGetHcclCommPtr(comm, group);
}
HcclResult HcomGenerateCclOpTag(const char* opType, s64 hcomComm, const char* group, char* sTag) {
    return hcomGenerateCclOpTagPtr(opType, hcomComm, group, sTag);
}
HcclResult HcomGetCommCCLBufferSize(const char* group, uint64_t& size) {
    return hcomGetCommCCLBufferSizePtr(group, size);
}
HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag) {
    return hcomGetL0TopoTypeExPtr(group, topoType, flag);
}
HcclResult HcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag) {
    return hcomGetRankSizeExPtr(group, rankSize, flag);
}

// ---------- 查询函数实现 ----------
#define DEFINE_QUERY(name) extern "C" bool HcommIsSupport##name(void) { return g_##name##Supported; }

DEFINE_QUERY(HcomGetRankSize)
DEFINE_QUERY(HcomGetLocalRankSize)
DEFINE_QUERY(HcomGetRankId)
DEFINE_QUERY(HcomGetLocalRankId)
DEFINE_QUERY(HcomGetWorldRankFromGroupRank)
DEFINE_QUERY(HcomGetGroupRankFromWorldRank)
DEFINE_QUERY(HcomCreateGroup)
DEFINE_QUERY(HcomDestroyGroup)
DEFINE_QUERY(HcomSetGradFusionByIndex)
DEFINE_QUERY(HcomSetGradFusionBySize)
DEFINE_QUERY(HcomInitByRankTable)
DEFINE_QUERY(HcomDestroy)
DEFINE_QUERY(HcomGetCommHandleByGroup)
DEFINE_QUERY(GetGroupNameByOpBaseHcom)
DEFINE_QUERY(HcomCreateComResourceByComm)
DEFINE_QUERY(HcomTopoInfoRegCallback)
DEFINE_QUERY(HcomGetandClearOverFlowTasks)
DEFINE_QUERY(HcomGetWorkflowMode)
DEFINE_QUERY(HcomSetWorkflowMode)
DEFINE_QUERY(HcomCalcOpOnline)
DEFINE_QUERY(HcomCalcOpResOffline)
DEFINE_QUERY(HcomGetMemType)
DEFINE_QUERY(HcomGetBandWidthPerNPU)
DEFINE_QUERY(HcomGetServerNumAndDeviceNumPerServer)
DEFINE_QUERY(HcomGetSecAddrCopyFlag)
DEFINE_QUERY(HcomInitByString)
DEFINE_QUERY(HcomInitByMasterInfo)
DEFINE_QUERY(HcomCreateCommCCLbuffer)
DEFINE_QUERY(HcomGetInCCLbuffer)
DEFINE_QUERY(HcomGetOutCCLbuffer)
DEFINE_QUERY(HcomSetLaunchKernelMode)
DEFINE_QUERY(HcomGetAicpuOpStreamNotify)
DEFINE_QUERY(HcomMc2AiCpuStreamAllocAndGet)
DEFINE_QUERY(HcomSetDumpDebugMode)
DEFINE_QUERY(HcomGetAlgorithm)
DEFINE_QUERY(HcomGetAlgExecParam)
DEFINE_QUERY(HcomSetAutoTuneMode)
DEFINE_QUERY(HcomGetDeviceType)
DEFINE_QUERY(HcomSetProfilingMode)
DEFINE_QUERY(HcomGetSplitStrategy)
DEFINE_QUERY(HcomFindGroup)
DEFINE_QUERY(HcomSelectAlg)
DEFINE_QUERY(HcomCalcAivCoreNum)
DEFINE_QUERY(HcomSetWorkspaceResource)
DEFINE_QUERY(HcomSetGlobalWorkSpace)
DEFINE_QUERY(HcomSetAivCoreLimit)
DEFINE_QUERY(HcomReleaseSubComms)
DEFINE_QUERY(HcomUnloadTask)
DEFINE_QUERY(HcomClearAivSyncBuf)
DEFINE_QUERY(HcomSetAttachedStream)
DEFINE_QUERY(HcomSupportDeterministicOptim)
DEFINE_QUERY(HcomTbeMemClean)
DEFINE_QUERY(HcomGetInitStatus)
DEFINE_QUERY(HcomAllGather)
DEFINE_QUERY(HcomAllGatherV)
DEFINE_QUERY(HcomAllReduce)
DEFINE_QUERY(HcomReduce)
DEFINE_QUERY(HcomBroadcast)
DEFINE_QUERY(HcomReduceScatter)
DEFINE_QUERY(HcomReduceScatterV)
DEFINE_QUERY(HcomSend)
DEFINE_QUERY(HcomReceive)
DEFINE_QUERY(HcomAlltoAllV)
DEFINE_QUERY(HcomAlltoAllVC)
DEFINE_QUERY(HcomAllToAll)
DEFINE_QUERY(HcomGetHcclComm)
DEFINE_QUERY(HcomGenerateCclOpTag)
DEFINE_QUERY(HcomGetCommCCLBufferSize)
DEFINE_QUERY(HcomGetL0TopoTypeEx)
DEFINE_QUERY(HcomGetRankSizeEx)