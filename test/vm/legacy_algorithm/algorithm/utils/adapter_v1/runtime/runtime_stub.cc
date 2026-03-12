#include "runtime_stub.h"
#include "mmpa_api.h"
#include "device_info_recorder.h"
#include "transformer.h"
#include "mem_layout.h"

using namespace checker;

HcclResult hrtSetDevice(s32 deviceLogicId)
{
    return HCCL_SUCCESS;
}

HcclResult hrtResetDevice(s32 deviceLogicId)
{
    return HCCL_SUCCESS;
}

HcclResult hrtGetDevice(s32 *deviceLogicId)
{
    u32 rankid = RankInfoRecorder::Global()->GetRankId();
    *deviceLogicId = RankInfoRecorder::Global()->rankId2phyId[rankid];
    return HCCL_SUCCESS;
}

ErrContextPub hrtErrMGetErrorContextPub()
{
}

void hrtErrMSetErrorContextPub(ErrContextPub errorContextPub)
{

    return;
}

HcclResult hrtRaGetSingleSocketVnicIpInfo(u32 phy_id, DeviceIdType deviceType, u32 deviceId, hccl::HcclIpAddress &vnicIP)
{
    // 初始化ra接口
    return HCCL_SUCCESS;
}

HcclResult hrtGetPairDeviceLinkType(u32 phyDevId, u32 otherPhyDevId, LinkTypeInServer &linkType)
{
    linkType = LinkTypeRecorder::Global()->devLinkTypeMap_[RankInfoRecorder::Global()->GetDevType()][phyDevId][otherPhyDevId];
    return HCCL_SUCCESS;
    // 返回linkType，会将其加入deviceLinkTypeMap_变量中
}

HcclResult hrtGetDeviceIndexByPhyId(u32 devicePhyId, u32 &deviceLogicId)
{
    deviceLogicId = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    *value = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtGetStreamId(HcclRtStream stream, s32 &streamId)
{
    streamId = 1;
    return HCCL_SUCCESS;
}

HcclResult hrtStreamActive(HcclRtStream active_stream, HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult hrtCtxGetCurrent(HcclRtContext *ctx)
{
    return HCCL_SUCCESS;
}

HcclResult hrtCtxSetCurrent(HcclRtContext ctx)
{
    return HCCL_SUCCESS;
}

HcclResult hrtFree(void *devPtr)
{
    return HCCL_SUCCESS;
}

// 这边不能返回空指针，返回空指针的话，会导致CCL buffer申请失败
HcclResult hrtMalloc(void **devPtr, u64 size, bool Level2Address)
{
    *devPtr = (void*)0x01;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceType(DevType &devType)
{
    devType = g_CheckerDevType2HcclDevType[RankInfoRecorder::Global()->GetDevType()];
    return HCCL_SUCCESS;
}

HcclResult hrtGetHccsPortNum(u32 deviceLogicId, s32 &num)
{
    num = 7; // 该接口打桩为天成场景的port数
    return HCCL_SUCCESS;
}

extern "C" {
HcclResult hrtGetDeviceSatMode(rtFloatOverflowMode_t* floatOverflowMode)
{
    *floatOverflowMode = RT_OVERFLOW_MODE_INFNAN;
    return HCCL_SUCCESS;
}
}


HcclResult hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType hcclModuleType,
    HcclRtDeviceInfoType hcclInfoType, s64 &val)
{
    u32 rankId  = RankInfoRecorder::Global()->GetRankId();
    if (hcclInfoType == HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID) {
        val = DeviceInfoRecorder::Global()->rankId2superdeviceId[rankId];
    } else if (hcclInfoType == HcclRtDeviceInfoType::HCCL_INFO_TYPE_SERVER_ID) {
        val = RankInfoRecorder::Global()->rankId2serverId[rankId];
    } else {
        HCCL_ERROR("This type [%d] is not mockered", hcclInfoType);
    }
    return HCCL_SUCCESS;
}

HcclResult hrtDeviceGetBareTgid(u32 *pid)
{
    CHK_PTR_NULL(pid);
    *pid == 1;
    return HCCL_SUCCESS;
}

HcclResult hrtGetHostIf(vector<pair<string, HcclIpAddress>> &hostIfs, u32 devPhyId)
{
    return HCCL_SUCCESS;
}

HcclResult hrtDrvGetPlatformInfo(uint32_t *info)
{
    return HCCL_SUCCESS;
}

HcclResult hrtDrvGetDevNum(uint32_t *num_dev)
{
    // 参数有效性检查
    CHK_PTR_NULL(num_dev);
    *num_dev = 0;
    return HCCL_SUCCESS;
}

thread_local u32 g_devicePhyId = INVALID_UINT;
#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtGetDevicePhyIdByIndex(u32 deviceLogicId, u32 &devicePhyId, bool isRefresh)
{
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}  // extern "C"
#endif

HcclResult hrtFreeHost(void *hostPtr)
{
    free(hostPtr);
    return HCCL_SUCCESS;
}

HcclResult hrtMallocHost(void **hostPtr, u64 size)
{
    *hostPtr = malloc(size);
    if (*hostPtr == nullptr) {
        HCCL_ERROR("malloc mem failed, size is %llu", size);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult hrtOpenTrace(void)
{
    return HCCL_SUCCESS;
}

void hrtTraceDestroy(TraHandle handle)
{
    return;
}

HcclResult hrtTraceSubmit(TraHandle handle, const void *buffer, u32 bufSize)
{
    return HCCL_SUCCESS;
}

HcclResult hrtTraceCreateWithAttr(const char *objName, TraHandle &handle)
{
    return HCCL_SUCCESS;
}

HcclResult hrtTraceSave(TracerType tracerType, bool syncFlag)
{
    return HCCL_SUCCESS;
}

HcclResult hrtMemSyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind)
{
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    if (MemLayout::Global()->GetBufferType((u64)dst) == BufferType::AIV_COMMINFO) {
        u64 dstAddr = 0;
        u64 dstSize = 0;
        MemLayout::Global()->GetRealAddr((u64)dst, dstAddr, dstSize);
       if (memcpy_s((void *)dstAddr, dstSize, src, count) != 0) {
           return HCCL_E_MEMORY;
       }
    }
    return HCCL_SUCCESS;
}

HcclResult hrtRaGetDeviceAllNicIP(vector<vector<HcclIpAddress>> &ipAddr)
{
    return HCCL_SUCCESS;
}

HcclResult hrtMemSet(void *dst, uint64_t destMax, uint64_t count)
{
    return HCCL_SUCCESS;
}

HcclResult hrtTraceSetGlobalAttr(const TraceGlobalAttr *attr)
{
    return HCCL_SUCCESS;
}

uint64_t MsprofSysCycleTime()
{
    return 0;
}

HcclResult hrtMemAsyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count,
    HcclRtMemcpyKind kind, rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult  hrtFunctionRegister(BinHandle binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                                uint32_t funcMode)
{
    return HCCL_SUCCESS;
}

HcclResult hrtDevBinaryRegister(const rtDevBinary_t *bin, BinHandle *handle)
{
    return HCCL_SUCCESS;
}

HcclResult hrtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
    rtStream_t stream, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
{
    return HCCL_SUCCESS;
}

int mmDladdr(void *addr, mmDlInfo *info)
{
    return 0;
}

int rtModelFake = 0;
rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus *status,
                                 rtModel_t *captureMdl)
{
    *captureMdl = &rtModelFake;
    return RT_ERROR_NONE;
}

rtError_t rtModelGetId(rtModel_t mdl, uint32_t *modelId){
    return RT_ERROR_NONE;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode)
{
    return ACL_SUCCESS;
}

aclError aclrtSetOpExecuteTimeOutV2(uint64_t timeout, uint64_t *actualTimeout)
{
    return ACL_SUCCESS;
}

aclError aclrtGetOpTimeOutInterval(uint64_t *interval)
{
    return ACL_SUCCESS;
}

HcclResult GetStreamCaptureInfo(rtStream_t stream, rtModel_t &rtModel, bool &isCapture)
{
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    rtStreamGetCaptureInfo(stream, &captureStatus, &rtModel);
    if (captureStatus == rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_ACTIVE) {
        isCapture = true;
    } else {
        isCapture = false;
    }
    return HCCL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
    aclrtFuncHandle *funcHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
    aclrtBinHandle *binHandle)
{
    return ACL_SUCCESS;
}

rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm, rtKernelLaunchCfg_t *cfg,
    void *hostArgs, uint32_t argsSize, rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum)
{
    return RT_ERROR_NONE;
}

HcclResult hrtRaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion) {
    *interfaceVersion = 1;
    return HCCL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
    aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserve)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *param, size_t paramSize, aclrtParamHandle *paramHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    return ACL_SUCCESS;
}