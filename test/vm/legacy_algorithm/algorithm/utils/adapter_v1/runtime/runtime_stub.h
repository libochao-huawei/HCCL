#ifndef RUNTIME_STUB_H
#define RUNTIME_STUB_H

#include "base.h"
#include "runtime/base.h"
#include "hccl_common.h"
#include "hccl_ip_address.h"
#include "adapter_error_manager_pub.h"
#include "adapter_hccp_common.h"
#include "hccl_ip_address.h"
#include "adapter_trace.h"
#include "link_type_recorder.h"
#include <vector>
#include <string>
#include "rank_info_recorder.h"
#include "acl/acl_rt.h"
#include "runtime/rts/rts_kernel.h"

using namespace std;
using namespace hccl;

#ifdef __cplusplus
extern "C" {
#endif

HcclResult hrtGetDevice(s32 *deviceLogicId);

HcclResult hrtMalloc(void **devPtr, u64 size, bool Level2Address);

HcclResult hrtGetDeviceType(DevType &devType);

HcclResult hrtGetHccsPortNum(u32 deviceLogicId, s32 &num);

HcclResult hrtGetDeviceIndexByPhyId(u32 devicePhyId, u32 &deviceLogicId);

HcclResult hrtFree(void *devPtr);
HcclResult hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType hcclModuleType,
    HcclRtDeviceInfoType hcclInfoType, s64 &val);

HcclResult hrtGetDeviceSatMode(rtFloatOverflowMode_t* floatOverflowMode);
HcclResult hrtMemSyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count, HcclRtMemcpyKind kind);
#ifdef __cplusplus
}
#endif

HcclResult hrtSetDevice(s32 deviceLogicId);

HcclResult hrtGetPairDeviceLinkType(u32 phyDevId, u32 otherPhyDevId, LinkTypeInServer &linkType);

HcclResult hrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value);

HcclResult hrtGetStreamId(HcclRtStream stream, s32 &streamId);

HcclResult hrtStreamActive(HcclRtStream active_stream, HcclRtStream stream);

HcclResult hrtCtxGetCurrent(HcclRtContext *ctx);

HcclResult hrtCtxSetCurrent(HcclRtContext ctx);

void hrtErrMSetErrorContextPub(ErrContextPub errorContextPub);

ErrContextPub hrtErrMGetErrorContextPub();

HcclResult hrtRaGetSingleSocketVnicIpInfo(u32 phy_id, DeviceIdType deviceType, u32 deviceId, hccl::HcclIpAddress &vnicIP);



HcclResult hrtGetHostIf(vector<pair<string, HcclIpAddress>> &hostIfs, u32 devPhyId);

HcclResult hrtDrvGetPlatformInfo(uint32_t *info);

HcclResult hrtDrvGetDevNum(uint32_t *num_dev);

HcclResult hrtFreeHost(void *hostPtr);

HcclResult hrtMallocHost(void **hostPtr, u64 size);
HcclResult hrtResetDevice(s32 deviceLogicId);

HcclResult hrtOpenTrace(void);

void hrtTraceDestroy(TraHandle handle);

HcclResult hrtTraceSubmit(TraHandle handle, const void *buffer, u32 bufSize);

HcclResult hrtTraceCreateWithAttr(const char *objName, TraHandle &handle);

HcclResult hrtTraceSetGlobalAttr(const TraceGlobalAttr *attr);

HcclResult hrtTraceSave(TracerType tracerType, bool syncFlag);
HcclResult hrtRaGetDeviceAllNicIP(std::vector<std::vector<hccl::HcclIpAddress>> &ipAddr);

uint64_t MsprofSysCycleTime();
using BinHandle = void *;
HcclResult hrtMemAsyncCopy(void *dst, uint64_t destMax, const void *src, uint64_t count,
    HcclRtMemcpyKind kind, rtStream_t stream);

HcclResult  hrtFunctionRegister(BinHandle binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                                uint32_t funcMode);

HcclResult hrtDevBinaryRegister(const rtDevBinary_t *bin, BinHandle *handle);

HcclResult hrtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
    rtStream_t stream, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo);

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
    aclrtFuncHandle *funcHandle);

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle);

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
    aclrtBinHandle *binHandle);

rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm, rtKernelLaunchCfg_t *cfg,
    void *hostArgs, uint32_t argsSize, rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum);

#endif
