/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: rts stub
 */

#include "rts_stub.h"
#include <string>
#include <cstring>
#include <iostream>
#include "FakeStreamMgr.h"
#include "hccl_sim_pub_stub.h"
// #include "common/aicpu_hccl_def.h"
#include "dispatcher_ffts_pub.h"
#include "../hccp_stub/fake_socket.h"
#include "runtime/rts/rts.h"
#include "SimRunnerMgr.h"

/**
 * 设计思路：
 * 任务下发阶段：只接收task并存储
 * 任务执行阶段(外部触发stream的sync时)：轮询各个stream，执行未阻塞任务。
 * */
#ifdef __cplusplus
extern "C" {

struct FftsGraphNode {
    int graphId;
    int predCnt;
    vector<int> succList;
    bool isLastNode;
};

std::map<uint32_t, rtDataType_t> fftsSqeDataTypeMap = {
    {0, rtDataType_t::RT_DATA_TYPE_INT8},
    {1, rtDataType_t::RT_DATA_TYPE_INT16},
    {2, rtDataType_t::RT_DATA_TYPE_INT32},
    {6, rtDataType_t::RT_DATA_TYPE_FP16},
    {7, rtDataType_t::RT_DATA_TYPE_FP32},
    {0xf, rtDataType_t::RT_DATA_TYPE_UINT32},  // 0xf不支持
    {0x8, rtDataType_t::RT_DATA_TYPE_BFP16}
};

std::map<uint32_t, rtRecudeKind_t> fftsSqeReduceOpMap = {
    {1, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD},
    {2, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {3, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MIN}
};

std::map<u8, rtRecudeKind_t> RT_UB_REDUCE_OP_CODE_MAP = {
    {0xA, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_ADD},
    {0x8, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MAX},
    {0x9, rtRecudeKind_t::RT_MEMCPY_SDMA_AUTOMATIC_MIN}
};

std::map<uint8_t, rtDataType_t> RT_UB_REDUCE_DATA_TYPE_MAP = {
    {0x0, rtDataType_t::RT_DATA_TYPE_INT8},
    {0x1, rtDataType_t::RT_DATA_TYPE_INT16},
    {0x2, rtDataType_t::RT_DATA_TYPE_INT32},
    {0x3, rtDataType_t::RT_DATA_TYPE_UINT8},
    {0x4, rtDataType_t::RT_DATA_TYPE_UINT16},
    {0x5, rtDataType_t::RT_DATA_TYPE_UINT32},
    {0x6, rtDataType_t::RT_DATA_TYPE_FP16},
    {0x7, rtDataType_t::RT_DATA_TYPE_FP32},
    {0x8, rtDataType_t::RT_DATA_TYPE_BFP16},
    {0x9, rtDataType_t::RT_DATA_TYPE_BFP32}  // DataType::BF16_SAT新类型不确定如何转换
};

typedef struct task_info_s
{
    u32 streamId;
    u32 taskId;
} task_info_t;
 
thread_local task_info_t task_info;

void LaunchSqeTask(int streamId, FakeSqe sqe)
{
    auto fakeStreamMgr = SimRunnerMgr::GetInstance().GetFakeStreamMgr();
    if (fakeStreamMgr != nullptr) {
        fakeStreamMgr->Append(streamId, sqe);
        // 记录task和streamId信息
        task_info.taskId = ++task_info.taskId;
        task_info.streamId = streamId;
    }
}

// Notify related functionsrtEventCreate
rtError_t rtNotifyCreate(int32_t deviceId, rtNotify_t *notify)
{
    *notify =
        static_cast<void *>(SimRunnerMgr::GetInstance().GetFakeStreamMgr()->GetFakeNotifyMgr()->CreateNotify(deviceId));  // deviceId is rankId in ST
    return 0;
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->GetFakeNotifyMgr()->DestroyNotify(static_cast<int *>(notify));
    return 0;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_RECORD;
    sqe.notifyId = *(static_cast<int *>(notify));
    LaunchSqeTask(*static_cast<int *>(stream), sqe);
    return 0;
}

rtError_t rtNotifyAicpuRecord(int notifyId, int notifyCnt, int streamId)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_RECORD;
    sqe.notifyId = notifyId;
    sqe.notifyCnt = notifyCnt;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->Append(streamId, sqe);
    return 0;
}

bool LaunchFftsNotifyRecord(rtNotify_t notify, rtStream_t stm, FftsGraphNode graphNode)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_RECORD;
    sqe.notifyId = *(static_cast<int *>(notify));
    sqe.predCnt = graphNode.predCnt;
    sqe.succList = graphNode.succList;
    sqe.isLastNode = graphNode.isLastNode;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->AppendGraph(*(static_cast<int *>(stm)), graphNode.graphId, sqe);
    return true;
}

rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_WAIT;
    sqe.notifyId = *(static_cast<int *>(notify));
    LaunchSqeTask(*static_cast<int *>(stm), sqe);
    return 0;
}

rtError_t rtNotifyAicpuWait(int notifyId, int notifyCnt, int streamId)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_WAIT;
    sqe.notifyId = notifyId;
    sqe.notifyCnt = notifyCnt;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->Append(streamId, sqe);
    return 0;
}

bool LaunchFftsNotifyWait(rtNotify_t notify, rtStream_t stm, FftsGraphNode graphNode)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_WAIT;
    sqe.notifyId = *(static_cast<int *>(notify));
    sqe.predCnt = graphNode.predCnt;
    sqe.succList = graphNode.succList;
    sqe.isLastNode = graphNode.isLastNode;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->AppendGraph(*(static_cast<int *>(stm)), graphNode.graphId, sqe);
    return true;
}

bool LaunchFftsLabelRecord(rtStream_t stm, FftsGraphNode graphNode)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::HCCL_LABEL;
    sqe.predCnt = graphNode.predCnt;
    sqe.succList = graphNode.succList;
    sqe.isLastNode = graphNode.isLastNode;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->AppendGraph(*(static_cast<int *>(stm)), graphNode.graphId, sqe);
    return true;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    rtNotifyWait(notify, stream);
    return 0;
}

rtError_t rtIpcOpenNotify(rtNotify_t *notify, const char_t *name)
{
    return 0;
}

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t *devAddrOffset)
{
    return 0;
}

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char *key, size_t len, uint64_t flag)
{
    return 0;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t *notifyId)
{
    *notifyId = *(static_cast<int *>(notify));
    return 0;
}

rtError_t rtSetIpcNotifyPid(const char_t *name, int32_t pid[], int32_t num)
{
    return 0;
}

rtError_t rtGetNotifyAddress(rtNotify_t notify, uint64_t *const notifyAddres)
{
    *notifyAddres = *(static_cast<int *>(notify));
    return 0;
}

// memory related functions
rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    int ret = simRunnerMgr.GetShmPoolMgr()->CalcMem(devPtr, size);
    if (ret != 0) {
        HCCL_DEBUG("[rtMalloc]: Malloc memory for rank[%d], failed. size=[%lu]", size);
        return -1;
    }
    return 0;
}

rtError_t rtHostMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    *devPtr = static_cast<void *>(new char[size]);
    memset(*devPtr, 0, size);
    return 0;
}

rtError_t rtHostFree(void *devPtr)
{
    delete[] static_cast<char *>(devPtr);
    return 0;
}

aclError aclrtFree(void *devPtr)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    // Aicpu场景下: device侧内存统一释放，此处rtFree不做处理
    // Host场景下: 需要通过打印查看是否存在rtFree释放内存场景
    if (!simRunnerMgr.GetShmPoolMgr()->CheckIfDeviceAddress(devPtr)) {
        HCCL_DEBUG("[aclrtFree]Current is aicpu mode...");
        delete[] static_cast<char *>(devPtr);
    }
    return ACL_SUCCESS;
}

aclError aclrtMallocHostWithCfg(void **hostPtr, uint64_t size, aclrtMallocConfig *cfg)
{
    *hostPtr = static_cast<void *>(new char[size]);
    memset(*hostPtr, 0, size);
    return 0;
}

aclError aclrtFreeHost(void *hostPtr)
{
    delete[] static_cast<char *>(hostPtr);
    return 0;
}

rtError_t rtReduceAsync(
    void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtRecudeKind_t kind, rtDataType_t type, rtStream_t stm)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::SDMA_REDUCE;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    sqe.dataType = type;
    sqe.reduceOp = kind;
    LaunchSqeTask(*static_cast<int *>(stm), sqe);
    return 0;
}

rtError_t rtReduceAicpuAsync(
    void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtRecudeKind_t kind, rtDataType_t type, int streamId)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::SDMA_REDUCE;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    sqe.dataType = type;
    sqe.reduceOp = kind;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->Append(streamId, sqe);
    return 0;
}

bool LaunchFftsReduceAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtRecudeKind_t kind,
    rtDataType_t type, rtStream_t stm, FftsGraphNode graphNode)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::SDMA_REDUCE;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    sqe.dataType = type;
    sqe.reduceOp = kind;
    sqe.predCnt = graphNode.predCnt;
    sqe.succList = graphNode.succList;
    sqe.isLastNode = graphNode.isLastNode;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->AppendGraph(*static_cast<int *>(stm), graphNode.graphId, sqe);
    return 0;
}

rtError_t rtReduceAsyncV2(void *dst, uint64_t destMax, const void *src, uint64_t count, rtRecudeKind_t kind,
    rtDataType_t type, rtStream_t stm, void *overflowAddr)
{
    rtReduceAsync(dst, destMax, src, count, kind, type, stm);
    return 0;
}

rtError_t rtReduceAsyncWithCfgV2(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtRecudeKind_t kind,
    rtDataType_t type, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    rtReduceAsync(dst, destMax, src, cnt, kind, type, stm);
    return 0;
}

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind)
{
    if (cnt > destMax) {
        return 1;
    }
    memcpy(dst, src, cnt);
    return 0;
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    if (count > destMax) {
        return 1;
    }
    memcpy(dst, src, count);
    return ACL_SUCCESS;
}

rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm)
{
    if (cnt > destMax) {
        return 1;
    }
    FakeSqe sqe{};
    sqe.type = FakeSqeType::MEM_CPY;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    LaunchSqeTask(*static_cast<int *>(stm), sqe);
    return 0;
}

rtError_t rtMemcpyAicpuAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind, int streamId)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::MEM_CPY;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->Append(streamId, sqe);
    return 0;
}

bool LaunchFftsMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind,
    rtStream_t stm, FftsGraphNode graphNode)
{
    if (cnt > destMax) {
        return false;
    }
    FakeSqe sqe{};
    sqe.type = FakeSqeType::MEM_CPY;
    sqe.dst = dst;
    sqe.src = src;
    sqe.count = cnt;
    sqe.predCnt = graphNode.predCnt;
    sqe.succList = graphNode.succList;
    sqe.isLastNode = graphNode.isLastNode;
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->AppendGraph(*static_cast<int *>(stm), graphNode.graphId, sqe);
    return true;
}

rtError_t rtMemcpyAsyncWithCfgV2(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind,
    rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
{
    rtMemcpyAsync(dst, destMax, src, cnt, kind, stm);
    return 0;
}

aclError aclrtIpcMemGetExportKey(void *devPtr, size_t size, char *key, size_t len, uint64_t flag)
{
    return 0;
}

aclError aclrtIpcMemSetImportPid(const char *key, int32_t *pid, size_t num)
{
    return 0;
}

rtError_t rtIpcDestroyMemoryName(const char_t *name)
{
    return 0;
}

// device related functions
rtError_t rtGetDeviceIndexByPhyId(uint32_t phyId, uint32_t *devIndex)
{
    *devIndex = phyId;
    return 0;
}

aclError aclrtSetDevice(int32_t deviceId)
{
    SimRunnerMgr::GetInstance().SetDeviceId(deviceId);
    return 0;
}

rtError_t rtSetDevice(int32_t devId)
{
    SimRunnerMgr::GetInstance().SetDeviceId(devId);
    return 0;
}

rtError_t rtGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t *val)
{
    DevType deviceType;
    HcclResult ret = hrtGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        return -RT_ERROR_NONE;
    }

    if (deviceType == DevType::DEV_TYPE_910_93) {
        *val = 0;
        return RT_ERROR_NONE;
    }
    
    *val = phyId / 8;
    return RT_ERROR_NONE;
}

aclError aclrtResetDevice(int32_t deviceId)
{
    return 0;
}

rtError_t rtGetDevicePhyIdByIndex(uint32_t devIndex, uint32_t *phyId)
{
    *phyId = devIndex;
    // *phyId = SimRunnerMgr::GetInstance().GetNetDeviceMgr()->GetDevPhyId(devIndex);
    return 0;
}

aclError aclrtGetDevice(int32_t *deviceId)
{
    *deviceId = SimRunnerMgr::GetInstance().GetDeviceId();
    return 0;
}

// stream related function
rtError_t rtStreamCreateWithFlags(rtStream_t *stm, int32_t priority, uint32_t flags)
{
    *stm = static_cast<void *>(SimRunnerMgr::GetInstance().GetFakeStreamMgr()->CreateStream());
    return 0;
}

aclError aclrtDestroyStreamForce(aclrtStream stream)
{
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->DestroyStream(static_cast<int *>(stream));
    return 0;
}

rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout)
{
    return 0;
}

rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId)
{
    *taskId = task_info.taskId;
    *streamId = task_info.streamId;
    return 0;
}

aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream)
{
    return 0;
}

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    return 0;
}

rtError_t rtStreamGetMode(rtStream_t const stm, uint64_t *const stmMode)
{
    return 0;
}

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue *value)
{
    return 0;
}

rtError_t rtStreamSetMode(rtStream_t stm, const uint64_t stmMode)
{
    return 0;
}

aclError aclrtStreamGetId(aclrtStream stream, int32_t *streamId)
{
    *streamId = *static_cast<int *>(stream);
    return 0;
}

// other functions
rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag)
{
    return 0;
}

rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc)
{
    return 0;
}

rtError_t rtDevBinaryUnRegister(void *hdl)
{
    return 0;
}

rtError_t rtAicpuKernelLaunchWithFlag(const rtKernelLaunchNames_t *launchNames, uint32_t blockDim,
    const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags)
{
    return 0;
}

aclError aclrtIpcMemImportByKey(void **devPtr, const char *key, uint64_t flag)
{
    return 0;
}

rtError_t rtGetP2PStatus(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t *status)
{
    *status = 1;
    return 0;
}

rtError_t rtRDMASend(uint32_t sqIndex, uint32_t wqeIndex, rtStream_t stm)
{
    return 0;
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
{
    return 0;
}

rtError_t rtKernelLaunch(
    const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc, rtStream_t stm)
{
    return 0;
}

rtError_t rtGetSocVersion(char_t *ver, const uint32_t maxLen)
{
    if (strlen(GetFakeSocVersionStub()) >= 32) {
        throw std::exception();
    }
    strcpy(ver, GetFakeSocVersionStub());
    return 0;
}

rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t *floatOverflowMode)
{
    return 0;
}

rtError_t rtPointerGetAttributes(rtPointerAttributes_t *attributes, const void *ptr)
{
    return 0;
}

rtError_t rtIpcCloseMemory(const void *ptr)
{
    return 0;
}

rtError_t rtDeviceGetBareTgid(uint32_t *pid)
{
    return 0;
}

rtError_t rtRDMADBSend(uint32_t dbIndex, uint64_t dbInfo, rtStream_t stm)
{
    return 0;
}
}

rtError_t rtNotifyCreateWithFlag(int32_t device, rtNotify_t *notify, uint32_t flag)
{
    return rtNotifyCreate(device, notify);
}

rtError_t rtNotifyGetPhyInfo(rtNotify_t notify, uint32_t *phyDevId, uint32_t *tsId)
{
    int notifyId = *(static_cast<int *>(notify));
    int rankId = SimRunnerMgr::GetInstance().GetFakeStreamMgr()->GetFakeNotifyMgr()->GetRankIdByNotifyId(notifyId);
    *phyDevId = rankId;
    *tsId = 0;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo *notifyInfo)
{
    notifyInfo->phyId = 1;
    notifyInfo->tsId = 3;
    notifyInfo->flag = 0;
    return RT_ERROR_NONE;
}

drvError_t drvGetLocalDevIDByHostDevID(uint32_t remote_udevid, uint32_t *local_devid)
{
    *local_devid = remote_udevid;
    return DRV_ERROR_NONE;
}

rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, rtStream_t stm, rtKernelLaunchCfg_t *cfg,
    void *hostArgs, uint32_t argsSize, rtPlaceHolderInfo_t *placeHolderArray, uint32_t placeHolderNum)
{
    return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
    const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flags)
{
    auto shmPoolMgr = SimRunnerMgr::GetInstance().GetShmPoolMgr();
    auto kernelNameAddr = reinterpret_cast<const char *>(argsInfo->args) + argsInfo->kernelNameAddrOffset;
    char kernelName[32];
    strcpy(kernelName, kernelNameAddr);
    int rankId = 0;
    aclrtGetDevice(&rankId);
    pid_t pid_d_temp = SimRunnerMgr::GetInstance().GetPid(rankId);
    HCCL_DEBUG("Aicpu KernelLaunch, kernelName:%s, rankId:%d, pid:%d", kernelName, rankId, pid_d_temp);
    HCCL_INFO("[rtAicpuKernelLaunchExWithArgs]context[%llu]", reinterpret_cast<KFCTaskCommStub *>(argsInfo->args)->context);
    if (std::string(kernelName) == "RunAicpuKfcResInitV2") {
        shmPoolMgr->SetSignalCmd(rankId, '0', reinterpret_cast<u64>(argsInfo->args));
        kill(pid_d_temp, SIGUSR1);  // 子进程的pid
        while (shmPoolMgr->GetDeviceFlag(rankId) == true) {
            sleep(1);
        }
    } else if (std::string(kernelName) == "RunAicpuRpcSrvLaunchV2") {
        // host侧argsInfo->args是一个临时变量，此处需要做h2d转换
        KFCTaskCommStub *task = reinterpret_cast<KFCTaskCommStub *>(argsInfo->args);
        OpTilingDataStub *tilingData = reinterpret_cast<OpTilingDataStub *>(task->tilingData);
        u64 tillingDataSize = sizeof(OpTilingDataStub) + tilingData->length;
        void *tilling = nullptr;
        // 1.申请一片内存，device侧地址
        rtError_t rt_ret = rtMalloc(&tilling, tillingDataSize, RT_MEMORY_P2P_HBM, 0);
        // 2.把数据copy过去
        memcpy(tilling, tilingData, tillingDataSize);  // 转化为host
        // 3.地址是*tilling，转化为u64
        u64 tempTillingAddr = reinterpret_cast<u64>(tilling);
        // 4，把argsinfo 拿出来，把argsinfo->args 改成*tilling
        u64 tempAddr = shmPoolMgr->SetTilingData(rankId, task->context, tempTillingAddr);  // 保证是devive的

        shmPoolMgr->SetSignalCmd(rankId, '1', tempAddr);
        kill(pid_d_temp, SIGUSR1);  // 子进程的pid
        while (shmPoolMgr->GetDeviceFlag(rankId) == true) {
            sleep(1);
        }
    } else if (std::string(kernelName) == "HcclKernelEntrance") {  //2.0 aicpu
        auto *kernelParam = reinterpret_cast<HcclKernelLaunchParamStub *>(argsInfo->args);
        HCCL_DEBUG("kernelParam->algName:%s", kernelParam->kernel.algName);
        u64 tempAddr = shmPoolMgr->SetKernelParamLite(rankId, argsInfo->args);
        shmPoolMgr->SetSignalCmd(rankId, '2', tempAddr);
        kill(pid_d_temp, SIGUSR1);  // 子进程的pid
        while (shmPoolMgr->GetDeviceFlag(rankId) == true) {
            sleep(1);
        }
    } else {
        HCCL_ERROR("[rtAicpuKernelLaunchExWithArgs]unsupported kernelName:%s", kernelName);
    }
    return RT_ERROR_NONE;
}

rtError_t rtIpcOpenNotifyWithFlag(rtNotify_t *notify, const char_t *name, uint32_t flag)
{
    return rtIpcOpenNotify(notify, name);
}

rtError_t rtStreamGetSqid(const rtStream_t stream, uint32_t *sqId)
{
    if (stream == NULL || sqId == NULL) {
        return 1;
    }
    *sqId = reinterpret_cast<uintptr_t>(const_cast<void *>(stream));
    return RT_ERROR_NONE;
}

// 采用v2.0的打桩方式
rtError_t rtStreamDestroy(rtStream_t stream)
{
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->DestroyStream(static_cast<int *>(stream));
    return 0;
}

rtError_t rtGetIsHeterogenous(int32_t *heterogenous)
{
    *heterogenous = 0;
    return 0;
}

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority)
{
    return 0;
}

rtError_t rtStreamSynchronize(rtStream_t stream)
{
    SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetSqeTasks();
    SimRunnerMgr::GetInstance().GetFakeStreamMgr()->Sync(*static_cast<int *>(stream));
    return 0;
}

bool ParseFftsPlusNotifyCtx(rtFftsPlusComCtx_t *ctx, rtStream_t stream, FftsGraphNode graphNode)
{
    rtFftsPlusNotifyCtx_t *nofifyCtx = reinterpret_cast<rtFftsPlusNotifyCtx_t *>(ctx);
    rtNotify_t notify = reinterpret_cast<rtNotify_t>(nofifyCtx->notifyIdBase);
    return LaunchFftsNotifyWait(&notify, stream, graphNode);
}

bool ParseFftsPlusWriteValueCtx(rtFftsPlusComCtx_t *ctx, rtStream_t stream, FftsGraphNode graphNode)
{
    const u32 shift = 32;
    const u32 notifyFlag = 4;  // 标识notify record
    rtFftsPlusWriteValueCtx_t *writeCtx = reinterpret_cast<rtFftsPlusWriteValueCtx_t *>(ctx);
    if (writeCtx->res11 == notifyFlag) {
        u64 notifyId = ((u64)writeCtx->writeAddressBaseH << shift) | (u64)writeCtx->writeAddressBaseL;
        rtNotify_t notify = reinterpret_cast<rtNotify_t>(notifyId);
        return LaunchFftsNotifyRecord(&notify, stream, graphNode);
    }
    return RT_ERROR_NONE;
}

bool ParseFftsLabelCtx(rtStream_t stream, FftsGraphNode graphNode)
{
    return LaunchFftsLabelRecord(stream, graphNode);
}

bool ParseFftsPlusSdmaCtx(rtFftsPlusComCtx_t *ctx, rtStream_t stream, FftsGraphNode graphNode)
{
    const u32 shift = 32;
    const u32 invalidParam = 0xf;
    rtFftsPlusSdmaCtx_t *sdmaCtx = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(ctx);
    // 解析src和dst的起始地址
    u64 srcAddr = ((u64)sdmaCtx->sourceAddressBaseH << shift) | (u64)sdmaCtx->sourceAddressBaseL;
    const void *src = reinterpret_cast<const void *>(srcAddr);
    u64 destAddr = ((u64)sdmaCtx->destinationAddressBaseH << shift) | (u64)sdmaCtx->destinationAddressBaseL;
    void *dst = reinterpret_cast<void *>(destAddr);
    u64 count = sdmaCtx->tailDataLength;
    // 根据sdmaSqeHeader判断是否为reduce操作
    if (sdmaCtx->sdmaSqeHeader == hccl::SDMA_FP32_ATOMIC_MOVE_SQE) {
        return LaunchFftsMemcpyAsync(dst, count, src, count, RT_MEMCPY_RESERVED, stream, graphNode);
    }

    // 解析FftsSdmaSqeHeader结构获取Reduce所需参数信息
    hccl::FftsSdmaSqeHeader *sqeHeader = reinterpret_cast<hccl::FftsSdmaSqeHeader *>(&sdmaCtx->sdmaSqeHeader);
    uint32_t datatype = sqeHeader->bit.datatype;
    uint32_t reduceOp = sqeHeader->bit.opcode;
    if (datatype == invalidParam || reduceOp == invalidParam) {
        HCCL_ERROR(
            "fftsSdmaSqeHeader is invaild, sqeHeader->bit.datatype:%u, sqeHeader->bit.opcode:%u.", datatype, reduceOp);
        return false;
    }

    // 根据映射关系解析数据类型和ReduceOp类型
    rtDataType_t dataType = fftsSqeDataTypeMap[datatype];
    rtRecudeKind_t reduceKind = fftsSqeReduceOpMap[reduceOp];
    return LaunchFftsReduceAsync(dst, count, src, count, reduceKind, dataType, stream, graphNode);
}

bool ParseFftsPlusTask(const rtFftsPlusTaskInfo_t *task, const rtStream_t stream)
{
    if (task == nullptr || task->descBuf == nullptr || stream == nullptr) {
        HCCL_ERROR("ParseFftsPlusTask failed to check input parameters.");
        return false;
    }

    // 计算context的总数量并遍历解析
    auto ctxNum = task->descBufLen / sizeof(rtFftsPlusComCtx_t);
    for (auto index = 0; index < ctxNum; index++) {
        // 从缓存中逐个解析context
        rtFftsPlusComCtx_t *ctx = reinterpret_cast<rtFftsPlusComCtx_t *>(
            reinterpret_cast<u64>(task->descBuf) + index * sizeof(rtFftsPlusComCtx_t));
        if (ctx == nullptr) {
            HCCL_ERROR("parse rtFftsPlusComCtx_t failed, index:%u.", index);
            continue;
        }

        // 解析ffts子图生产者和消费者信息
        vector<int> succList;
        int predCnt = static_cast<int>(ctx->predCnt);
        int succNum = static_cast<int>(ctx->successorNum);
        for (int i = 0; i < succNum; i++) {
            succList.push_back(ctx->successorList[i]);
        }

        // 构造ffts子图节点信息结构
        FftsGraphNode graphNode;
        graphNode.graphId = index;
        graphNode.predCnt = predCnt;
        graphNode.succList = succList;
        graphNode.isLastNode = (index == ctxNum - 1);

        auto contextType = ctx->contextType;
        switch (contextType) {
            case RT_CTX_TYPE_NOTIFY_WAIT:
                // NOTIFY_WAIT
                ParseFftsPlusNotifyCtx(ctx, stream, graphNode);
                break;
            case RT_CTX_TYPE_WRITE_VALUE:
                // NOTIFY_RECORD
                ParseFftsPlusWriteValueCtx(ctx, stream, graphNode);
                break;
            case RT_CTX_TYPE_SDMA:
                // MEMCPY and REDUCE
                ParseFftsPlusSdmaCtx(ctx, stream, graphNode);
                break;
            case RT_CTX_TYPE_LABEL:
                // FFTS_LABEL占位
                ParseFftsLabelCtx(stream, graphNode);
                break;
            default:
                HCCL_ERROR("contextType is invalid, contextType:%u.", contextType);
                break;
        }
    }

    return true;
}

rtError_t rtGeneralCtrl(uintptr_t *ctrl, uint32_t num, uint32_t type)
{
    std::cout << "FFTS模式开始下发SQE pid:" << getpid() << std::endl;
    if (ctrl == nullptr || num < 2 || type != RT_GNL_CTRL_TYPE_FFTS_PLUS) {
        HCCL_ERROR("rtGeneralCtrl failed to check FFTS+ mode input parameters.");
        return -RT_ERROR_NONE;
    }

    rtFftsPlusTaskInfo_t *task = reinterpret_cast<rtFftsPlusTaskInfo_t *>(ctrl[0]);
    rtStream_t stream = reinterpret_cast<rtStream_t>(ctrl[1]);
    if (task == nullptr || stream == nullptr) {
        return -RT_ERROR_NONE;
    }

    int ret = ParseFftsPlusTask(task, stream) ? RT_ERROR_NONE : -RT_ERROR_NONE;
    return ret;
}

rtError_t rtMemcpyAsyncWithoutCheckKind(void *dst, uint64_t destMax, const void *src, uint64_t cnt, rtMemcpyKind_t kind, rtStream_t stm)
{
    return rtMemcpyAsyncWithoutCheckKind(dst, destMax, src, cnt, kind, stm);
}

rtError_t rtUbDbSend(rtUbDbInfo_t *dbInfo, rtStream_t stm)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    int wqe_index = dbInfo->info[0].functionId;
    FakeWqe wqe = simRunnerMgr.GetFakeUb()->GetWqe(wqe_index);
    uint64_t notifyAddr = wqe.notifyAddr;
    int notifyId = simRunnerMgr.GetShmPoolMgr()->GetNotifyId(notifyAddr);

    if (wqe.type == WqeType::WRITE) {
        FakeSqe sqe{};
        sqe.type = FakeSqeType::NOTIFY_RECORD;
        sqe.notifyId = notifyId;
        LaunchSqeTask(*static_cast<int *>(stm), sqe);
    } else {
        std::cout << "[rtUbDbSend]wqe.type not support: " << static_cast<int>(wqe.type) << std::endl;
    }

    return RT_ERROR_NONE;
}

rtError_t rtUbDirectSend(rtUbWqeInfo_t *wqeInfo, rtStream_t stm)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    int wqe_index = wqeInfo->functionId;
    FakeWqe wqe = simRunnerMgr.GetFakeUb()->GetWqe(wqe_index);
    uint64_t notifyAddr = wqe.notifyAddr;
    int notifyId = simRunnerMgr.GetShmPoolMgr()->GetNotifyId(notifyAddr);

    if (wqe.type == WqeType::WRITE_WITH_NOTIFY) {
        // 1.先下发WRITE(MEM_CPY)
        FakeSqe sqe{};
        sqe.type = FakeSqeType::MEM_CPY;
        sqe.src = (void *)wqe.localAddr;
        sqe.dst = (void *)wqe.remoteAddr;
        sqe.count = wqe.size;
        LaunchSqeTask(*static_cast<int *>(stm), sqe);

        // 2.再下发Notify(NOTIFY_RECORD)
        sqe = {};
        sqe.type = FakeSqeType::NOTIFY_RECORD;
        sqe.notifyId = notifyId;
        LaunchSqeTask(*static_cast<int *>(stm), sqe);
    } else if (wqe.type == WqeType::REDUCE_WITH_NOTIFY) {
        // 1.先下发Reduce
        FakeSqe sqe{};
        sqe.type = FakeSqeType::SDMA_REDUCE;
        sqe.src = (void *)wqe.localAddr;
        sqe.dst = (void *)wqe.remoteAddr;
        sqe.count = wqe.size;
        sqe.dataType = RT_UB_REDUCE_DATA_TYPE_MAP[wqe.reduceDataType];
        sqe.reduceOp = RT_UB_REDUCE_OP_CODE_MAP[wqe.reduceOpType];
        LaunchSqeTask(*static_cast<int *>(stm), sqe);

        // 2.再下发Notify
        sqe = {};
        sqe.type = FakeSqeType::NOTIFY_RECORD;
        sqe.notifyId = notifyId;
        LaunchSqeTask(*static_cast<int *>(stm), sqe);
    } else {
        std::cout << "[rtUbDirectSend]wqe.type not support: " << static_cast<int>(wqe.type) << std::endl;
    }

    return RT_ERROR_NONE;
}

rtError_t rtWriteValue(rtWriteValueInfo_t * const info, rtStream_t const stm)
{
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyCreate(const int32_t deviceId, rtCntNotify_t *const cntNotify)
{
    *cntNotify = static_cast<void *>(SimRunnerMgr::GetInstance().GetFakeStreamMgr()->GetFakeNotifyMgr()->CreateNotify(deviceId));
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyRecord(rtCntNotify_t const inCntNotify, rtStream_t const stm,
                            const rtCntNtyRecordInfo_t * const info)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_RECORD;
    sqe.notifyId = *(static_cast<int *>(inCntNotify));
    sqe.notifyCnt = static_cast<int>(info->value);
    LaunchSqeTask(*static_cast<int *>(stm), sqe);
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyWaitWithTimeout(rtCntNotify_t const inCntNotify, rtStream_t const stm,
                                     const rtCntNtyWaitInfo_t * const info)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::NOTIFY_WAIT;
    sqe.notifyId = *(static_cast<int *>(inCntNotify));
    sqe.notifyCnt = static_cast<int>(info->value);
    LaunchSqeTask(*static_cast<int *>(stm), sqe);
    return RT_ERROR_NONE;
}

rtError_t rtReleaseDevResAddress(rtDevResInfo * const resInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyDestroy(rtCntNotify_t const inCntNotify)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetCntNotifyId(rtCntNotify_t inCntNotify, uint32_t * const notifyId)
{
    *notifyId = *(static_cast<int *>(inCntNotify));
    return RT_ERROR_NONE;
}

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void *devInfo)
{
    return RT_ERROR_NONE;
}

rtError_t rtCCULaunch(rtCcuTaskInfo_t *taskInfo, rtStream_t const stm)
{
    FakeSqe sqe{};
    sqe.type = FakeSqeType::CCU_SQE;
    sqe.ccuTaskInfo = *taskInfo;
    // sqe.devId = SimRunnerMgr::GetInstance().GetDeviceId();

    LaunchSqeTask(*static_cast<int *>(stm), sqe);
    return RT_ERROR_NONE;
}

aclError aclrtQueryEventWaitStatus(aclrtEvent event, aclrtEventWaitStatus *status)
{
    return ACL_SUCCESS;
}

rtError_t rtGetDevResAddress(rtDevResInfo *const resInfo, rtDevResAddrInfo *const addrInfo)
{
    if (resInfo == nullptr || (resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_NOTIFY_RECORD && resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR)) {
        // 非NotifyRecord场景暂不处理
        return RT_ERROR_NONE;
    }

    uint32_t notifyId = resInfo->resId;
    uint32_t len = 8;
    *(addrInfo->resAddress) = SimRunnerMgr::GetInstance().GetShmPoolMgr()->GetNotifyAddr(notifyId);
    *(addrInfo->len) = len;
    return RT_ERROR_NONE;
}

// 从llt_hccl_stub.cc挪来
aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback)
{
    return ACL_SUCCESS;
}

rtError_t rtsSetDeviceTaskAbortCallback(const char_t *regName, rtsDeviceTaskAbortCallback callback, void *args)
{
    return RT_ERROR_NONE;
}

rtError_t rtResourceClean(int32_t devId, rtIdType_t type)
{
    return RT_ERROR_NONE;
}

/**
 * @ingroup AscendCL
 * @brief get the number of available streams.
 * @param [out] streamCount   the number of available streams currently
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclrtGetStreamAvailableNum(uint32_t *streamCount)
{
    *streamCount = 1024;
    return ACL_SUCCESS;
}

aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtProcessReport(int32_t timeout)
{
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    return ACL_SUCCESS;
}

rtError_t rtEventCreateWithFlag(rtEvent_t* event, uint32_t flag)
{
    return HCCL_SUCCESS;
}

rtError_t rtStreamSwitchEx(void *ptr,  rtCondition_t condition, void *value_ptr,
                           rtStream_t true_stream, rtStream_t stream, rtSwitchDataType_t dataType)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceCount(int32_t *count)
{
    /*打桩函数先默认设备上芯片数量为8*/
    // *count = 8;
    *count = 2;  // 先测试双卡
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceMode(rtDeviceMode *deviceMode)
{
    return RT_ERROR_NONE;
}

aclError aclrtGetCurrentContext(aclrtContext *ctx)
{
    *ctx = nullptr;
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    return ACL_SUCCESS;
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode)
{
    return ACL_SUCCESS;
}

rtError_t rtMemcpyD2DAddrAsync(void *dst, uint64_t destMax, uint64_t destOffset, const void *src, uint64_t count,
                            uint64_t srcOffset, rtStream_t stream)
{
    return RT_ERROR_NONE;
}

aclError aclrtCreateEvent(aclrtEvent *event)
{
    return ACL_SUCCESS;
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t *eventId)
{
    *eventId = 0;
    return ACL_SUCCESS;
}

aclError aclrtDestroyEvent(aclrtEvent event)
{
    return ACL_SUCCESS;
}

aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    return ACL_SUCCESS;
}

aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus *status)
{
    return ACL_SUCCESS;
}

aclError aclrtNotifyBatchReset(aclrtNotify *notifies, size_t num)
{
    return ACL_SUCCESS;
}

rtError_t rtStreamClear(rtStream_t stm, rtClearStep_t step)
{
    return RT_ERROR_NONE;
}

rtError_t rtSetIpcMemorySuperPodPid(const char *name, u32 peerSdid, s32 peerPid[], int num)
{
    return aclrtIpcMemSetImportPid(name, peerPid, num);
}

rtError_t rtGetVisibleDeviceIdByLogicDeviceId(const int32_t logicDeviceId, int32_t *visibleDeviceId)
{
    *visibleDeviceId = logicDeviceId;
    return RT_ERROR_NONE;
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    if (maxCount == 321) {
        return -1;
    }
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    // if(moduleType == RT_MODULE_TYPE_SYSTEM && infoType == RT_INFO_TYPE_PHY_CHIP_ID) {
    //     *value = deviceId;
    // } else if ((moduleType == RT_MODULE_TYPE_AICORE || moduleType == RT_MODULE_TYPE_VECTOR_CORE) && infoType == INFO_TYPE_CORE_NUM) {
    //     *value = 32;
    // } else if (moduleType == RT_MODULE_TYPE_SYSTEM && infoType == INFO_TYPE_SDID ||
    //     infoType == INFO_TYPE_SERVER_ID || infoType == INFO_TYPE_SUPPER_POD_ID) {
    //     *value = 1;
    // } else {
    //     return 1;
    // }
    return  RT_ERROR_NONE;
}

rtError_t rtSetIpcNotifySuperPodPid(const char *name, u32 peerSdid, s32 peerPid)
{
    return rtSetIpcNotifyPid(name, &peerPid, 1);
}

rtError_t rtMemcpyEx(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind)
{
    rtError_t ret;
    ret = (rtError_t)memcpy_s(dst, count, src, count);
    if (ret) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                                     uint32_t funcMode)
{
    return RT_ERROR_NONE;
}

rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                   rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo) {
    return RT_ERROR_NONE;
}

aclError aclrtCreateContext(aclrtContext *ctx, int32_t deviceId)
{
    static int rtCtx = 0;
    *ctx = &rtCtx;
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext context)
{
    return ACL_SUCCESS;
}

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t *cqId, uint32_t *logicCqId)
{
    static uint32_t i = 0U;
    *logicCqId = i++;
    return RT_ERROR_NONE;
}

rtError_t rtCtxGetOverflowAddr(void **overflowAddr)
{
    *overflowAddr = (void *)0x1;
    return RT_ERROR_NONE;
}

rtError_t rtKernelGetAddrAndPrefCnt(void *hdl, const uint64_t tilingKey, const void * const stubFunc,
                                            const uint32_t flag, void **addr, uint32_t *prefetchCnt)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetDevArgsAddr(rtStream_t stm, rtArgsEx_t *argsInfo, void **devArgsAddr, void **argsHandle)
{
    return RT_ERROR_NONE;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMapMem(void *virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)offset;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtUnmapMem(void *virPtr)
{
    (void)virPtr;
    return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(
    aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags, uint64_t *shareableHandle)
{
    (void)handle;
    (void)handleType;
    (void)flags;
    (void)shareableHandle;
    return ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle *handle)
{
    (void)shareableHandle;
    (void)deviceId;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t *pid, size_t pidNum)
{
    (void)shareableHandle;
    (void)pid;
    (void)pidNum;
    return ACL_SUCCESS;
}

aclError aclrtReserveMemAddress(void **virPtr, size_t size, size_t alignment, void *expectPtr, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    return ACL_SUCCESS;
}

aclError aclrtReleaseMemAddress(void *virPtr)
{
    (void)virPtr;
    return ACL_SUCCESS;
}

int rtModelFake = 0;
rtError_t rtStreamGetCaptureInfo(rtStream_t stm, rtStreamCaptureStatus *status, rtModel_t *captureMdl)
{
    *captureMdl = &rtModelFake;
    return RT_ERROR_NONE;
}

rtError_t rtModelGetId(rtModel_t mdl, uint32_t *modelId)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    captureMdl = &rtModelFake;
    return RT_ERROR_NONE;
}

rtError_t rtIpcSetMemoryAttr(const char *name, uint32_t type, uint64_t attr)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *value)
{
    return rtGetPairPhyDevicesInfo(devId, otherDevId, infoType, value);
}

rtError_t rtGetPairPhyDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *value)
{
    DevType deviceType;
    HcclResult ret = hrtGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {

        return -RT_ERROR_NONE;
    }
    if (deviceType == DevType::DEV_TYPE_910B) {

        if ((devId / 8) != (otherDevId / 8)) {
            *value = 1;  // PXI
        } else {
            *value = 0;  // HCCS
        }
        return RT_ERROR_NONE;
    }
    if (deviceType == DevType::DEV_TYPE_910_93) {
        // 0-1 2-3 4-5 6-7
        if ((abs(static_cast<int>(devId) - static_cast<int>(otherDevId)) == 1) && ((devId + otherDevId) % 4 == 1)) {
            *value = 5;  // SIO
        } else {
            *value = 6;  // HCCS_SW
        }
        return RT_ERROR_NONE;
    }

    if ((devId / 4) != (otherDevId / 4))  // 若当前为标卡/虚拟机/非同一clustor
    {
        *value = 1;  // PXI
    } else {
        *value = 0;  // HCCS
    }

    return RT_ERROR_NONE;
}

rtError_t rtThreadExchangeCaptureMode(rtStreamCaptureMode *mode)
{
    return RT_ERROR_NONE;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t *value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtMallocWithCfg(void **devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig *cfg)
{
    return ACL_SUCCESS;
}

const char *aclrtGetSocName()
{
    if (strlen(GetFakeSocVersionStub()) >= 32) {
        throw std::exception();
    }
    return GetFakeSocVersionStub();
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode)
{
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t *value)
{
    *value = 1;
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromData(const void *data, size_t length,
    const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void **aicAddr, void **aivAddr)
{
    return ACL_SUCCESS;
}
#endif