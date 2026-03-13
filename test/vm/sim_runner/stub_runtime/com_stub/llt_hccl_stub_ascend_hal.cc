#include "ascend_hal.h"
#include "securec.h"
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <map>
#include <cstring>
#include "hccl_sim_comm_stub.h"
#include "../rts_stub/SimRunnerMgr.h"
#include "./device/hccl_sim_aicpu_sqe.h"

#define PLAT_COMBINE(arch, chip, ver) (((arch) << 16U) | ((chip) << 8U) | (ver))

drvError_t halEschedAttachDevice(unsigned int devId)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedDettachDevice(unsigned int devId)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrpEx(unsigned int devId, struct esched_grp_para *grpPara, unsigned int *grpId)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubscribeEvent(
    unsigned int devId, unsigned int grpId, unsigned int threadId, unsigned long long eventBitmap)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedQueryInfo(
    unsigned int devId, ESCHED_QUERY_TYPE type, struct esched_input_info *input, struct esched_output_info *output)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedWaitEvent(
    unsigned int devId, unsigned int grpId, unsigned int threadId, int timeout, struct event_info *event)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedRegisterAckFunc(unsigned int grpId, EVENT_ID event_id,
    void (*ackFunc)(unsigned int devId, unsigned int subevent_id, char *msg, unsigned int msgLen))
{
    return DRV_ERROR_NONE;
}

drvError_t drvGetDevNum(uint32_t *num)
{
    *num = 1;
    return DRV_ERROR_NONE;
}

drvError_t halGetChipInfo(uint32_t devId, halChipInfo *chipInfo)
{
    auto socVersion = GetFakeSocVersionStub();
    strncpy((char *)chipInfo->name, socVersion, MAX_CHIP_NAME - 1);
    return DRV_ERROR_NONE;
}

drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr)
{
    *dstPtr = srcPtr;
    return DRV_ERROR_NONE;
}

drvError_t halHostUnregister(void *hostPtr, uint32_t devid)
{
    return DRV_ERROR_NONE;
}

drvError_t halMemCtl(int type, void *param_value, size_t param_value_size, void *out_value, size_t *out_size_ret)
{
    if (param_value_size >= sizeof(supportFeaturePara)) {
        supportFeaturePara *ptr = static_cast<supportFeaturePara *>(out_value);
        ptr->support_feature = CTRL_SUPPORT_PCIE_BAR_MEM_MASK;
    }
    return DRV_ERROR_NONE;
}

drvError_t drvGetPlatformInfo(uint32_t *info)
{
    *info = 0;
    return DRV_ERROR_NONE;
}

drvError_t drvQueryProcessHostPid(
    int pid, unsigned int *chip_id, unsigned int *vfid, unsigned int *host_pid, unsigned int *cp_type)
{
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value)
{
    *value = PLAT_COMBINE(0, 5, 0);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEvent(unsigned int devId, struct event_summary *event)
{
    return DRV_ERROR_NONE;
}

drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType)
{
    return DRV_ERROR_NONE;
}

drvError_t halGetAPIVersion(int *halAPIVersion)
{
    if (halAPIVersion == nullptr) {
        return DRV_ERROR_INVALID_VALUE;
    }
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeRegister(uint32_t devId, struct halSensorNodeCfg *cfg, uint64_t *handle)
{
    *handle = 1;
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUnregister(uint32_t devId, uint64_t handle)
{
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUpdateState(uint32_t devId, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceGetPhyIdByIndex(unsigned int deviceLogicId, unsigned int *devicePhyId)
{
    return DRV_ERROR_NONE;
}

#define MEMCPY_SIZE_MAX (2 * 1024 * 1024 * 1024UL - 1)
drvError_t drvMemcpy(DVdeviceptr dst, size_t destMax, DVdeviceptr src, size_t ByteCount)
{
    int ret = 0;
    int dstSize, srcSize;
    size_t tmp;

    if (!dst || !src || (ByteCount == 0) || (destMax == 0)) {
        return DRV_ERROR_INVALID_VALUE;
    }

    while ((ByteCount) && (ret == 0)) {
        if (ByteCount <= MEMCPY_SIZE_MAX) {
            dstSize = destMax;
            srcSize = ByteCount;
            ret = memcpy_s((void *)dst, dstSize, (void *)src, srcSize);
            tmp = ByteCount;
        } else {
            ret = memcpy_s((void *)dst, MEMCPY_SIZE_MAX, (void *)src, MEMCPY_SIZE_MAX);
            dst += MEMCPY_SIZE_MAX;
            src += MEMCPY_SIZE_MAX;
            tmp = MEMCPY_SIZE_MAX;
        }

        ByteCount -= tmp;
        destMax -= tmp;
    }

    if (ret) {
        return DRV_ERROR_INVALID_HANDLE;
    }

    return DRV_ERROR_NONE;
}

// 删除会引起段错误
int halGrpQuery(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff, unsigned int *outLen)
{
    if (cmd == GRP_QUERY_GROUPS_OF_PROCESS) {
        GroupQueryOutput *outTmpBuf = reinterpret_cast<GroupQueryOutput *>(outBuff);
        *outLen = sizeof(GrpQueryGroupsOfProcInfo);
        strcpy(outTmpBuf->grpQueryGroupsOfProcInfo[0].groupName, "test");
    } else {
        *outLen = 0;
    }
    return DRV_ERROR_NONE;
}

pid_t drvDeviceGetBareTgid(void)
{
    return getpid();
}

drvError_t halResourceIdCheck(struct drvResIdKey *info)
{
    return DRV_ERROR_NONE;
}

drvError_t halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo *info)
{
    if (info == nullptr) {
        return DRV_ERROR_INNER_ERR;
    }
    int head = 0;
    switch (info->prop) {
        case DRV_SQCQ_PROP_SQ_HEAD: {
            info->value[0] = head;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_DEPTH: {
            info->value[0] = hccl::HCCL_SQE_MAX_CNT;  // 2048
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_TAIL: {
            info->value[0] = head;  // 代表sqBuffer一直为空
            return DRV_ERROR_NONE;
        };
        case DRV_SQCQ_PROP_SQ_BASE: {
            uint8_t *buffer = SimRunnerMgr::GetInstance().GetFakeStreamMgr()->GetSqBufferAddr();
            info->value[0] = reinterpret_cast<uintptr_t>(buffer) & 0xFFFFFFFF;
            info->value[1] = reinterpret_cast<uintptr_t>(buffer) >> 32;
        }
        default:
            return DRV_ERROR_NONE;
    }
}

drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo *info)
{
    if (info->prop == DRV_SQCQ_PROP_SQ_TAIL) {
        auto socVersion = GetFakeSocVersionStub();
        if (strcmp(socVersion, "Ascend910_9391") == 0) {
            CopyA3SqBufferStub(devId, info);
        } else if (strcmp(socVersion, "Ascend910_9591") == 0) {
            CopyA5SqBufferStub(devId, info);
        }
    }
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdInfoGet(struct drvResIdKey *key, drvResIdProcType type, uint64_t *value)
{
    return DRV_ERROR_NONE;
}

drvError_t halTsdrvCtl(uint32_t devId, int cmd, void *param, size_t paramSize, void *out, size_t *outSize)
{
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdRestore(struct drvResIdKey *info)
{
    return DRV_ERROR_NONE;
}