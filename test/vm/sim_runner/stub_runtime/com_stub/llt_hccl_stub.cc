#include <securec.h>
#include <map>
#include "llt_hccl_stub.h"
#include "adapter_rts.h"
#include "adapter_hal.h"
#include "aicpu_sharder.h"  // aicpu sharding
#include "tsd/tsd_client.h"

/*记录当前操作的设备*/
__thread int32_t current_dev;
__thread int32_t current_die;
__thread rtDeviceMode current_devMode;

typedef struct TraceAttr {
    bool exitSave;  // exec save when AtraceDestroy
} TraceAttr;

typedef struct TraceGlobalAttr {
    uint8_t saveMode;  // 0: local save; 1: send to remote and save
    uint8_t deviceId;  // 0: default; 32~63:vf
    uint32_t pid;      // 0: default; if saveMode=1, means host pid
    uint8_t reserve[32];
} TraceGlobalAttr;

typedef enum TracerType {
    TRACER_TYPE_SCHEDULE = 0,
    TRACER_TYPE_PROGRESS = 1,
    TRACER_TYPE_STATISTICS = 2,
    TRACER_TYPE_MAX,
} TracerType;

#ifdef __cplusplus
extern "C" {
#endif

int32_t QueryPartitionMapPsId(uint64_t key, uint32_t *psId)
{
    *psId = 0;
    return 0;
}

int32_t InitPartitionMap(uint32_t partitionNum, uint32_t psNum, const uint32_t psId[])
{
    return 0;
}

int32_t GetBatchPsIds(uint64_t *keys, uint32_t *psIds[], uint32_t num)
{
    for (int i = 0; i < num; i++) {
        (*psIds)[i] = 0;
    }
    return 0;
}

HcclResult AtraceSubmit(int32_t handle, const void *buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

void AtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

HcclResult UtraceSubmit(int32_t handle, const void *buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

void UtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

int32_t AtraceCreateWithAttr(int32_t tracerType, const char *objName, const TraceAttr *attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

int32_t UtraceCreateWithAttr(int32_t tracerType, const char *objName, const TraceAttr *attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

int32_t UtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    return 0;
}

int32_t UtraceSetGlobalAttr(const TraceGlobalAttr *attr)
{
    (void)(attr);
    return 0;
}

int ibv_get_cq_event_stub(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cq_context)
{
    if (!channel) {
        return -1;
    }
    return 0;
}

void ibv_ack_cq_events_stub(struct ibv_cq *cq, unsigned int nevents)
{
    return;
}

void ibv_query_qp_stub(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr)
{
    return;
}

int ibv_ext_post_send_stub(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr,
    struct ibv_post_send_ext_attr *ext_attr, struct ibv_post_send_ext_resp *ext_resp)
{
    return 0;
}

namespace hccl
{
std::map<std::string, void *> dlRaFuntionPtrMap = {
    {"ra_qp_create", (void *)&ra_qp_create},
    {"ra_get_qp_context", (void *)&ra_get_qp_context},
    {"ra_get_tsqp_depth", (void *)&ra_get_tsqp_depth},
    {"ra_set_tsqp_depth", (void *)&ra_set_tsqp_depth},
    {"ra_qp_destroy", (void *)&ra_qp_destroy},
    {"ra_qp_connect_async", (void *)&ra_qp_connect_async},
    {"ra_get_qp_status", (void *)&ra_get_qp_status},
    {"ra_deinit", (void *)&ra_deinit},
    {"ra_get_notify_base_addr", (void *)&ra_get_notify_base_addr},
    {"ra_get_sockets", (void *)&ra_get_sockets},
    {"ra_init", (void *)&ra_init},
    {"ra_is_first_used", (void *)&ra_is_first_used},
    {"ra_is_last_used", (void *)&ra_is_last_used},
    {"ra_mr_dereg", (void *)&ra_mr_dereg},
    {"ra_mr_reg", (void *)&ra_mr_reg},
    {"ra_register_mr", (void *)&ra_register_mr},
    {"ra_deregister_mr", (void *)&ra_deregister_mr},
    {"ra_rdev_deinit", (void *)&ra_rdev_deinit},
    {"ra_rdev_init", (void *)&ra_rdev_init},
    {"ra_rdev_init_v2", (void *)&ra_rdev_init_v2},
    {"ra_send_wr", (void *)&ra_send_wr},
    {"ra_send_wrlist", (void *)&ra_send_wrlist},
    {"ra_send_wrlist_ext", (void *)&ra_send_wrlist_ext},
    {"ra_socket_batch_close", (void *)&ra_socket_batch_close},
    {"ra_socket_batch_connect", (void *)&ra_socket_batch_connect},
    {"ra_socket_deinit", (void *)&ra_socket_deinit},
    {"ra_socket_init", (void *)&ra_socket_init},
    {"ra_socket_init_v1", (void *)&ra_socket_init_v1},
    {"ra_socket_listen_start", (void *)&ra_socket_listen_start},
    {"ra_socket_listen_stop", (void *)&ra_socket_listen_stop},
    {"ra_socket_recv", (void *)&ra_socket_recv},
    {"ra_socket_send", (void *)&ra_socket_send},
    {"ra_socket_set_white_list_status", (void *)&ra_socket_set_white_list_status},
    {"ra_socket_get_white_list_status", (void *)&ra_socket_get_white_list_status},
    {"ra_socket_white_list_add", (void *)&ra_socket_white_list_add},
    {"ra_socket_white_list_del", (void *)&ra_socket_white_list_del},
    {"ra_get_ifnum", (void *)&ra_get_ifnum},
    {"ra_get_ifaddrs", (void *)&ra_get_ifaddrs},
    {"ra_get_interface_version", (void *)&ra_get_interface_version},
    {"ra_epoll_ctl_add", (void *)&ra_epoll_ctl_add},
    {"ra_epoll_ctl_mod", (void *)&ra_epoll_ctl_mod},
    {"ra_epoll_ctl_del", (void *)&ra_epoll_ctl_del},
    {"ra_set_tcp_recv_callback", (void *)&ra_set_tcp_recv_callback},
    {"ra_cq_create", (void *)&ra_cq_create},
    {"ra_cq_destroy", (void *)&ra_cq_destroy},
    {"ra_normal_qp_create", (void *)&ra_normal_qp_create},
    {"ra_normal_qp_destroy", (void *)&ra_normal_qp_destroy},
    {"ra_set_qp_attr_qos", (void *)&ra_set_qp_attr_qos},
    {"ra_set_qp_attr_timeout", (void *)&ra_set_qp_attr_timeout},
    {"ra_set_qp_attr_retry_cnt", (void *)&ra_set_qp_attr_retry_cnt},
    {"ra_create_comp_channel", (void *)&ra_create_comp_channel},
    {"ra_destroy_comp_channel", (void *)&ra_destroy_comp_channel},
    {"ra_get_cqe_err_info", (void *)&ra_get_cqe_err_info},
    {"ra_create_srq", (void *)&ra_create_srq},
    {"ra_destroy_srq", (void *)&ra_destroy_srq},
    {"ra_qp_create_with_attrs", (void *)&ra_qp_create_with_attrs},
    {"ra_ai_qp_create", (void *)&ra_ai_qp_create},
    {"ra_send_wr_v2", (void *)&ra_send_wr_v2},
    {"ra_poll_cq", (void *)&ra_poll_cq},
    {"ra_recv_wrlist", (void *)&ra_recv_wrlist},
    {"ra_socket_get_vnic_ip_infos", (void *)&ra_socket_get_vnic_ip_infos},
    {"ra_rdev_get_support_lite", (void *)&ra_rdev_get_support_lite},
    {"ra_create_event_handle", (void *)&ra_create_event_handle},
    {"ra_ctl_event_handle", (void *)&ra_ctl_event_handle},
    {"ra_wait_event_handle", (void *)&ra_wait_event_handle},
    {"ra_destroy_event_handle", (void *)&ra_destroy_event_handle},
    {"ra_qp_batch_modify", (void *)&ra_qp_batch_modify},
    {"ra_get_notify_mr_info", (void *)&ra_get_notify_mr_info},
    {"ra_rdev_init_with_backup", (void *)&ra_rdev_init_with_backup},
    {"ra_rdev_get_cqe_err_info_list", (void *)ra_rdev_get_cqe_err_info_list},
    {"ra_typical_qp_create", (void *)&ra_typical_qp_create},
    {"ra_typical_qp_modify", (void *)&ra_typical_qp_modify},
    {"ra_typical_send_wr", (void *)&ra_typical_send_wr},
    {"ra_rdev_get_port_status", (void *)&ra_rdev_get_port_status},
    {"ra_get_qp_attr", (void*)&ra_get_qp_attr},
    {"ra_socket_batch_abort", (void*)&ra_socket_batch_abort}
};

std::map<std::string, void *> dlTdtFuntionPtrMap = {
    {"TsdOpen", (void *)&TsdOpen},
    {"TsdProcessOpen", (void *)&TsdProcessOpen},
    {"ProcessCloseSubProcList", (void *)&ProcessCloseSubProcList},
    {"TsdCapabilityGet", (void*)&TsdCapabilityGet}
};

std::map<std::string, void*> dlHalFuntionPtrMap = {
    {"halEschedSubmitEvent", (void*)&halEschedSubmitEvent},
    {"halEschedAttachDevice", (void*)&halEschedAttachDevice},
    {"halEschedDettachDevice", (void*)&halEschedDettachDevice},
    {"halEschedCreateGrp", (void*)&halEschedCreateGrp},
    {"halEschedCreateGrpEx", (void*)&halEschedCreateGrpEx},
    {"halEschedSubscribeEvent", (void*)&halEschedSubscribeEvent},
    {"halEschedWaitEvent", (void*)&halEschedWaitEvent},
    {"halEschedRegisterAckFunc", (void*)&halEschedRegisterAckFunc},
    {"drvMemcpy", (void*)&drvMemcpy},
    {"drvDeviceGetBareTgid", (void*)&drvDeviceGetBareTgid},
    {"halGrpQuery", (void*)&halGrpQuery},
    {"drvGetDevNum", (void*)&drvGetDevNum},
    {"halGetDeviceInfo", (void*)&halGetDeviceInfo},
    {"halEschedQueryInfo", (void*)&halEschedQueryInfo},
    {"drvGetPlatformInfo", (void*)&drvGetPlatformInfo},
    {"halGetChipInfo", (void*)&halGetChipInfo},
    {"halBindCgroup", (void*)&halBindCgroup},
    {"drvDeviceGetPhyIdByIndex", (void*)&drvDeviceGetPhyIdByIndex},
    {"halHostRegister", (void*)&halHostRegister},
    {"halHostUnregister", (void*)&halHostUnregister},
    {"halMemCtl", (void*)&halMemCtl},
    {"halGetAPIVersion", (void*)&halGetAPIVersion},
    {"halSensorNodeRegister", (void *)&halSensorNodeRegister},
    {"halSensorNodeUnregister", (void *)&halSensorNodeUnregister},
    {"halSensorNodeUpdateState", (void *)&halSensorNodeUpdateState},
    {"drvQueryProcessHostPid", (void *)&drvQueryProcessHostPid}
};

std::map<std::string, void*> dlIbvFuntionPtrMap = {
    {"ibv_get_cq_event", (void*)&ibv_get_cq_event_stub},
    {"ibv_ack_cq_events", (void*)&ibv_ack_cq_events_stub},
    {"ibv_query_qp", (void*)&ibv_query_qp_stub}
};

std::map<std::string, void*> dlrdmaFuntionPtrMap = {
    {"ibv_ext_post_send", (void*)&ibv_ext_post_send_stub}
};

std::map<std::string, void*> dlHddsFuntionPtrMap = {
    {"QueryPartitionMapPsId", (void*)&QueryPartitionMapPsId},
    {"InitPartitionMap", (void*)&InitPartitionMap},
    {"GetBatchPsIds", (void*)&GetBatchPsIds}
};

std::map<std::string, void*> dlProfFuntionPtrMap = {
    {"MsprofRegisterCallback", (void*)&MsprofRegisterCallback},
    {"MsprofRegTypeInfo", (void*)&MsprofRegTypeInfo},
    {"MsprofReportApi", (void*)&MsprofReportApi},
    {"MsprofReportCompactInfo", (void*)&MsprofReportCompactInfo},
    {"MsprofReportAdditionalInfo", (void*)&MsprofReportAdditionalInfo},
    {"MsprofStr2Id", (void*)&MsprofStr2Id},
    {"MsprofSysCycleTime", (void*)&MsprofSysCycleTime}
};

std::map<std::string, void*> dlAtraceFuntionPtrMap = {
    {"AtraceDestroy", (void*)&AtraceDestroy},
    {"AtraceSubmit", (void*)&AtraceSubmit},
    {"AtraceCreateWithAttr", (void*)&AtraceCreateWithAttr}
};

std::map<std::string, void*> dlUtraceFuntionPtrMap = {
    {"UtraceDestroy", (void*)&UtraceDestroy},
    {"UtraceSubmit", (void*)&UtraceSubmit},
    {"UtraceCreateWithAttr", (void*)&UtraceCreateWithAttr},
    {"UtraceSetGlobalAttr", (void*)&UtraceSetGlobalAttr},
    {"UtraceSave", (void*)&UtraceSave}
};

static int dlRaHandle;
static int dlTdtHandle;
static int dlHalHandle;
static int dlIbvHandle;
static int dlHddsHandle;
static int dlqosHandle;
static int dlProfHandle;
static int dlAtraceHandle;
static int dlUtraceHandle;
static int dlHnsRdmav25Handle;
void* __HcclDlopenSub(const char *libName, int mode)
{
    HCCL_INFO("run dlopen(const char*[%s], int[%d])", libName, mode);
    std::string LibName(libName);
    if (LibName == "libra.so") {
        return &dlRaHandle;
    } else if (LibName == "libtsdclient.so" ) {
        return &dlTdtHandle;
    } else if (LibName == "libascend_hal.so" ) {
        return &dlHalHandle;
    } else if (LibName == "libibverbs.so" ) {
        return &dlIbvHandle;
    } else if (LibName == "libhdds_base.so") {
        return &dlHddsHandle;
    } else if (LibName == "libqos_manager.so") {
        return &dlqosHandle;
    } else if (LibName == "libprofapi.so") {
        return &dlProfHandle;
    } else if (LibName == "libascend_trace.so") {
        return &dlAtraceHandle;
    } else if (LibName == "libutrace.so") {
        return &dlUtraceHandle;
    } else if (LibName == "libhns-rdmav25.so") {
        return &dlHnsRdmav25Handle;
    } else if (LibName == "libruntime.so") {
        return reinterpret_cast<void *>(0x12345678);
    }
    return nullptr;
}

int __HcclDlcloseSub(void* handle)
{
    HCCL_INFO("run dlclose");
    handle = nullptr;
    return 0;
}

int ra_common_empty_func()
{
    return 0;
}

void* getRaFuncStubPtr(std::string funcName)
{
    if (dlRaFuntionPtrMap.count(funcName) == 0) {
        // 新增但未添加到Map的函数，全部按照空桩处理
        return (void *)&ra_common_empty_func;
    }

    return dlRaFuntionPtrMap[funcName];
}

void* __HcclDlsymSub(void* handle, const char* funcName)
{
    std::string tempName(funcName);
    if (handle == &dlRaHandle) {
        return getRaFuncStubPtr(tempName);
    } else if(handle == &dlTdtHandle) {
        return dlTdtFuntionPtrMap[tempName];
    } else if(handle == &dlHalHandle) {
        return dlHalFuntionPtrMap[tempName];
    } else if(handle == &dlIbvHandle) {
        return dlIbvFuntionPtrMap[tempName];
    } else if(handle == &dlHddsHandle) {
        return dlHddsFuntionPtrMap[tempName];
    } else if(handle == &dlqosHandle) {
        return nullptr;
    } else if(handle == &dlProfHandle) {
        return dlProfFuntionPtrMap[tempName];
    } else if (handle == &dlAtraceHandle) {
        return dlAtraceFuntionPtrMap[tempName];
    } else if(handle == &dlUtraceHandle) {
        return dlUtraceFuntionPtrMap[tempName];
    } else if(handle == &dlHnsRdmav25Handle) {
        return dlrdmaFuntionPtrMap[tempName];
    }
    return nullptr;
}
strong_alias(__HcclDlopenSub, HcclDlopen);
strong_alias(__HcclDlcloseSub, HcclDlclose);
strong_alias(__HcclDlsymSub, HcclDlsym);
}

const std::unordered_map<std::string, DevType> SOC_VER_CONVERT{
    {"Ascend310P1", DevType::DEV_TYPE_310P1},
    {"Ascend310P3", DevType::DEV_TYPE_310P3},
    {"Ascend910", DevType::DEV_TYPE_910},
    {"Ascend910A", DevType::DEV_TYPE_910},
    {"Ascend910B", DevType::DEV_TYPE_910},
    {"Ascend910ProA", DevType::DEV_TYPE_910},
    {"Ascend910ProB", DevType::DEV_TYPE_910},
    {"Ascend910PremiumA", DevType::DEV_TYPE_910},
    {"Ascend910B1", DevType::DEV_TYPE_910B},
    {"Ascend910B2", DevType::DEV_TYPE_910B},
    {"Ascend910B2C", DevType::DEV_TYPE_910B},
    {"Ascend910B3", DevType::DEV_TYPE_910B},
    {"Ascend910B4", DevType::DEV_TYPE_910B},
    {"Ascend910_9391", DevType::DEV_TYPE_910_93},
    {"Ascend910_9381", DevType::DEV_TYPE_910_93},
    {"Ascend910_9372", DevType::DEV_TYPE_910_93},
    {"Ascend910_9591", DevType::DEV_TYPE_910_95},
    {"nosoc", DevType::DEV_TYPE_NOSOC}
};

HcclResult __hrtGetDeviceTypeStub(DevType &devType)
{
#ifndef HCCD
    s8 targetChipVer[CHIP_VERSION_MAX_LEN] = {0};
    rtGetSocVersion(reinterpret_cast<char *>(targetChipVer), CHIP_VERSION_MAX_LEN);
    auto iter = SOC_VER_CONVERT.find(reinterpret_cast<char *>(targetChipVer));
    if (iter == SOC_VER_CONVERT.end()) {
        HCCL_ERROR("[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get illegal chipver, chip_ver[%s].", \
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), targetChipVer);
        return HCCL_E_RUNTIME;
    }
    devType = iter->second;
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDeviceType]The helper does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
strong_alias(__hrtGetDeviceTypeStub, hrtGetDeviceType);

HcclResult __hrtGetDeviceStub(s32 *deviceLogicId)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(deviceLogicId);

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        *deviceLogicId = 0;
        return HCCL_SUCCESS;
    }
    aclError ret = 0;
    ret = aclrtGetDevice(deviceLogicId);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_WARNING("[Get][Device]errNo[0x%016llx] rtGet device fail, "\
        "please make sure that device is set. return[%d], para:deviceLogicId[%d]",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *deviceLogicId), HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDevice]The helper does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
strong_alias(__hrtGetDeviceStub, hrtGetDevice);

HcclResult __hrtGetDevicePhyIdByIndexStub(u32 deviceLogicId, u32 &devicePhyId, bool isRefresh)
{
#ifndef HCCD
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        devicePhyId = 0;
        return HCCL_SUCCESS;
    }

    rtError_t ret = rtGetDevicePhyIdByIndex(deviceLogicId, &devicePhyId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[Get][DevicePhyId]errNo[0x%016llx] rtGet device PhyId by index failed, return[%d], "\
            "para: devIndex[%u], phyId[%u]", HCCL_ERROR_CODE(HCCL_E_DRV), ret, deviceLogicId, devicePhyId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDevicePhyIdByIndex]The helper does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
strong_alias(__hrtGetDevicePhyIdByIndexStub, hrtGetDevicePhyIdByIndex);
#ifdef __cplusplus
}  // extern "C"
#endif

uint32_t GetCPUNum()
{
    return 1;
}

void ParallelFor(int64_t total, int64_t perUnitSize, const aicpu::SharderWork &work)
{
    return;
}
 
namespace Adx {
void AdumpPrintWorkSpace(
    const void *workSpaceAddr, const size_t dumpWorkSpaceSize, rtStream_t stream, const char *opType, bool enableSync)
{
    return;
}
};  // namespace Adx