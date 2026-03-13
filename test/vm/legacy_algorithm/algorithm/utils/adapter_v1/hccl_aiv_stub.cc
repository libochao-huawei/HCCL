#include <unordered_map>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "hccl_common.h"
#include "hccl_aiv.h"
#include "aiv_base_stub.h"
#include "mem_layout.h"
#include "aiv_task_queue_stub.h"

using namespace AscendC;
using namespace checker;

namespace hccl {
#define GM_ADDR uint8_t*

#define KERNEL_ARGS_DEF \
GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, \
GM_ADDR buffIn4, GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, \
GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11, \
GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, \
GM_ADDR buffOut0, GM_ADDR buffOut1, GM_ADDR buffOut2, GM_ADDR buffOut3, \
GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7, \
GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, \
GM_ADDR buffOut12, GM_ADDR buffOut13, GM_ADDR buffOut14, GM_ADDR buffOut15, \
GM_ADDR input, GM_ADDR output, uint32_t rank, uint32_t rankSize, uint64_t len, \
uint32_t dataType, uint32_t reduceOp, uint32_t root, int32_t tag, bool isOpBase, uint64_t bufferSize, \
int32_t aivRdmaStep, bool useAivRdmaSmall, int32_t serverNum, uint32_t devType, GM_ADDR headCountMem, \
GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter, uint32_t deterministic

#define KERNEL_ARGS_DEF_A3 \
GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffOut0, GM_ADDR buffOut1, GM_ADDR bufferSize, \
GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, GM_ADDR isEnableCounter, \
GM_ADDR input, GM_ADDR output, uint32_t rank, uint32_t rankSize, uint64_t len, \
uint32_t dataType, uint32_t reduceOp, uint32_t root, int32_t tag, bool isOpBase, \
int32_t serverNum, uint32_t devType, uint32_t deterministic

constexpr uint32_t SIZE_OF_INT32 = 4;

#define EXTERN_KERNEL_ARGS_DEF \
KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define EXTERN_KERNEL_ARGS_DEF_V2 \
KERNEL_ARGS_DEF, ExtraArgsV2 extraArgs

typedef void (*aivFunc)(KERNEL_ARGS_DEF);
typedef void (*aivFuncExtra)(EXTERN_KERNEL_ARGS_DEF);
typedef void (*aivFuncExtraV2)(EXTERN_KERNEL_ARGS_DEF_V2);
typedef void (*aivFuncExtraA3)(KERNEL_ARGS_DEF_A3);

enum class KernelLaunchMode {
    LAUNCH_MODE_ARGS_BASE = 0,  // Launch模式，基础参数
    LAUNCH_MODE_ARGS_EXTRA,     // Launch模式，基础参数+ExtraArgs
    LAUNCH_MODE_ARGS_EXTRA_V2,  // Launch模式，基础参数+ExtraArgsV2
    LAUNCH_MODE_ARGS_EXTRA_A3   // Launch模式，A3跨机
};

using AivKernelInfo = struct AivKernelInfoDef {
    const char* kernelName;
    HcclCMDType cmdType;
    HcclDataType dataType;
    KernelArgsType argsType;

    AivKernelInfoDef(const char* kernelName, HcclCMDType cmdType, HcclDataType dataType,
        KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
        : kernelName(kernelName), cmdType(cmdType), dataType(dataType), argsType(argsType)
    {
    }
};

static std::vector<AivKernelInfo> g_aivKernelInfoList = {
    // allreduce
    {"aiv_all_reduce_float", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_reduce_half", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_reduce_int16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_reduce_int32_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_reduce_int8_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_reduce_bfloat16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_reduce_cn_float", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_half", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_int16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_int32_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_int8_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_reduce_cn_bfloat16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    // alltoall alltoallvc
    {"aiv_all_to_all_vc_half", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_vc_int16_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_vc_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_vc_float", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_vc_int32_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_vc_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_vc_int8_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_vc_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_vc_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // alltoallv
    {"aiv_all_to_all_v_half", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_v_int16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_v_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_v_float", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_v_int32_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_v_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_v_int8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_v_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_v_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // alltoallv a3
    {"aiv_all_to_all_v_sp_half",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int16_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint16_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_float",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int32_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint32_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int8_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint8_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT8, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_bfloat16_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    // alltoall
    {"aiv_all_to_all_half", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_int16_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_float", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_int32_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_int8_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // reducescatter
    {"aiv_reduce_scatter_float", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_half", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_int16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_int32_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_int8_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_reduce_scatter_cn_float", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_half", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_int16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_int32_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_int8_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_reduce_scatter_cn_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    // reducescatterv
    {"aiv_reduce_scatter_v_float", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_v_half", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_v_int16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_v_int32_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_v_int8_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_v_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_BFP16},
     // allgather
    {"aiv_all_gather_half", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_int16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_uint16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_float", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_int32_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_uint32_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_int8_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_uint8_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_bfloat16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_gather_cn_half", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT16, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_float", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int32_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint32_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT32, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_int8_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_uint8_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT8, KernelArgsType::ARGS_TYPE_SIMPLE},
    {"aiv_all_gather_cn_bfloat16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SIMPLE},
     // allgatherv
    {"aiv_all_gather_v_half", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_v_int16_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_v_uint16_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_v_float", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_v_int32_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_v_uint32_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_v_int8_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_v_uint8_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_v_bfloat16_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // broadcast
    {"aiv_broadcast_half", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_broadcast_int16_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_broadcast_uint16_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_broadcast_float", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_broadcast_int32_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_broadcast_uint32_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_broadcast_int8_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_broadcast_uint8_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_broadcast_bfloat16_t", HcclCMDType::HCCL_CMD_BROADCAST, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // 同步
    {"hccl_aiv_sync", HcclCMDType::HCCL_CMD_INVALID, HcclDataType::HCCL_DATA_TYPE_RESERVED},
    {"hccl_aiv_sync_cn", HcclCMDType::HCCL_CMD_INVALID, HcclDataType::HCCL_DATA_TYPE_RESERVED, KernelArgsType::ARGS_TYPE_SIMPLE},
};

extern "C" {
    extern void aiv_all_reduce_float(KERNEL_ARGS_DEF);
    extern void aiv_all_reduce_half(KERNEL_ARGS_DEF);
    extern void aiv_all_reduce_int16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_reduce_int32_t(KERNEL_ARGS_DEF);
    extern void aiv_all_reduce_int8_t(KERNEL_ARGS_DEF);
    extern void aiv_all_reduce_bfloat16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_half(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_int16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_uint16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_float(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_int32_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_uint32_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_int8_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_uint8_t(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_bfloat16_t(KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_float(KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_half(KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_int16_t(KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_int32_t(KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_int8_t(KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_bfloat16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_half(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_int16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_uint16_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_float(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_int32_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_uint32_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_int8_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_uint8_t(KERNEL_ARGS_DEF);
    extern void aiv_all_gather_bfloat16_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_half(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_int16_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_uint16_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_float(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_int32_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_uint32_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_int8_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_uint8_t(KERNEL_ARGS_DEF);
    extern void aiv_broadcast_bfloat16_t(KERNEL_ARGS_DEF);
    extern void hccl_aiv_sync(KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_half(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_int16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_uint16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_float(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_int32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_uint32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_int8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_uint8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_vc_bfloat16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_half(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_int16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_uint16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_float(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_int32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_uint32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_int8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_uint8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_bfloat16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_half(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_int16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_uint16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_float(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_int32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_uint32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_int8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_uint8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_gather_v_bfloat16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_v_float(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_v_half(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_v_int16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_v_int32_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_v_int8_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_reduce_scatter_v_bfloat16_t(EXTERN_KERNEL_ARGS_DEF);
    extern void aiv_all_to_all_v_sp_half(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_int16_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_uint16_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_float(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_int32_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_uint32_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_int8_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_uint8_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_bfloat16_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_int64_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_uint64_t(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_to_all_v_sp_double(EXTERN_KERNEL_ARGS_DEF_V2);
    extern void aiv_all_reduce_cn_float(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_reduce_cn_half(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_reduce_cn_int16_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_reduce_cn_int32_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_reduce_cn_int8_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_reduce_cn_bfloat16_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_half(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_int16_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_uint16_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_float(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_int32_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_uint32_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_int8_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_uint8_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_all_gather_cn_bfloat16_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_reduce_scatter_cn_float(KERNEL_ARGS_DEF_A3);
    extern void aiv_reduce_scatter_cn_half(KERNEL_ARGS_DEF_A3);
    extern void aiv_reduce_scatter_cn_int16_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_reduce_scatter_cn_int32_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_reduce_scatter_cn_int8_t(KERNEL_ARGS_DEF_A3);
    extern void aiv_reduce_scatter_cn_bfloat16_t(KERNEL_ARGS_DEF_A3);
}

//用于函数名字符串和函数对象的映射
std::unordered_map<const char*, aivFunc> aivFuncMap = {
    {"aiv_all_reduce_float", aiv_all_reduce_float},
    {"aiv_all_reduce_half", aiv_all_reduce_half},
    {"aiv_all_reduce_int16_t", aiv_all_reduce_int16_t},
    {"aiv_all_reduce_int32_t", aiv_all_reduce_int32_t},
    {"aiv_all_reduce_int8_t", aiv_all_reduce_int8_t},
    {"aiv_all_reduce_bfloat16_t", aiv_all_reduce_bfloat16_t},
    {"aiv_all_to_all_half", aiv_all_to_all_half},
    {"aiv_all_to_all_int16_t", aiv_all_to_all_int16_t},
    {"aiv_all_to_all_uint16_t", aiv_all_to_all_uint16_t},
    {"aiv_all_to_all_float", aiv_all_to_all_float},
    {"aiv_all_to_all_int32_t", aiv_all_to_all_int32_t},
    {"aiv_all_to_all_uint32_t", aiv_all_to_all_uint32_t},
    {"aiv_all_to_all_int8_t", aiv_all_to_all_int8_t},
    {"aiv_all_to_all_uint8_t", aiv_all_to_all_uint8_t},
    {"aiv_all_to_all_bfloat16_t", aiv_all_to_all_bfloat16_t},
    {"aiv_reduce_scatter_float", aiv_reduce_scatter_float},
    {"aiv_reduce_scatter_half", aiv_reduce_scatter_half},
    {"aiv_reduce_scatter_int16_t", aiv_reduce_scatter_int16_t},
    {"aiv_reduce_scatter_int32_t", aiv_reduce_scatter_int32_t},
    {"aiv_reduce_scatter_int8_t", aiv_reduce_scatter_int8_t},
    {"aiv_reduce_scatter_bfloat16_t", aiv_reduce_scatter_bfloat16_t},
    {"aiv_all_gather_half", aiv_all_gather_half},
    {"aiv_all_gather_int16_t", aiv_all_gather_int16_t},
    {"aiv_all_gather_uint16_t", aiv_all_gather_uint16_t},
    {"aiv_all_gather_float", aiv_all_gather_float},
    {"aiv_all_gather_int32_t", aiv_all_gather_int32_t},
    {"aiv_all_gather_uint32_t", aiv_all_gather_uint32_t},
    {"aiv_all_gather_int8_t", aiv_all_gather_int8_t},
    {"aiv_all_gather_uint8_t", aiv_all_gather_uint8_t},
    {"aiv_all_gather_bfloat16_t", aiv_all_gather_bfloat16_t},
    {"aiv_broadcast_half", aiv_broadcast_half},
    {"aiv_broadcast_int16_t", aiv_broadcast_int16_t},
    {"aiv_broadcast_uint16_t", aiv_broadcast_uint16_t},
    {"aiv_broadcast_float", aiv_broadcast_float},
    {"aiv_broadcast_int32_t", aiv_broadcast_int32_t},
    {"aiv_broadcast_uint32_t", aiv_broadcast_uint32_t},
    {"aiv_broadcast_int8_t", aiv_broadcast_int8_t},
    {"aiv_broadcast_uint8_t", aiv_broadcast_uint8_t},
    {"aiv_broadcast_bfloat16_t", aiv_broadcast_bfloat16_t},
    {"hccl_aiv_sync", hccl_aiv_sync},
};

std::unordered_map<const char*, aivFuncExtra> aivFuncExtraMap = {
    {"aiv_all_to_all_vc_half", aiv_all_to_all_vc_half},
    {"aiv_all_to_all_vc_int16_t", aiv_all_to_all_vc_int16_t},
    {"aiv_all_to_all_vc_uint16_t", aiv_all_to_all_vc_uint16_t},
    {"aiv_all_to_all_vc_float", aiv_all_to_all_vc_float},
    {"aiv_all_to_all_vc_int32_t", aiv_all_to_all_vc_int32_t},
    {"aiv_all_to_all_vc_uint32_t", aiv_all_to_all_vc_uint32_t},
    {"aiv_all_to_all_vc_int8_t", aiv_all_to_all_vc_int8_t},
    {"aiv_all_to_all_vc_uint8_t", aiv_all_to_all_vc_uint8_t},
    {"aiv_all_to_all_vc_bfloat16_t", aiv_all_to_all_vc_bfloat16_t},
    {"aiv_all_to_all_v_half", aiv_all_to_all_v_half},
    {"aiv_all_to_all_v_int16_t", aiv_all_to_all_v_int16_t},
    {"aiv_all_to_all_v_uint16_t", aiv_all_to_all_v_uint16_t},
    {"aiv_all_to_all_v_float", aiv_all_to_all_v_float},
    {"aiv_all_to_all_v_int32_t", aiv_all_to_all_v_int32_t},
    {"aiv_all_to_all_v_uint32_t", aiv_all_to_all_v_uint32_t},
    {"aiv_all_to_all_v_int8_t", aiv_all_to_all_v_int8_t},
    {"aiv_all_to_all_v_uint8_t", aiv_all_to_all_v_uint8_t},
    {"aiv_all_to_all_v_bfloat16_t", aiv_all_to_all_v_bfloat16_t},
    {"aiv_all_gather_v_half", aiv_all_gather_v_half},
    {"aiv_all_gather_v_int16_t", aiv_all_gather_v_int16_t},
    {"aiv_all_gather_v_uint16_t", aiv_all_gather_v_uint16_t},
    {"aiv_all_gather_v_float", aiv_all_gather_v_float},
    {"aiv_all_gather_v_int32_t", aiv_all_gather_v_int32_t},
    {"aiv_all_gather_v_uint32_t", aiv_all_gather_v_uint32_t},
    {"aiv_all_gather_v_int8_t", aiv_all_gather_v_int8_t},
    {"aiv_all_gather_v_uint8_t", aiv_all_gather_v_uint8_t},
    {"aiv_all_gather_v_bfloat16_t", aiv_all_gather_v_bfloat16_t},
    {"aiv_reduce_scatter_v_float", aiv_reduce_scatter_v_float},
    {"aiv_reduce_scatter_v_half", aiv_reduce_scatter_v_half},
    {"aiv_reduce_scatter_v_int16_t", aiv_reduce_scatter_v_int16_t},
    {"aiv_reduce_scatter_v_int32_t", aiv_reduce_scatter_v_int32_t},
    {"aiv_reduce_scatter_v_int8_t", aiv_reduce_scatter_v_int8_t},
    {"aiv_reduce_scatter_v_bfloat16_t", aiv_reduce_scatter_v_bfloat16_t},
};

std::unordered_map<const char*, aivFuncExtraV2> aivFuncExtraV2Map = {
    {"aiv_all_to_all_v_sp_half", aiv_all_to_all_v_sp_half},
    {"aiv_all_to_all_v_sp_int16_t", aiv_all_to_all_v_sp_int16_t},
    {"aiv_all_to_all_v_sp_uint16_t", aiv_all_to_all_v_sp_uint16_t},
    {"aiv_all_to_all_v_sp_float", aiv_all_to_all_v_sp_float},
    {"aiv_all_to_all_v_sp_int32_t", aiv_all_to_all_v_sp_int32_t},
    {"aiv_all_to_all_v_sp_uint32_t", aiv_all_to_all_v_sp_uint32_t},
    {"aiv_all_to_all_v_sp_int8_t", aiv_all_to_all_v_sp_int8_t},
    {"aiv_all_to_all_v_sp_uint8_t", aiv_all_to_all_v_sp_uint8_t},
    {"aiv_all_to_all_v_sp_bfloat16_t", aiv_all_to_all_v_sp_bfloat16_t},
};

std::unordered_map<const char*, aivFuncExtraA3> aivFuncExtraA3Map = {
    {"aiv_all_reduce_cn_float", aiv_all_reduce_cn_float},
    {"aiv_all_reduce_cn_half", aiv_all_reduce_cn_half},
    {"aiv_all_reduce_cn_int16_t", aiv_all_reduce_cn_int16_t},
    {"aiv_all_reduce_cn_int32_t", aiv_all_reduce_cn_int32_t},
    {"aiv_all_reduce_cn_int8_t", aiv_all_reduce_cn_int8_t},
    {"aiv_all_reduce_cn_bfloat16_t", aiv_all_reduce_cn_bfloat16_t},
    {"aiv_all_gather_cn_half", aiv_all_gather_cn_half},
    {"aiv_all_gather_cn_int16_t", aiv_all_gather_cn_int16_t},
    {"aiv_all_gather_cn_uint16_t", aiv_all_gather_cn_uint16_t},
    {"aiv_all_gather_cn_float", aiv_all_gather_cn_float},
    {"aiv_all_gather_cn_int32_t", aiv_all_gather_cn_int32_t},
    {"aiv_all_gather_cn_uint32_t", aiv_all_gather_cn_uint32_t},
    {"aiv_all_gather_cn_int8_t", aiv_all_gather_cn_int8_t},
    {"aiv_all_gather_cn_uint8_t", aiv_all_gather_cn_uint8_t},
    {"aiv_all_gather_cn_bfloat16_t", aiv_all_gather_cn_bfloat16_t},
    {"aiv_reduce_scatter_cn_float", aiv_reduce_scatter_cn_float},
    {"aiv_reduce_scatter_cn_half", aiv_reduce_scatter_cn_half},
    {"aiv_reduce_scatter_cn_int16_t", aiv_reduce_scatter_cn_int16_t},
    {"aiv_reduce_scatter_cn_int32_t", aiv_reduce_scatter_cn_int32_t},
    {"aiv_reduce_scatter_cn_int8_t", aiv_reduce_scatter_cn_int8_t},
    {"aiv_reduce_scatter_cn_bfloat16_t", aiv_reduce_scatter_cn_bfloat16_t},
};

HcclResult RegisterKernel(DevType deviceType)
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel(DevType deviceType)
{
    return HcclResult::HCCL_SUCCESS;
}

const char* GetAivKernelFunc(HcclCMDType cmdType, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
{
    for (auto &aivKernelInfo: g_aivKernelInfoList) {
        if (cmdType == aivKernelInfo.cmdType && dataType == aivKernelInfo.dataType && argsType == aivKernelInfo.argsType) {
            return aivKernelInfo.kernelName;
        }
    }

    HCCL_ERROR("[AIV][GetAivKernelFunc] get aiv function name failed, cmdType %u, dataType %u, argsType %u", cmdType, dataType, argsType);
    return nullptr;
}

HcclResult ExecuteKernelLaunchImpl(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs,
    AivProfilingInfo& aivProfilingInfo, KernelLaunchMode launchMode, void* extraArgsPtr = nullptr)
{
    //如果输入为0，直接返回
    if (opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV 
        || opArgs.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        RankId rankId = RankInfoRecorder::Global()->GetRankId();
        bool recvNull = true;
        bool sendNull = true;
        switch (launchMode) {
            case KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA: {
                ExtraArgs extraArgs = *(ExtraArgs*)extraArgsPtr;
                if (extraArgs.recvCounts[rankId] > 0) {
                    recvNull = false;
                }
                if (extraArgs.sendCounts[rankId] > 0) {
                    sendNull = false;
                }
                if (recvNull && sendNull) {
                    return HcclResult::HCCL_SUCCESS; 
                }
                break;
            }
            case KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA_V2: {
                ExtraArgsV2 extraArgs = *(ExtraArgsV2*)extraArgsPtr;
                if (extraArgs.recvCounts[rankId] > 0) {
                    recvNull = false;
                }
                if (extraArgs.sendCounts[rankId] > 0) {
                    sendNull = false;
                }
                if (recvNull && sendNull) {
                    return HcclResult::HCCL_SUCCESS; 
                }
                break;
            }
        }
    } else {
        if (opArgs.count == 0) {
            return HcclResult::HCCL_SUCCESS;
        }
    }

    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
    if (topoArgs.devType == DevType::DEV_TYPE_910_93 && opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV &&
        topoArgs.serverNum > 1) {
        argsType = KernelArgsType::ARGS_TYPE_SUPERPOD;
    }

    bool isLimitCmdType = (opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLREDUCE) ||
                          (opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLGATHER) ||
                          (opArgs.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER);

    if (topoArgs.devType == DevType::DEV_TYPE_910_93 && (topoArgs.serverNum > 1 || algArgs.deterministic == 1) &&
        isLimitCmdType) {
        argsType = KernelArgsType::ARGS_TYPE_SIMPLE;
        launchMode = KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA_A3;
    }

    s32 tag = resourceArgs.aivTag;
    block_num = resourceArgs.blockDim;
    block_idx = 0;

    checker::MemLayout::Global()->InitBlockMem(resourceArgs.blockDim);

    uint8_t* buffersIn[MAX_RANK_SIZE] = {}; // 注册的CCLIN地址，所有卡可访问
    uint8_t* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问

    for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
        buffersIn[i] = (uint8_t*) resourceArgs.buffersIn[i];
        buffersOut[i] = (uint8_t*) resourceArgs.buffersOut[i];
    }

    for (u32 blkIdx = 0; blkIdx < resourceArgs.blockDim; blkIdx++) {
        switch (launchMode) {
            case KernelLaunchMode::LAUNCH_MODE_ARGS_BASE: {
                const char* funcName = GetAivKernelFunc(opArgs.cmdType, opArgs.dataType);
                CHK_PTR_NULL(funcName);

                auto func = aivFuncMap.find(funcName);
                if (func != aivFuncMap.end()) {
                    func->second(buffersIn[0], buffersIn[1], buffersIn[2], buffersIn[3], buffersIn[4], buffersIn[5], buffersIn[6], buffersIn[7],
                        buffersIn[8], buffersIn[9], buffersIn[10], buffersIn[11], buffersIn[12], buffersIn[13], buffersIn[14], buffersIn[15],
                        buffersOut[0], buffersOut[1], buffersOut[2], buffersOut[3], buffersOut[4], buffersOut[5], buffersOut[6], buffersOut[7],
                        buffersOut[8], buffersOut[9], buffersOut[10], buffersOut[11], buffersOut[12], buffersOut[13], buffersOut[14], buffersOut[15],
                        (uint8_t*)opArgs.input, (uint8_t*)opArgs.output, topoArgs.rank, topoArgs.rankSize, opArgs.count, opArgs.dataType, opArgs.op,
                        opArgs.root, tag, opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum, (uint32_t)topoArgs.devType,
                        (uint8_t*)aivProfilingInfo.counter.headCountMem, (uint8_t*)aivProfilingInfo.counter.tailCountMem, (uint8_t*)aivProfilingInfo.counter.addOneMem,
                        aivProfilingInfo.counter.memSize, aivProfilingInfo.counter.isEnableCounter, algArgs.deterministic);
                } else {
                    HCCL_ERROR("[AIV][ExecuteKernelLaunchImpl] get aiv function ptr failed, function name[%s]", funcName);
                    return HCCL_E_PARA;
                }
                break;
            }
            case KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA: {
                const char* funcName = GetAivKernelFunc(opArgs.cmdType, opArgs.dataType);
                CHK_PTR_NULL(funcName);

                ExtraArgs extraArgs = *(ExtraArgs*)extraArgsPtr;
                auto func = aivFuncExtraMap.find(funcName);
                if (func != aivFuncExtraMap.end()) {
                    func->second(buffersIn[0], buffersIn[1], buffersIn[2], buffersIn[3], buffersIn[4], buffersIn[5], buffersIn[6], buffersIn[7],
                        buffersIn[8], buffersIn[9], buffersIn[10], buffersIn[11], buffersIn[12], buffersIn[13], buffersIn[14], buffersIn[15],
                        buffersOut[0], buffersOut[1], buffersOut[2], buffersOut[3], buffersOut[4], buffersOut[5], buffersOut[6], buffersOut[7],
                        buffersOut[8], buffersOut[9], buffersOut[10], buffersOut[11], buffersOut[12], buffersOut[13], buffersOut[14], buffersOut[15],
                        (uint8_t*)opArgs.input, (uint8_t*)opArgs.output, topoArgs.rank, topoArgs.rankSize, opArgs.count, opArgs.dataType, opArgs.op,
                        opArgs.root, tag, opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum, (uint32_t)topoArgs.devType,
                        (uint8_t*)aivProfilingInfo.counter.headCountMem, (uint8_t*)aivProfilingInfo.counter.tailCountMem, (uint8_t*)aivProfilingInfo.counter.addOneMem,
                        aivProfilingInfo.counter.memSize, aivProfilingInfo.counter.isEnableCounter, algArgs.deterministic, extraArgs);
                } else {
                    HCCL_ERROR("[AIV][ExecuteKernelLaunchImpl] get aiv function ptr failed, function name[%s]", funcName);
                    return HCCL_E_PARA;
                }
                break;
            }
            case KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA_V2: {
                const char* funcName = GetAivKernelFunc(opArgs.cmdType, opArgs.dataType, argsType);
                CHK_PTR_NULL(funcName);

                ExtraArgsV2 extraArgs = *(ExtraArgsV2*)extraArgsPtr;
                auto func = aivFuncExtraV2Map.find(funcName);
                if (func != aivFuncExtraV2Map.end()) {
                    func->second(buffersIn[0], buffersIn[1], buffersIn[2], buffersIn[3], buffersIn[4], buffersIn[5], buffersIn[6], buffersIn[7],
                        buffersIn[8], buffersIn[9], buffersIn[10], buffersIn[11], buffersIn[12], buffersIn[13], buffersIn[14], buffersIn[15],
                        buffersOut[0], buffersOut[1], buffersOut[2], buffersOut[3], buffersOut[4], buffersOut[5], buffersOut[6], buffersOut[7],
                        buffersOut[8], buffersOut[9], buffersOut[10], buffersOut[11], buffersOut[12], buffersOut[13], buffersOut[14], buffersOut[15],
                        (uint8_t*)opArgs.input, (uint8_t*)opArgs.output, topoArgs.rank, topoArgs.rankSize, opArgs.count, opArgs.dataType, opArgs.op,
                        opArgs.root, tag, opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum, (uint32_t)topoArgs.devType,
                        (uint8_t*)aivProfilingInfo.counter.headCountMem, (uint8_t*)aivProfilingInfo.counter.tailCountMem, (uint8_t*)aivProfilingInfo.counter.addOneMem,
                        aivProfilingInfo.counter.memSize, aivProfilingInfo.counter.isEnableCounter, algArgs.deterministic, extraArgs);
                } else {
                    HCCL_ERROR("[AIV][ExecuteKernelLaunchImpl] get aiv function ptr failed, function name[%s]", funcName);
                    return HCCL_E_PARA;
                }
                break;
            }
            case KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA_A3: {
                const char *funcName = GetAivKernelFunc(opArgs.cmdType, opArgs.dataType, argsType);
                CHK_PTR_NULL(funcName);
                auto func = aivFuncExtraA3Map.find(funcName);
                if (func != aivFuncExtraA3Map.end()) {
                    func->second(buffersIn[0],
                        buffersIn[1],
                        buffersOut[0],
                        buffersOut[1],
                        (uint8_t *)resourceArgs.bufferSize,
                        (uint8_t *)aivProfilingInfo.counter.headCountMem,
                        (uint8_t *)aivProfilingInfo.counter.tailCountMem,
                        (uint8_t *)aivProfilingInfo.counter.addOneMem,
                        (uint8_t *)aivProfilingInfo.counter.isEnableCounter,
                        (uint8_t *)opArgs.input,
                        (uint8_t *)opArgs.output,
                        topoArgs.rank,
                        topoArgs.rankSize,
                        opArgs.count,
                        opArgs.dataType,
                        opArgs.op,
                        opArgs.root,
                        tag,
                        opArgs.isOpBase,
                        topoArgs.serverNum,
                        (uint32_t)topoArgs.devType,
                        algArgs.deterministic);
                } else {
                    HCCL_ERROR(
                        "[AIV][ExecuteKernelLaunchImpl] get aiv function ptr failed, function name[%s]", funcName);
                    return HCCL_E_PARA;
                }
                break;
            }
            default: {
                HCCL_ERROR("[AIV][ExecuteKernelLaunchImpl] launchMode[%d] is invalid", launchMode);
                return HCCL_E_PARA;
            }
        }

        block_idx++;
    }
    
    RankId rankId = RankInfoRecorder::Global()->GetRankId();
    AivTaskQueueStub::Global()->AppendAivTaskStubInMainStream(rankId);

    HCCL_INFO("[AIV][ExecuteKernelLaunch] ExecuteKernelLaunch stub function invoked");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs,
    AivProfilingInfo& aivProfilingInfo)
{
    CHK_RET(ExecuteKernelLaunchImpl(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo, KernelLaunchMode::LAUNCH_MODE_ARGS_BASE));
    return HcclResult::HCCL_SUCCESS;
}
 

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgs &extraArgs,
    AivProfilingInfo& aivProfilingInfo)
{
    CHK_RET(ExecuteKernelLaunchImpl(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo, KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA, (void*)&extraArgs));
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgsV2 &extraArgs,
    AivProfilingInfo& aivProfilingInfo)
{
    CHK_RET(ExecuteKernelLaunchImpl(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo, KernelLaunchMode::LAUNCH_MODE_ARGS_EXTRA_V2, (void*)&extraArgs));
    return HcclResult::HCCL_SUCCESS;
}
 
void TaskAivProfilerWrap(const AivOpArgs& opArgs, const AivTopoArgs& topoArgs,
    const AivResourceArgs& resourceArgs, const AivAlgArgs& algArgs, const AivProfilingInfo& aivProfilingInfo,
    void* flagMem)
{
}
 
HcclResult ClearAivSyncBuf(void** cclBuffersOut, const AivResourceArgs &resourceArgs, const AivTopoArgs &topoArgs, AivAlgArgs algArgs)
{
    return HCCL_SUCCESS;
}

}
