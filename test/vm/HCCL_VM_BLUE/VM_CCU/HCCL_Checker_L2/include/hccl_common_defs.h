#ifndef HCCL_COMMON_DEFS_H
#define HCCL_COMMON_DEFS_H

#include <vector>
#include <cstdint>
#include <string>
#include <array>

namespace HcclSim {

constexpr uint32_t MOVE_TOW_BYTES   = 16;
constexpr uint32_t MOVE_THREE_BYTES = 24;
constexpr uint32_t SOCKET_VNIC_IP_INFOS_INTERFACE = 55;
constexpr uint32_t GET_NOTIFY_BA = 14;
constexpr uint32_t MOVE_20_BITS = 20;
constexpr uint32_t MOVE_16_BITS = 16;

constexpr int DIE_NUM = 2;
constexpr uint32_t BYTE_NUM_4K                  = 4096;

constexpr uint32_t CCU_INSTRUCTION_NUM = 32 * 1024;
constexpr uint32_t CCU_INSTRUCTION_SIZE = 32;

constexpr uint16_t CCU_RESOURCE_GSA_MAX         = 3072;
constexpr uint16_t CCU_RESOURCE_XN_MAX          = 3072;
constexpr uint16_t CCU_RESOURCE_CKE_MAX         = 1024;
constexpr uint16_t CCU_RESOURCE_LOOP_ENGINE_MAX = 200;
constexpr uint16_t CCU_RESOURCE_MEM_SLICE_4K    = 4096;

constexpr uint16_t CCU_RESOURCE_MS_NUM          = 1536;
constexpr uint32_t CCU_RESOURCE_MS_SIZE         = CCU_RESOURCE_MS_NUM * BYTE_NUM_4K;

constexpr uint32_t URMA_EID_LEN = 16;
using SimEid = std::array<uint8_t, URMA_EID_LEN>;

#define RT_CCU_SQE_ARGS_LEN     (13U)

enum ProtocolType {
    SIM_PROTOCOL_HCCS,
    SIM_PROTOCOL_ROCE,
    SIM_PROTOCOL_NUM
};

enum HcclVmResult {
    // 0 ~ 4095: HCCL业务错误码
    HCCL_SIM_SUCCESS = 0,               /**< success */
    HCCL_SIM_E_PARA = 1,                /**< parameter error */
    HCCL_SIM_E_PTR = 2,                 /**< empty pointer */
    HCCL_SIM_E_MEMORY = 3,              /**< memory error */
    HCCL_SIM_E_INTERNAL = 4,            /**< internal error */
    HCCL_SIM_E_NOT_SUPPORT = 5,         /**< not support feature */
    HCCL_SIM_E_NOT_FOUND = 6,           /**< not found specific resource */
    HCCL_SIM_E_UNAVAIL = 7,             /**< resource unavailable */
    HCCL_SIM_E_SYSCALL = 8,             /**< call system interface error */
    HCCL_SIM_E_TIMEOUT = 9,             /**< timeout */
    HCCL_SIM_E_OPEN_FILE_FAILURE = 10,  /**< open file fail */
    HCCL_SIM_E_TCP_CONNECT = 11,        /**< tcp connect fail */
    HCCL_SIM_E_ROCE_CONNECT = 12,       /**< roce connect fail */
    HCCL_SIM_E_TCP_TRANSFER = 13,       /**< tcp transfer fail */
    HCCL_SIM_E_ROCE_TRANSFER = 14,      /**< roce transfer fail */
    HCCL_SIM_E_RUNTIME = 15,            /**< call runtime api fail */
    HCCL_SIM_E_DRV = 16,                /**< call driver api fail */
    HCCL_SIM_E_PROFILING = 17,          /**< call profiling api fail */
    HCCL_SIM_E_CCE = 18,                /**< call cce api fail */
    HCCL_SIM_E_NETWORK = 19,            /**< call network api fail */
    HCCL_SIM_E_AGAIN = 20,              /**< try again */
    HCCL_SIM_E_REMOTE = 21,             /**< error cqe */
    HCCL_SIM_E_SUSPENDING = 22,         /**< error communicator suspending */
    HCCL_SIM_E_OPRETRY_FAIL = 23,       /**< retry constraint */
    HCCL_SIM_E_IN_STATUS = 1041,        /**< The error information is in the status. */
    HCCL_SIM_E_RESERVED,                /**< reserved */

    // 4096 ~ 8191: HCCL_SIM错误码
    // 4096 ~ 5119: HCCL_SIM Host错误码
    HCCL_SIM_HOST_ERROR_CMD = 4096,
    HCCL_SIM_HOST_SUCCESS_CMD = 4097,
    HCCL_SIM_HOST_ERROR_PARSE_CMD = 4098,
    HCCL_SIM_HOST_ERROR_RESOURCE = 4099,
    HCCL_SIM_HOST_EXIT = 4100,
    HCCL_SIM_HOST_ERROR_EXE = 4101,
    HCCL_SIM_HOST_RESERVED = 5119,

    // 5120 ~ 6143: HCCL_SIM Porxy错误码
    HCCL_SIM_PROXY_ERROR_CMD = 5120,
    HCCL_SIM_PROXY_RESERVED = 6143,

    // 6144 ~ 7167: HCCL_SIM SHM错误码
    HCCL_SIM_SHM_ERROR_CMD = 6144,
    HCCL_SIM_SHM_TASK_COLLECTION_ERROR,
    HCCL_SIM_SHM_NOT_INIT,
    HCCL_SIM_SHM_BUFFER_OUT_OF_MEMORY,
    HCCL_SIM_SHM_PROXY_OUT_OF_RANGE,
    HCCL_SIM_SHM_OBJ_NOT_FOUND,
    HCCL_SIM_SHM_FAIL,
    HCCL_SIM_SHM_RESERVED = 7167,

    // 7168 ~ 8191: HCCL_SIM IPC错误码
    HCCL_SIM_IPC_ERROR_CMD = 6144,
    HCCL_SIM_IPC_RESERVED = 8191,

    // 8192 ~ 9215: HCCL_SIM VIRTUAL_RUNTIME错误码
    HCCL_SIM_VRT_ERROR_CMD = 8192,
    HCCL_SIM_VRT_RESERVED = 9215
};

enum HcclVmMode {
    CHECKER = 0,
    RUNNER  = 1
};

}   // namespace HcclSim

using PhyDeviceId = uint32_t;
using ServerMeta = std::vector<PhyDeviceId>;
using SuperPodMeta = std::vector<ServerMeta>;
using TopoMeta = std::vector<SuperPodMeta>;

static inline uint32_t ShmGetPhyDeviceTotalCount(const TopoMeta& topo_meta) {
    uint32_t total_count = 0;

    // 第一层：遍历所有 SuperPod（超级节点）
    for (const auto& super_pod : topo_meta) {
        // 第二层：遍历每个 SuperPod 中的所有 Server（服务器）
        for (const auto& server : super_pod) {
            // 第三层：累加每个 Server 中的 PhyDeviceId 数量
            total_count += static_cast<uint32_t>(server.size());
        }
    }
    return total_count;
}

enum class HccLTaskMetaType : char {
    NOTIFY_WAIT,
    NOTIFY_RECORD,
    REDUCE,
    MEM_CPY,
    CCU_GRAPH,
    AIV_GRAPH,
    EVENT_WAIT,
    EVENT_RECORD
};

typedef enum {
    SUCCESS = 0,
    FAILURE = 1,
    PENDING = 2,
} HccLTaskStatus;

union HcclTaskCid {
    struct {
        uint64_t commId : 16;
        uint64_t rankId : 16;
        uint64_t index  : 32;
    } field;
    uint64_t value;
};

#pragma pack(push, 1)
typedef struct {
    uint32_t    srcRankId;
    uint64_t    srcOffset;
    uint32_t    dstRankId;
    uint64_t    dstOffset;
    uint64_t    len;
    uint8_t     protocol;
} TransMemTask;

typedef struct {
    uint32_t    srcRankId;
    uint64_t    srcOffset;
    uint32_t    dstRankId;
    uint64_t    dstOffset;
    uint8_t     dataType;   // HcclDataType from hccl_types.h
    uint64_t    dataCount;
    uint8_t     reduceOp;   // HcclReduceOp from hccl_types.h
    uint8_t     protocol;
} ReduceTask;

typedef struct {
    uint32_t    srcRankId;
    uint64_t    notifyId;
    uint32_t    dstRankId;
    uint8_t     notifyCount;
    uint8_t     protocol;
} NotifyTask;

typedef struct {
    uint8_t    dieId;
    uint8_t    missionId;
    uint16_t   timeout;
    uint16_t   instStartId;
    uint16_t   instCnt;
    uint32_t   key;
    uint32_t   argSize;
    uint64_t   args[RT_CCU_SQE_ARGS_LEN];
} CcuTask;

typedef struct {
    HccLTaskMetaType    taskType;
    uint16_t            commId;
    uint32_t            rankId;
    uint64_t            streamId;
    union {
        TransMemTask    transMem;
        ReduceTask      reduce;
        NotifyTask      notify;
        CcuTask         ccu;
    }                   taskData;
} HcclTaskMetaData;

typedef struct {
    uint64_t    taskCid;
    uint16_t    dispatchId;
} HcclTaskReq;

typedef struct {
    uint64_t    taskCid;
    uint16_t    status;
} HcclTaskRsp;

#pragma pack(pop)

enum BufferType {
    INPUT = 0,
    OUTPUT,
    CCL,
    RESERVED
};

class DataSlice {
public:
    DataSlice() : type_(BufferType::INPUT), offset_(0), size_(0) {}
    DataSlice(BufferType type, uint64_t offset, uint64_t size) : type_(type), offset_(offset), size_(size) {}
    std::string Describe() const { return "BufferType: " + std::to_string(static_cast<int>(type_)) +
        ", Offset: " + std::to_string(offset_) + ", Size: " + std::to_string(size_); }
    inline BufferType GetType() const { return type_; }
    inline uint64_t GetOffset() const { return offset_; }
    inline uint64_t GetSize() const { return size_; }
    void SetBufferType(const BufferType type) { type_ = type; }
    void SetOffset(uint64_t offset) { offset_ = offset; }
    void SetSize(uint64_t size) { size_ = size; }
private:
    BufferType type_;
    uint64_t   offset_;
    uint64_t   size_;
};

#endif