#ifndef SIM_MODEL_DEFS_H
#define SIM_MODEL_DEFS_H

#include <cstdint>
#include "hccl_common_defs.h"

using namespace HcclSim;

namespace sim {

typedef struct {
    uint64_t id;  // PK
    uint64_t pod_id;
    char version[16];
} Server;

typedef struct {
    uint64_t id;  // PK
    uint64_t server_id;
    char ip_addr[40]; // ip地址
    uint8_t arch;
} Host;

typedef struct {
    uint64_t id;       // PK
    uint64_t host_id;  // FK
    uint64_t pid;
    uint64_t thread_id;
    uint64_t timeout_config_ms;
    uint64_t current_ctx_id;  // FK
} Runner;

typedef struct {
    uint64_t id;         // PK
    uint32_t server_id;  // FK
    uint32_t logic_id{0xFFFF};
    uint32_t physical_id;
    uint32_t overflow_mode;
    char soc_version[16];
    uint32_t status;// 0 可用， 1不可用
} Device;

enum TsDevType {
    TS_DEV_TYPE_SCALAR  = 0,
    TS_DEV_TYPE_CPU     = 1,
    TS_DEV_TYPE_CCU     = 2
};

typedef struct {
    uint64_t id;            // PK
    uint64_t device_id;     // FK
    uint8_t type;   // 0:Scalar, 1:CCU, 2:CPU
} TaskSchedulerDevice;

enum ComputeDieType {
    COMPUTE_TYPE_CUBE       = 0,
    COMPUTE_TYPE_VECTOR     = 1,
    COMPUTE_TYPE_HYBRID     = 2
};

typedef struct {
    uint64_t id;            // PK
    uint64_t ts_id;     // FK
    uint8_t type;       // 0:Cube 1:Vector 2: HybridCompute
} ComputeDie;

typedef struct {
    uint64_t id;            // PK
    uint64_t device_id;     // FK
    uint8_t overflow_status;
    uint8_t synchronize_strategy;
    uint8_t synchronize_timeout;
    uint8_t capability_mask;
    uint8_t run_by_host;
    uint8_t ts_core;
    uint8_t online_status;
} DeviceStatus;

typedef struct {
    uint64_t id;         // PK
    uint64_t device_id;  // FK
    uint64_t ccu_id;     // FK
    // uint8_t eid[16U];
    uint32_t func_id;
    uint64_t rdma_handle; // ccu建链过程使用，暂时rdma_handle等价与ctx_handle
    char name[128]; // 对应topo文件中的port_id: 0/0 ~ 0/8
    char ip_addr[40]; // ip地址
    uint8_t protocol; // 协议类型：HCCS/ROCE/SIO/PCIE等
    uint8_t status; // 0: 未使用 1: 已使用
} Port;

typedef struct {
    uint64_t id;         // PK
    uint64_t device_id;  // FK
    uint8_t die_id;
    uint8_t status;
} Ccu;

typedef struct {
    uint64_t id;      // PK
    uint64_t ccu_id;  // FK
    uint8_t instr_space[CCU_INSTRUCTION_NUM][CCU_INSTRUCTION_SIZE]; // 指令空间1M: 32K个指令
    uint16_t instr_cnt{0};
    uint16_t state{0};
    // uint64_t xn[CCU_RESOURCE_XN_MAX];  // xn寄存器
    // uint64_t gsa[CCU_RESOURCE_GSA_MAX]; // gsa寄存器
    // uint16_t cke[CCU_RESOURCE_CKE_MAX]; // cke寄存器
    // char ms[CCU_RESOURCE_MS_SIZE]; // ms寄存器
} CcuResource; // todo: 后续可能改为关系表，数据保存在共享内存

typedef struct {
    uint64_t id;          // PK
    uint64_t src_dev_id;  // FK
    uint64_t dst_dev_id;  // FK
    uint8_t link_type;
    uint8_t access_by_remote;
} DeviceConnection;

typedef struct {
    uint64_t id;  // PK
    uint64_t src_port;
    uint64_t dst_port;
    uint8_t  type; // 类型：0 - 框内，1 - 出框(dst_port=0)
} EndPointPair;

typedef struct {
    uint64_t id;                  // PK
    uint64_t end_point_pair_id;  // FK
    uint16_t channel_id;
    uint32_t src_rank;
    uint32_t dst_rank;
    uint8_t  src_die;
    uint8_t  dst_die;
} CcuChannel;

typedef struct {
    uint64_t id;  // PK
    uint64_t run_id;
    uint64_t device_id;
    uint8_t is_default;
    uint64_t float_overflow_addr;
    uint8_t capture_mode;
} Context;

typedef struct {
    uint64_t id;      // PK
    uint64_t ctx_id;  // 上下文ID (FK)
    uint64_t sq_base_addr;
    uint8_t is_primary_default;
    uint8_t is_other_default;
    uint8_t priority;
    uint8_t schedule_strategy;
    uint8_t failure_mode;
    uint32_t user_tag;
    uint8_t overflow_switch;
    uint8_t activated;
    uint8_t capture_status;
    uint8_t task_complete_status;
} Stream;

typedef struct {
    uint64_t id;  // PK
    uint64_t stream_id;
    uint64_t cid;
    uint64_t seq_number;
    uint8_t type;
} Task;

typedef struct {
    uint64_t event_id;
    uint64_t excute_time_ms;
    uint64_t finish_time_ms;
    uint32_t op_timeout_s;
} EventSyncTask;

typedef struct {
    uint64_t id;  // PK
    uint64_t create_ctx_id;
    uint64_t device_notify_seq;
    uint8_t value;
} Notify;

typedef struct {
    uint64_t id;  // PK
    uint64_t notify_id;
    uint8_t name_or_key[16];
    uint8_t create_pid;
} IpcNotify;

typedef struct {
    uint64_t ipc_id;  // FK
    uint64_t visitor_pid;
} IpcNotifyVistorList;

typedef struct {
    uint64_t notify_id;  // FK
} NotifyRecordTask;

typedef struct {
    uint64_t notify_id;  // FK
} NotifyWaitTask;

typedef struct {
    uint64_t id;             // PK
    uint64_t create_ctx_id;  // FK
    uint64_t event_flag;
    uint64_t device_res_seq;
    uint64_t created_time;
    uint8_t status;
} Event;

typedef struct {
    uint64_t id;
    uint64_t device_id;
    uint64_t addr;
    uint64_t size;
    uint8_t type;
    uint64_t ref_count;
} PhyMemBlock;

enum VirMemType {
    VIR_MEM_TYPE_HOST,
    VIR_MEM_TYPE_DEV
};

typedef struct {
    uint64_t id;
    uint64_t start_ptr;
    uint64_t size;
    uint64_t ctx_id;
    uint64_t phy_mem_id;
    uint64_t owner_pid;
    uint8_t src_type;//0: host 1: device
    uint8_t policy;
} VirtualMemBlock;

typedef struct {
    uint64_t id;
    uint64_t vir_mem_id;
    uint8_t create_pid;
} IpcMemRecord;

typedef struct {
    uint64_t id;
    uint64_t name_or_key;
    uint64_t pid;
    uint8_t create_pid;
} IpcMemWhiteList;

typedef struct {
    uint64_t id;
    uint64_t vir_mem_id;
    uint64_t phy_mem_id;
    uint8_t create_pid;
} FdMemRecord;

typedef struct {
    uint64_t id;
    uint64_t name_or_key;
    uint64_t pid;
    uint8_t create_pid;
} FdMemWhiteList;


typedef struct {
    uint64_t id;
    uint32_t device_id;
    uint8_t role; // 0:server,1:client
    uint8_t state; // 0: inited 1:listened 2:connected
    uint64_t ip_id; // ip id
} RaSocket;

typedef struct {
    uint64_t id;        // PK
    uint64_t server_id; // FK
    uint64_t client_id; // FK
    uint32_t ref_cnt;
} RaSocketPair;

// todo: 数据建模？
typedef struct {
    uint64_t id;        // PK
    uint32_t rank_id; // todo: 后续可能需要跟device关联
    uint64_t base_addr;
    uint8_t  buf_type;
    uint8_t  reserved;
    uint64_t size;
    uint64_t global_offset;
} MemoryLayout;

// ReduceScatterV AllGatherV使用
struct VDataDesTagInner {
    uint16_t dataType;   // 数据类型
    uint32_t count{0}; // rank size
    uint64_t displs[64];    // 每个rank的数据在sendBuf中的偏移量（单位为dataType）
    uint64_t counts[64];    // 每个rank在sendBuf中的数据size，第i个元素表示需要向rank i发送/接受的数据量
};

struct All2AllDataDesTagInner {
    uint16_t sendType;   // 发送数据的数据类型
    uint16_t recvType;   // 接收数据的数据类型
    uint64_t sendCount;              // 发送数据量 (All2All)
    uint64_t recvCount;              // 接收数据量 (All2All)
    uint32_t count{0}; // count = rankSize * rankSize
    uint64_t sendCountMatrix[4096];   // (All2AllVC) sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
};

typedef struct {
    uint64_t id;        // PK
    uint32_t rank_id;
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
} SimModelData;

typedef struct {
    uint64_t id;        // PK
    uint32_t rank_id;
    uint64_t device_id; // FK
} Rank;

}
#endif