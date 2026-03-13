#ifndef HCCL_MC2_TYPE_STUB_H
#define HCCL_MC2_TYPE_STUB_H
#include "fwk_types.h"

struct Mc2ServerCfg {
    uint32_t version;
    uint8_t  debugMode;
    uint8_t  sendArgIndex;
    uint8_t  recvArgIndex;
    uint8_t  commOutArgIndex;
    uint8_t  reserved[8];
};

struct Mc2CommConfig {
    uint8_t  skipLocalRankCopy;
    uint8_t  skipBufferWindowCopy;
    uint8_t  stepSize;
    char     reserved[13];
    char     groupName[128]; // 指定通信域
    char     algConfig[128]; // 指定算法
    uint32_t opType; // 算子类型
    uint32_t reduceType; // reduce类型，sum,max等
    uint32_t dataType; // 输入数据类型
    uint32_t outputDataType; // 输出数据类型
};

struct Mc2Tiling {
    uint32_t            version; // 版本
    uint32_t            commConfigNum; // commComfig的个数，每个通信切片一个hcclConfig
    struct Mc2ServerCfg serverCfg; // 计算部分tiling
    struct Mc2CommConfig commConfig; // 通信部分tiling，共有Mc2CommConfig个，每个通信切片一个hcclConfig
};

namespace sim_runner {
constexpr uint32_t MAX_RANK_NUM          = 32; // 最大卡数
struct HcclCombinOpParam {
    uint64_t workSpace; // client和server之间通信的地址
    uint64_t workSpaceSize; // client和server之间通信的空间大小
    uint32_t rankId; // 当前卡rankId
    uint32_t rankDim; // 总卡数
    uint64_t winSize; // ccu不使用
    uint64_t windowsIn[MAX_RANK_NUM]; // ccu不使用
    uint64_t windowsOut[MAX_RANK_NUM]; // ccu不使用

    // for ccu
    uint64_t xnAddr;  // Xn寄存器起始地址
    uint64_t ckeAddr; // CKE寄存器起始地址
    uint64_t sprAddr; // 接收SPR参数的内存起始地址 // todo 命名
};

struct HcclCommParamDesc {
    uint64_t version : 4;   // 版本号，当前是1
    uint64_t groupNum : 4;  // groupMatmul的输入数量，每个group对应一个输入和一个输出地址
    uint64_t hasFfts : 1;   // 910下是否是ffts融合算子（多一个ffts_addr参数
    uint64_t tilingDataPtrOff : 7; // tilingdata指针所在的参数索引, 此处修改为tilingDataPtr的Offset，需要二次索引到tilingData
    uint64_t
        isDyn : 48; // 输入参数是否是动态输入，从IR输入开始计算，不包含前面的参数，is_dyn是一个bitmap，每个bit对应一个IR输入，如果是动态输入则为1，否则是0
};
}

enum class AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_ALL,
    HCCL_CMD_HALF_ALLTOALLV,
    HCCL_CMD_RESERVED
};

const std::map<OpType, AicpuComType> OP_TO_MC2_MAP = {
    {OpType::ALLREDUCE, AicpuComType::HCCL_CMD_ALLREDUCE},
    {OpType::REDUCE, AicpuComType::HCCL_CMD_REDUCE},
    {OpType::BROADCAST, AicpuComType::HCCL_CMD_BROADCAST},
    {OpType::ALLGATHER, AicpuComType::HCCL_CMD_ALLGATHER},
    {OpType::ALLTOALL, AicpuComType::HCCL_CMD_ALLTOALL},
    {OpType::REDUCESCATTER, AicpuComType::HCCL_CMD_REDUCE_SCATTER},
    {OpType::SEND, AicpuComType::HCCL_CMD_SEND},
    {OpType::RECV, AicpuComType::HCCL_CMD_RECEIVE},
};
#endif
