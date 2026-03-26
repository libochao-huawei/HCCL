#ifndef OPS_HCCL_ALLGATHER_2IN2OUT_COMMON_H
#define OPS_HCCL_ALLGATHER_2IN2OUT_COMMON_H

#include <cstdint>
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_primitives.h"
#include "acl/acl_rt.h"
#include "log.h"

namespace ops_hccl_allgather_2in2out {

constexpr uint32_t kRouteNum = 2;
constexpr uint32_t kThreadNum = 3;
constexpr uint32_t kSubThreadNum = 2;
constexpr uint32_t kControlNotifyNum = 2;
constexpr uint32_t kCustomTimeout = 1800;
constexpr uint32_t kCommIdentifierMaxLength = 128;
constexpr uint32_t kOpNameLength = 64;
constexpr uint32_t kTagLength = kCommIdentifierMaxLength + kOpNameLength;
constexpr uint32_t kNotifyAck = 0;
constexpr uint32_t kNotifyData = 1;
// 阶段 2 先只支持 OP_BASE，后续再接真实 workflow 查询。
constexpr uint32_t kWorkflowModeOpBase = 1;

enum PathType : uint32_t {
    PATH_FALLBACK_NATIVE = 0,
    PATH_FUSED_SMALLCOUNT = 1,
};

enum TopologyType : uint32_t {
    TOPO_SINGLE_NODE = 0,
    TOPO_INTRA_SUPERPOD_MULTI_NODE = 1,
    TOPO_UNSUPPORTED = 255,
};

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct CommPeerRes {
    uint32_t peerRank = 0;
    ChannelHandle channelHandle = nullptr;
    CommBuffer remoteBuffer;
    uint32_t ackNotifyIdx = kNotifyAck;
    uint32_t dataNotifyIdx = kNotifyData;
};

struct ThreadSyncRes {
    // mainToSub[i] 表示：主线程要唤醒第 i 个子线程时，往“子线程自己的 notify 槽位”写哪个索引。
    uint32_t mainToSub[kSubThreadNum] = {0};
    // subToMain[i] 表示：第 i 个子线程做完自己负责的 peer 子集后，往“主线程自己的 notify 槽位”回哪个索引。
    uint32_t subToMain[kSubThreadNum] = {0};
};

struct RouteLoopState {
    uint64_t loopMaxCount = 0;
    uint64_t totalLoopNum = 0;
};

struct RouteParam {
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t count = 0;
    uint64_t unitSize = 0;
    uint64_t totalBytes = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    RouteLoopState loopState;
    uint32_t routeId = 0;
};

struct AlgResourceCtx {
    ThreadHandle threads[kThreadNum] = {nullptr};
    uint32_t threadNum = 0;
    CommBuffer localCclBuffer;
    CommPeerRes *peerRes = nullptr;
    uint32_t peerNum = 0;
    // 注意：这里保存的是 device 侧可识别的 notify id，不是 Host 侧 aclrtNotify 句柄。
    uint32_t notifyIds[kControlNotifyNum] = {0};
    ThreadSyncRes threadSync;
};

struct CommMeta {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint32_t deviceType = 0;
    uint32_t workflowMode = 0;
    uint32_t topologyType = TOPO_UNSUPPORTED;
    char commName[kCommIdentifierMaxLength] = {0};
};

struct OpParam {
    char tag[kTagLength] = {0};
    char commName[kCommIdentifierMaxLength] = {0};
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint32_t deviceType = 0;
    uint32_t workflowMode = 0;
    uint32_t topologyType = TOPO_UNSUPPORTED;
    uint32_t pathType = PATH_FALLBACK_NATIVE;
    RouteParam routes[kRouteNum];
    AlgResourceCtx *resCtx = nullptr;
};

// 数据类型大小表，写法对齐现有样例与主仓中的 DATATYPE_SIZE_TABLE/SIZE_TABLE 风格。
constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {
    sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1
};

} // namespace ops_hccl_allgather_2in2out

#endif
