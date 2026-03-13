/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: TransportPub 头文件
 */

#ifndef TESTS_STUB_TRANSPORT_PUB_H
#define TESTS_STUB_TRANSPORT_PUB_H

#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "sal_pub.h"
#include "adapter_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "topoinfo_struct.h"
#include "hccl_socket.h"
#include "notify_pool.h"

namespace hccl {
/* Max number of data notify */
constexpr u32 HCCL_IPC_STR_BASE = 10; // IPC 字符串转数字的进制
constexpr u32 HCCL_IPC_STR_HEX = 16; // IPC 字符串转数字的进制
constexpr u32  MAX_OFFSET_STR_LEN = 2048; /* socket信息长度2K = 2 * 1024 */

class TransportBase;
class ExchangerBase;
struct RxMemoryInfo;
struct TxMemoryInfo;
enum class UserMemType;
class DeviceMem;
class MemNameRepository;
struct TransportResourceInfo;

enum class MachineType {
    MACHINE_SERVER_TYPE,
    MACHINE_CLIENT_TYPE,
    MACHINE_RESERVED_TYPE
};
enum class LinkMode {
    LINK_SIMPLEX_MODE,
    LINK_DUPLEX_MODE,
    LINK_RESERVED_MODE
};

// 传参数的时候都填充，link自己使用的时候区分
// RDMA:  machineType+serverId+local_rank_id+remote_rank_id+collectiveId
// TCP:   machineType+serverId+local_rank_id+remote_rank_id+collectiveId
// PCIE:  local_rank_id+remote_rank_id+collectiveId+localDeviceId+remoteDeviceId
using MachinePara = struct TagMachinePara {
public:
    MachineType machineType{MachineType::MACHINE_RESERVED_TYPE};  // client或者server
    LinkMode linkMode{LinkMode::LINK_RESERVED_MODE};
    std::string collectiveId{""};  // 本节点所在的通信域ID
    std::string tag{""};
    std::string serverId;                       // 本端server id

    HcclIpAddress localIpAddr;     // 本端rank ip
    HcclIpAddress remoteIpAddr;    // 对端rank ip

    u32 localSocketPort;
    u32 remoteSocketPort;

    s32 localDeviceId{-1};                      // 本端device physical id
    s32 remoteDeviceId{-1};                     // 对端device physical id
    s32 deviceLogicId;

    u32 localUserrank{INVALID_VALUE_RANKID};    // 本端user rank
    u32 remoteUserrank{INVALID_VALUE_RANKID};   // 对端user rank

    u32 localWorldRank{INVALID_VALUE_RANKID};   // 本端world group rank
    u32 remoteWorldRank{INVALID_VALUE_RANKID};  // 对端world group rank

    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    DevType deviceType{DevType::DEV_TYPE_COUNT};

    std::vector<std::shared_ptr<HcclSocket> > sockets;

    DeviceMem inputMem{DeviceMem()};
    DeviceMem outputMem{DeviceMem()};

    // link特性位图: bit0:0x1支持WRITE操作（源端发起数据传输，优选项）
    // bit1:0x2支持READ操作（目的端发起数据传输）。如果同时支持link优先选用目的端发起数据传输
    u64 linkAttribute{0x1}; // 初始设置为WRITE操作，从源端发起数据传输;

    bool supportDataReceivedAck{false};
    bool isAicpuModeEn{false};
    std::vector<u32> srcPorts; // 多qp配置的源端口号
    u32 notifyNum;
    TagMachinePara() {}

    TagMachinePara(const struct TagMachinePara &that)
    {
        machineType = (that.machineType);
        linkMode = (that.linkMode);
        serverId = (that.serverId);
        localIpAddr = (that.localIpAddr);
        remoteIpAddr = (that.remoteIpAddr);
        localDeviceId = (that.localDeviceId);
        remoteDeviceId = (that.remoteDeviceId);
        localUserrank = (that.localUserrank);
        remoteUserrank = (that.remoteUserrank);
        localWorldRank = (that.localWorldRank);
        remoteWorldRank = (that.remoteWorldRank);
        collectiveId = (that.collectiveId);
        deviceType = (that.deviceType);
        tag = (that.tag);
        inputMem = (that.inputMem);
        outputMem = (that.outputMem);
        linkAttribute = (that.linkAttribute);
        sockets = (that.sockets);
        supportDataReceivedAck = (that.supportDataReceivedAck);
        nicDeploy = (that.nicDeploy);
        localSocketPort = that.localSocketPort;
        remoteSocketPort = that.remoteSocketPort;
        isAicpuModeEn = that.isAicpuModeEn;
        deviceLogicId = that.deviceLogicId;
        srcPorts = that.srcPorts;
        notifyNum = that.notifyNum;
    }

    struct TagMachinePara &operator=(struct TagMachinePara &that)
    {
        if (&that != this) {
            machineType = (that.machineType);
            linkMode = (that.linkMode);
            serverId = (that.serverId);
            localIpAddr = (that.localIpAddr);
            remoteIpAddr = (that.remoteIpAddr);
            localDeviceId = (that.localDeviceId);
            remoteDeviceId = (that.remoteDeviceId);
            localUserrank = (that.localUserrank);
            remoteUserrank = (that.remoteUserrank);
            localWorldRank = (that.localWorldRank);
            remoteWorldRank = (that.remoteWorldRank);
            collectiveId = (that.collectiveId);
            deviceType = (that.deviceType);
            tag = (that.tag);
            inputMem = (that.inputMem);
            outputMem = (that.outputMem);
            linkAttribute = (that.linkAttribute);
            sockets = (that.sockets);
            supportDataReceivedAck = (that.supportDataReceivedAck);
            localSocketPort = that.localSocketPort;
            remoteSocketPort = that.remoteSocketPort;
            isAicpuModeEn = that.isAicpuModeEn;
            deviceLogicId = that.deviceLogicId;
            srcPorts = that.srcPorts;
            notifyNum = that.notifyNum;
        }

        return *this;
    }
};

struct TransportPara {
    std::chrono::milliseconds timeout;
    NICDeployment nicDeploy;
    u32 localDieID;
    u32 dstDieID;
    HcclIpAddress* selfIp;
    HcclIpAddress* peerIp;
    u32 peerPort;
    u32 selfPort;
    const void* transportResourceInfoAddr;
    size_t transportResourceInfoSize;
    u32 index;
    dev_t shmDev;
    bool isRootRank;
    u32 devLogicId;
    u32 proxyDevLogicId;
    s32 qpMode = 0;
    bool isESPs = false;
    bool virtualFlag = false;
};

class Transport {
public:

    struct Buffer {
        const void *addr{nullptr};
        u32 size{0};

        Buffer() : addr(nullptr), size(0) {}
        Buffer(const void *addr, u32 size) : addr(addr), size(size) {}
    };

    Transport(TransportType type, TransportPara& para, MachinePara &machinePara, LinkType linkType);
    Transport(TransportType type, TransportPara& para,
              const HcclDispatcher dispatcher,
              const std::unique_ptr<NotifyPool> &notifyPool,
              MachinePara &machinePara);
    virtual ~Transport();

    virtual HcclResult Init();
    virtual HcclResult DeInit();

    virtual HcclResult TxDataSignal(Stream &stream);
    virtual HcclResult RxDataSignal(Stream &stream);

    virtual HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream);
    virtual HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream &stream);

    virtual HcclResult TxWithReduce(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len,
                                    const HcclDataType datatype, HcclReduceOp redOp, Stream &stream);
    virtual HcclResult TxWithReduce(const std::vector<TxMemoryInfo> &txWithReduceMems, const HcclDataType datatype,
        HcclReduceOp redOp, Stream &stream);
    virtual HcclResult RxWithReduce(UserMemType recvSrcMemType, u64 recvSrcOffset, void *recvDst, u64 recvLen,
        void *reduceSrc, void *reduceDst, u64 reduceDataCount, HcclDataType reduceDatatype,
        HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr);
    virtual HcclResult RxWithReduce(const std::vector<RxWithReduceMemoryInfo> &rxWithReduceMems,
        HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream,
       const u64 reduceAttr);
    virtual bool IsSupportTransportWithReduce();

    virtual HcclResult ConnectAsync(u32& status);
    virtual HcclResult ConnectQuerry(u32& status);
    virtual HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream);
    virtual HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream &stream);

    virtual HcclResult TxAck(Stream &stream);
    virtual HcclResult RxAck(Stream &stream);
    HcclResult DataReceivedAck(Stream &stream);

    virtual HcclResult TxPrepare(Stream &stream);
    virtual HcclResult RxPrepare(Stream &stream);

    virtual HcclResult TxDone(Stream &stream);
    virtual HcclResult RxDone(Stream &stream);

    virtual HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream);
    virtual HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream);

    // 保证send语义完成
    virtual HcclResult TxWaitDone(Stream &stream);
    // 保证recv语义完成
    virtual HcclResult RxWaitDone(Stream &stream);
    // TxWaitDone、RxWaitDone共同出现保证sendrecv语义完成

    virtual HcclResult GetRemoteMem(UserMemType memType, void **remotePtr);
    virtual HcclResult GetRemoteMem(std::vector<void *> *remotePtr);
    HcclResult GetTxAckDevNotifyInfo(HcclSignalInfo &notifyInfo);
    HcclResult GetRxAckDevNotifyInfo(HcclSignalInfo &notifyInfo);
    HcclResult GetTxDataSigleDevNotifyInfo(HcclSignalInfo &notifyInfo);
    HcclResult GetRxDataSigleDevNotifyInfo(HcclSignalInfo &notifyInfo);

    hccl::LinkType GetLinkType() const;
    bool IsSpInlineReduce() const;
    bool GetSupportDataReceivedAck() const;
    u32 GetRemoteRank();

    void EnableUseOneDoorbell();

    bool GetUseOneDoorbellValue();

    HcclResult TxEnv(const void *ptr, const u64 len, Stream &stream);
    HcclResult RxEnv(Stream &stream);
    bool IsTransportRoce();

    virtual void Break()
    {
        return;
    }

    virtual HcclResult Write(
        const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream);
    virtual HcclResult Read(
        const void *localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream);

    virtual HcclResult PostReady(Stream &stream);
    virtual HcclResult WaitReady(Stream &stream);

    virtual HcclResult PostFin(Stream &stream);
    virtual HcclResult WaitFin(Stream &stream);

    virtual HcclResult PostFinAck(Stream &stream);
    virtual HcclResult WaitFinAck(Stream &stream);

    bool IsValid();
    HcclResult ReadSync(struct Transport::Buffer &localBuf, struct Transport::Buffer &remoteBuf, Stream &stream);
    HcclResult ReadAsync(struct Transport::Buffer &localBuf, struct Transport::Buffer &remoteBuf, Stream &stream);
    HcclResult WriteSync(struct Transport::Buffer &remoteBuf, struct Transport::Buffer &localBuf, Stream &stream);
    HcclResult WriteAsync(struct Transport::Buffer &remoteBuf, struct Transport::Buffer &localBuf, Stream &stream);
    HcclResult Fence();
    HcclResult ReadReduceSync(struct Transport::Buffer &localBuf, struct Transport::Buffer &remoteBuf,
        const HcclDataType datatype, HcclReduceOp redOp, Stream &stream);
    HcclResult Post(u32 notifyIdx, Stream &stream);
    HcclResult Wait(u32 notifyIdx, Stream &stream);
    inline TransportType GetTransportType() const
    {
        return transportType_;
    }

    TransportType transportType_;
    MachinePara machinePara_;
    hccl::LinkType linkType_;
};

using LINK = std::shared_ptr<Transport>;
}  // namespace hccl

#endif /* TRANSPORT_BASE_H */
