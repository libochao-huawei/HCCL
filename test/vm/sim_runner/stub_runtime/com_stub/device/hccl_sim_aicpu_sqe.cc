#include "hccl_sim_aicpu_sqe.h"
#include "../../rts_stub/SimRunnerMgr.h"
#include "../hccl_sim_comm_stub.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"
#include <bitset>

void CopyA3SqBufferStub(uint32_t devId, struct halSqCqConfigInfo *info)
{
    constexpr int sqeSize = hccl::HCCL_SQE_SIZE;     // 64
    constexpr int sqDepth = hccl::HCCL_SQE_MAX_CNT;  // 2048
    auto fakeStreamMgr = SimRunnerMgr::GetInstance().GetFakeStreamMgr();
    int tail = info->value[0];
    uint32_t sqId = info->sqId;
    int head = fakeStreamMgr->GetSqHead(sqId);             // 获取当前sqBuffer的头指针
    fakeStreamMgr->UpdataSqHead(sqId, tail);               // 更新头指针为当前尾指针
    uint8_t *sqBuffer = fakeStreamMgr->GetSqBufferAddr();  // 获取sqBuffer的地址
    // 计算本轮下发的SQE的数量
    int sqeCnt = (sqDepth + tail - head) % sqDepth;
    //  临时缓冲区，用于存放从sqBuffer拷贝的数据
    uint8_t tempBuffer[sqeSize * sqDepth];
    if (tail >= head) {
        // 数据未绕圈，直接拷贝
        memcpy(tempBuffer, sqBuffer + head * sqeSize, sqeCnt * sqeSize);
    } else {
        // 数据绕圈，分两次拷贝
        int firstPart = sqDepth - head;  // 从head到缓冲区  末尾的拷贝数量
        int secondPart = tail;           // 从缓冲区开头到tail的拷贝数量
        memcpy(tempBuffer, sqBuffer + head * sqeSize, firstPart * sqeSize);
        memcpy(tempBuffer + firstPart * sqeSize, sqBuffer, secondPart * sqeSize);
    }

    ShmCb *shmcb = GetShmCbBaseByRankTemp(devId);
    for (int i = 0; i < sqeCnt; i++) {
        int tempi = shmcb->head.aicpu.cnt + i;
        rtStarsSqeHeader_t *header =
            reinterpret_cast<rtStarsSqeHeader_t *>(reinterpret_cast<uint64_t>(&tempBuffer) + i * sqeSize);
        if (header->type == RT_STARS_SQE_TYPE_NOTIFY_RECORD) {  // NOTIFY_RECORD
            rtStarsNotifySqeV1_t tempsqe =
                *(rtStarsNotifySqeV1_t *)(reinterpret_cast<uint64_t>(&tempBuffer) + i * sqeSize);
            shmcb->head.aicpu.d2hSqe[tempi].type = FakeSqeType::NOTIFY_RECORD;
            shmcb->head.aicpu.d2hSqe[tempi].notifyId = (int)tempsqe.notify_id;
            shmcb->head.aicpu.d2hSqe[tempi].streamId = tempsqe.header.rtStreamId;
        } else if (header->type == RT_STARS_SQE_TYPE_NOTIFY_WAIT) {  // NOTIFY_WAIT
            rtStarsNotifySqeV1_t tempsqe =
                *(rtStarsNotifySqeV1_t *)(reinterpret_cast<uint64_t>(&tempBuffer) + i * sqeSize);
            shmcb->head.aicpu.d2hSqe[tempi].type = FakeSqeType::NOTIFY_WAIT;
            shmcb->head.aicpu.d2hSqe[tempi].notifyId = (int)tempsqe.notify_id;
            shmcb->head.aicpu.d2hSqe[tempi].streamId = tempsqe.header.rtStreamId;
        } else if (header->type == RT_STARS_SQE_TYPE_WRITE_VALUE) {  // NOTIFY_RECORD
            rtStarsWriteValueSqe_t tempsqe =
                *(rtStarsWriteValueSqe_t *)(reinterpret_cast<uint64_t>(&tempBuffer) + i * sqeSize);
            shmcb->head.aicpu.d2hSqe[tempi].type = FakeSqeType::NOTIFY_RECORD;
            shmcb->head.aicpu.d2hSqe[tempi].notifyId =
                (int)GetFull64BitAddr(tempsqe.write_addr_low, tempsqe.write_addr_high);
            shmcb->head.aicpu.d2hSqe[tempi].streamId = tempsqe.header.rtStreamId;
        } else if (header->type == RT_STARS_SQE_TYPE_SDMA) {
            rtStarsMemcpyAsyncSqe_t tempsqe =
                *(rtStarsMemcpyAsyncSqe_t *)(reinterpret_cast<uint64_t>(&tempBuffer) + i * sqeSize);
            if (tempsqe.opcode == 0) {  // MEM_CPY
                shmcb->head.aicpu.d2hSqe[tempi].type = FakeSqeType::MEM_CPY;
            } else {  // SDMA_REDUCE
                shmcb->head.aicpu.d2hSqe[tempi].type = FakeSqeType::SDMA_REDUCE;
                shmcb->head.aicpu.d2hSqe[tempi].reduceOp = ExtractCopyKind(tempsqe.opcode);
                shmcb->head.aicpu.d2hSqe[tempi].dataType = ExtractDataType(tempsqe.opcode);
            }
            shmcb->head.aicpu.d2hSqe[tempi].streamId = tempsqe.header.rtStreamId;
            shmcb->head.aicpu.d2hSqe[tempi].count = tempsqe.length;
            shmcb->head.aicpu.d2hSqe[tempi].dst =
                reinterpret_cast<void *>(GetFull64BitAddr(tempsqe.dst_addr_low, tempsqe.dst_addr_high));
            shmcb->head.aicpu.d2hSqe[tempi].src =
                reinterpret_cast<void *>(GetFull64BitAddr(tempsqe.src_addr_low, tempsqe.src_addr_high));
        } else {
            std::cout << "[ERROR] unsupport sqe type:" << info->prop << std::endl;
        }
    }
    shmcb->head.aicpu.cnt += sqeCnt;
}

void ParseDavidSDMASqe(ShmCb *shmcb, int index, void *sqeBuf)
{
    Hccl::RtDavidStarsMemcpySqe *sqe = reinterpret_cast<Hccl::RtDavidStarsMemcpySqe *>(sqeBuf);
    if (sqe->opcode == 0) {  // 表示是memcpy
        shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::MEM_CPY;
    } else {  // reduce
        shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::SDMA_REDUCE;
        shmcb->head.aicpu.d2hSqe[index].reduceOp = ExtractReduceTypeDavid(sqe->opcode);
        shmcb->head.aicpu.d2hSqe[index].dataType = ExtractDataTypeDavid(sqe->opcode);
    }

    shmcb->head.aicpu.d2hSqe[index].streamId = sqe->header.rtStreamId;
    shmcb->head.aicpu.d2hSqe[index].count = sqe->u.strideMode0.lengthMove;
    shmcb->head.aicpu.d2hSqe[index].dst =
        reinterpret_cast<void *>(GetFull64BitAddr(sqe->u.strideMode0.dstAddrLow, sqe->u.strideMode0.dstAddrHigh));
    shmcb->head.aicpu.d2hSqe[index].src =
        reinterpret_cast<void *>(GetFull64BitAddr(sqe->u.strideMode0.srcAddrLow, sqe->u.strideMode0.srcAddrHigh));
}

void ParseDavidNotifySqe(ShmCb *shmcb, int index, void *sqeBuf, bool isRecord)
{
    Hccl::RtDavidStarsNotifySqe *sqe = reinterpret_cast<Hccl::RtDavidStarsNotifySqe *>(sqeBuf);
    shmcb->head.aicpu.d2hSqe[index].type = isRecord ? FakeSqeType::NOTIFY_RECORD : FakeSqeType::NOTIFY_WAIT;
    shmcb->head.aicpu.d2hSqe[index].notifyId = (int)sqe->notifyId;
    shmcb->head.aicpu.d2hSqe[index].streamId = sqe->header.rtStreamId;
    std::bitset<32> bits(sqe->cntValue);  // 将 num 转换为 32 位二进制
    int count = bits.count();             // 统计 1 的个数
    shmcb->head.aicpu.d2hSqe[index].notifyCnt = count > 0 ? count : 1;
}

void ParseDavidUBWriteSqe(ShmCb *shmcb, int index, u64 wqeAddr, u16 streamId)
{
    Hccl::UdmaSqeWrite *ubWqe = reinterpret_cast<Hccl::UdmaSqeWrite *>(wqeAddr);
    // case1:UbConnLite::InlineWrite 写Notify
    shmcb->head.aicpu.d2hSqe[index].streamId = streamId;
    if (ubWqe->comm.inlineEn == 1) {
        u64 notifyAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
        int notifyId = GetNotifyId(notifyAddr);
        shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::NOTIFY_RECORD;
        shmcb->head.aicpu.d2hSqe[index].notifyId = notifyId;
        return;
    }

    // case2:UbConnLite::Write Copy
    u64 length = static_cast<u64>(ubWqe->u.sge.length);
    u64 srcAddr = GetFull64BitAddr(ubWqe->u.sge.dataAddrLow, ubWqe->u.sge.dataAddrHigh);
    u64 dstAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
    shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::MEM_CPY;
    shmcb->head.aicpu.d2hSqe[index].count = length;
    shmcb->head.aicpu.d2hSqe[index].src = reinterpret_cast<void *>(srcAddr);
    shmcb->head.aicpu.d2hSqe[index].dst = reinterpret_cast<void *>(dstAddr);

    // case3:UbConnLite::WriteReduce Reduce
    if (ubWqe->comm.udfFlag == 1) {
        shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::SDMA_REDUCE;
        shmcb->head.aicpu.d2hSqe[index].dataType = ExtractUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        shmcb->head.aicpu.d2hSqe[index].reduceOp = ExtractUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
    }
}

void ParseDavidUBWriteWithNotifySqe(ShmCb *shmcb, int index, u64 wqeAddr, u16 streamId)
{
    Hccl::UdmaSqeWriteWithNotify *ubWqe = reinterpret_cast<Hccl::UdmaSqeWriteWithNotify *>(wqeAddr);
    // 1.先构造MEM_CPY(或SDMA_REDUCE)所需参数
    u64 dstAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
    u64 srcAddr = GetFull64BitAddr(ubWqe->localU.sge.dataAddrLow, ubWqe->localU.sge.dataAddrHigh);
    u64 count = static_cast<u64>(ubWqe->localU.sge.length);
    shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::MEM_CPY;
    shmcb->head.aicpu.d2hSqe[index].streamId = streamId;
    shmcb->head.aicpu.d2hSqe[index].count = count;
    shmcb->head.aicpu.d2hSqe[index].src = reinterpret_cast<void *>(srcAddr);
    shmcb->head.aicpu.d2hSqe[index].dst = reinterpret_cast<void *>(dstAddr);
    if (ubWqe->comm.udfFlag == 1) {  // Reduce操作标识
        shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::SDMA_REDUCE;
        shmcb->head.aicpu.d2hSqe[index].dataType = ExtractUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        shmcb->head.aicpu.d2hSqe[index].reduceOp = ExtractUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
    }

    index++;
    shmcb->head.aicpu.cnt++;
    // 2.再构造NOTIFY_RECORD所需参数
    u64 notifyAddr = GetFull64BitAddr(ubWqe->notify.notifyAddrLow, ubWqe->notify.notifyAddrHigh);
    int notifyId = GetNotifyId(notifyAddr);
    shmcb->head.aicpu.d2hSqe[index].type = FakeSqeType::NOTIFY_RECORD;
    shmcb->head.aicpu.d2hSqe[index].notifyId = notifyId;
    shmcb->head.aicpu.d2hSqe[index].streamId = streamId;
}

void ParseDavidUDMASqe(ShmCb *shmcb, int index, void *sqeBuf)
{
    Hccl::RtDavidStarsUbdmaDBmodeSqe *ubSqe = reinterpret_cast<Hccl::RtDavidStarsUbdmaDBmodeSqe *>(sqeBuf);
    ShmPub *shmPub = GetShmPub();
    int jettyId = ubSqe->jettyId1;
    u64 sqVa = shmPub->socket.sqVaJettyIdMap[jettyId];          // 根据jettyId映射sqVa地址
    int piVal = shmPub->socket.piValJettyIdMap[jettyId];        // 根据jettyId映射piVal
    shmPub->socket.piValJettyIdMap[jettyId] = ubSqe->piValue1;  // 更新共享内存存放的piVal
    u64 wqeAddr = sqVa + piVal * 64;                            // 计算当前wqe的偏移地址
    Hccl::UdmaSqeCommon *ubCommon = reinterpret_cast<Hccl::UdmaSqeCommon *>(wqeAddr);
    while (true) {
        // 业务新增BatchTransfer流程，一个UB类型的SQE可能需要根据pi取多次WQE
        switch (ubCommon->opcode) {  // UdmaSqOpcode::UDMA_OPC_WRITE NOTIFY_RECORD
            case 3: {
                ParseDavidUBWriteSqe(shmcb, index, wqeAddr, ubSqe->header.rtStreamId);
                break;
            }
            case 5: {  // WRITE_WITH_NOTIFY_OPCODE UdmaSqeWriteWithNotify MEM_CPY + NOTIFY_RECORD
                ParseDavidUBWriteWithNotifySqe(shmcb, index, wqeAddr, ubSqe->header.rtStreamId);
                break;
            }
            default: {
                HCCL_ERROR("not support opcode: %d", ubCommon->opcode);
                break;
            }
        }

        if (ubCommon->cqe == 1) {
            return;  // 是最后一个WQE直接返回
        }

        // 更新SQE计数和WQE地址偏移，取下一个WQE
        shmcb->head.aicpu.cnt++;
        wqeAddr += 64;
        ubCommon = reinterpret_cast<Hccl::UdmaSqeCommon *>(wqeAddr);
        index = shmcb->head.aicpu.cnt;
    }
}

void CopyA5SqBufferStub(uint32_t devId, struct halSqCqConfigInfo *info)
{
    constexpr int sqeSize = hccl::HCCL_SQE_SIZE;     // 64
    constexpr int sqDepth = hccl::HCCL_SQE_MAX_CNT;  // 2048
    int tail = info->value[0];
    uint32_t sqId = info->sqId;
    auto fakeStreamMgr = SimRunnerMgr::GetInstance().GetFakeStreamMgr();
    int head = fakeStreamMgr->GetSqHead(sqId);
    fakeStreamMgr->UpdataSqHead(sqId, head);
    uint8_t *sqBuffer = fakeStreamMgr->GetSqBufferAddr();  // 获取sqBuffer的地址
    // 计算本轮下发的SQE的数量
    int sqeCnt = (sqDepth + tail - head) % sqDepth;
    //  临时缓冲区，用于存放从sqBuffer拷贝的数据
    uint8_t tempBuffer[sqeSize * sqDepth];
    if (tail >= head) {
        // 数据未绕圈，直接拷贝
        memcpy(tempBuffer, sqBuffer + head * sqeSize, sqeCnt * sqeSize);
    } else {
        // 数据绕圈，分两次拷贝
        int firstPart = sqDepth - head;  // 从head到缓冲区  末尾的拷贝数量
        int secondPart = tail;           // 从缓冲区开头到tail的拷贝数量
        memcpy(tempBuffer, sqBuffer + head * sqeSize, firstPart * sqeSize);
        memcpy(tempBuffer + firstPart * sqeSize, sqBuffer, secondPart * sqeSize);
    }

    ShmCb *shmcb = GetShmCbBaseByRankTemp(devId);
    for (int i = 0; i < sqeCnt; i++) {
        int tempi = shmcb->head.aicpu.cnt;  // 此处与A3实现不一样，因为会单次下发两个SQE
        int sqOffIndex = head + i;
        void *sqeBuf = static_cast<u8 *>(sqBuffer) + sqOffIndex * Hccl::AC_SQE_SIZE;
        Hccl::RtDavidStarsSqeHeader *header = reinterpret_cast<Hccl::RtDavidStarsSqeHeader *>(sqeBuf);
        switch (header->type) {
            case static_cast<int>(Hccl::RtDavidStarsSqeType::RT_DAVID_SQE_TYPE_SDMA): {
                ParseDavidSDMASqe(shmcb, tempi, sqeBuf);
                break;
            }
            case static_cast<int>(Hccl::RtDavidStarsSqeType::RT_DAVID_SQE_TYPE_NOTIFY_WAIT): {
                ParseDavidNotifySqe(shmcb, tempi, sqeBuf, false);
                break;
            }
            case static_cast<int>(Hccl::RtDavidStarsSqeType::RT_DAVID_SQE_TYPE_NOTIFY_RECORD): {
                ParseDavidNotifySqe(shmcb, tempi, sqeBuf, true);
                break;
            }
            case static_cast<int>(Hccl::RtDavidStarsSqeType::RT_DAVID_SQE_TYPE_UBDMA): {
                ParseDavidUDMASqe(shmcb, tempi, sqeBuf);
                break;
            }
            default: {
                HCCL_ERROR("not support sqe type: %d", header->type);
                break;
            }
        }
        shmcb->head.aicpu.cnt++;
    }
}