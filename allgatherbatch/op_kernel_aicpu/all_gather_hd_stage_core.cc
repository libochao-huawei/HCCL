#include "all_gather_hd_stage_core.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include "all_gather_nhr_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

namespace {
static u32 GetStepNumInterServer(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    return nSteps;
}

static void ReorderSequence(u32 start, u32 end, u32 len, std::vector<u32> &tree, std::vector<u32> &tmp)
{
    const u32 divideTwo = 2;

    for (u32 i = start; i < end; i++) {
        u32 offset = i - start;
        if ((offset & 1) == 0) {
            tmp[start + offset / divideTwo] = tree[i];
        } else {
            tmp[start + (offset + len) / divideTwo] = tree[i];
        }
    }
}

static void GetRankMapping(const u32 rankSize, std::vector<u32> &sliceMap)
{
    std::vector<u32> tree;
    for (u32 i = 0; i < rankSize; i++) {
        tree.push_back(i);
    }

    // 其他的再进行计算
    std::vector<u32> tmp(rankSize);
    u32 nSteps = GetStepNumInterServer(rankSize);
    u32 len = rankSize;

    for (u32 step = 0; step < nSteps; step++) {
        u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
        if (nSlices <= 1) {
            break;
        }

        bool endFlag = false;
        for (u32 part = 0; part * len < rankSize; part++) {
            u32 start = part * len;
            u32 end = std::min(start + len, rankSize);
            ReorderSequence(start, end, len, tree, tmp);

            if (((end - start) & 1) == 1) {
                endFlag = true;
            }
        }

        for (u32 i = 0; i < rankSize; i++) {
            tree[i] = tmp[i];
        }

        if (endFlag) {
            break;
        }

        len >>= 1;
    }

    // 因为取的是tree中rank的idx，所以直接返回反向的映射
    sliceMap.resize(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        sliceMap[tree[i]] = i;
    }
    return;
}
}

HcclResult AllGatherHDStage::ReverseId(u32 oriIdx, u32 &revIdx)
{
    revIdx = 0;
    u32 powerBase = 0;
    for (u32 i = 0; i < powerSteps_; i++) {
        powerBase = static_cast<u32>(pow(base, i));
        revIdx += (oriIdx / powerBase % base) * static_cast<u32>(pow(base, powerSteps_ - i - 1));
    }
    return HCCL_SUCCESS;
}

AllGatherHDStage::AllGatherHDStage(const OpParam &param, AlgResourceCtx &resCtx,
    ExecMem &execMem, std::vector<ChannelResource> &channels)
    : param_(param), resCtx_(resCtx), execMem_(execMem), channels_(channels)
{
    rank = param_.topoInfo.rank;
    rankSize = param_.topoInfo.rankSize;
    dataType_ = execMem_.dataType;
}

HcclResult AllGatherHDStage::RunPreCopy()
{
    //交换数据
    HCCL_INFO("RunPreCopy run: rank[%u] totalrank[%u] outputMem[%p] count[%llu]",
        rank, rankSize, execMem_.outputMem.addr, execMem_.count);
    std::vector<u32> noPowerMap;
    GetRankMapping(noPower_, noPowerMap);
    std::vector<u32> noPowerRevMap(noPower_);
    CHK_PRT_RET(noPowerMap.size() != noPower_,
        HCCL_ERROR("[AllGatherHDStage][RunPreCopy]rank[%u] count[%llu] failed",
            rank, execMem_.count),  HCCL_E_RESERVED);
    for (u32 i = 0; i < noPowerMap.size(); i++) {
        noPowerRevMap[noPowerMap[i]] = i;
    }
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_ ));
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_ ));
    u32 revRank = 0;
    u32 revIdx = 0;
    CHK_RET(ReverseId(groupIdx, revIdx));
    // 将revRank的数据写到本卡
    revRank = revIdx * noPower_ + noPowerMap[group];
    // 将本卡数据写到revRankrev卡
    groupIdx = rank % noPower_;
    group = rank / noPower_;
    CHK_RET(ReverseId(group, revIdx));
    u32 revRankrev = noPowerRevMap[groupIdx] * static_cast<u32>(pow(base, powerSteps_ ))  + revIdx;
    HcclMem UserMemIn = {HcclMemType::HCCL_MEM_TYPE_DEVICE, execMem_.inputPtr, totalSize_};
    if (revRank != rank && revRankrev != rank) {
        CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[revRank].handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[revRankrev].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        void *remMemPtr = static_cast<u8 *>(channels_[revRankrev].remoteBuffer.addr) + channels_[revRankrev].remoteBuffer.offset;
        void *srcPtr = execMem_.inputPtr;
        void *dstPtr = static_cast<u8 *>(remMemPtr) + (rank % (rankSize / static_cast<u32>(pow(base, finalSteps_)))) * totalSize_;
        CHK_RET(HcommWriteOnThread(resCtx_.mainThreadHandle, channels_[revRankrev].handle, dstPtr, srcPtr, totalSize_));
        CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[revRankrev].handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[revRank].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    } else {
        HcclMem dst = HcclMemRange(execMem_.outputMem, (rank % (rankSize / static_cast<u32>(pow(base, finalSteps_)))) * totalSize_, totalSize_);
        void *srcPtr = execMem_.inputPtr;
        void *dstPtr = dst.addr;
        CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dstPtr, srcPtr, totalSize_));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherNoPower()
{
    HCCL_INFO("[AllGatherHDStage][RunAllGather] rank[%u] tempAlg AllGatherNHR inputMem[%p] outputMem[%p] mem_size[%llu] "\
        "count[%llu]", rank, execMem_.inputMem.addr, execMem_.outputMem.addr, execMem_.outputMem.size(), execMem_.count);
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_ ));
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_ ));
    u32 revIdx = 0;
    CHK_RET(ReverseId(groupIdx, revIdx));
    u64 baseOffset = ((revIdx * noPower_) % (rankSize / static_cast<u32>(pow(base, finalSteps_))))* totalSize_;
    std::vector<Slice> slices;
    for (u32 i = 0; i< noPower_; i++){
        Slice temp;
        temp.offset = i * totalSize_;
        temp.size = totalSize_;
        slices.push_back(temp);
    }
    HcclMem nhrOutput = HcclMemRange(execMem_.outputMem, baseOffset, execMem_.outputMem.size() - baseOffset);
    // CHK_RET(tempAlg->Prepare(nhrOutput, nhrOutput, nhrOutput, execMem_.count, dataType_, stream_,
    //     reductionOp_, 0, slices, baseOffset));

    std::vector<ChannelResource> nhrChannels;
    for (u32 i = 0; i< noPower_; i++){
        u32 remote = i * static_cast<u32>(pow(base, powerSteps_ )) + groupIdx;
        nhrChannels.push_back(channels_[remote]);
    }
    // return tempAlg->RunAsync(group, noPower_, nhrChannels);
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::PrepareSliceData(u32 subRank, u32 subRankSize, u32 size, u32 batchSize, std::vector<Slice> &slices)
{
    Slice temp;
    u32 power = static_cast<u32>(log2(subRankSize));
    slices.clear();
    slices.reserve(power);
    for (u32 step = 0; step < power; step++) {
        u32 sliceNum = pow(base, step);
        u32 offset = static_cast<u32>(subRank ^ (1 << step)) / sliceNum  * sliceNum;
        temp.offset = (offset * size) % batchSize;
        temp.size = sliceNum * size;
        slices.push_back(temp);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::MainRecordSub(u32 threadNum)
{
    for (u32 tidx = 0; tidx < threadNum; tidx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx_.mainThreadHandle, resCtx_.subThreadHandles[tidx], resCtx_.mainNotifyIds[tidx])));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::SubWaitMain(u32 threadNum)
{
    // 从thread等待主thread的record
    for (u32 tidx = 0; tidx < threadNum; tidx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx_.subThreadHandles[tidx], resCtx_.mainNotifyIds[tidx], CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::MainWaitSub(u32 threadNum)
{
    for (u32 tidx = 0; tidx < threadNum; tidx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx_.mainThreadHandle, resCtx_.subNotifyIds[tidx], CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::SubRecordMain(u32 threadNum)
{
    for (u32 tidx = 0; tidx < threadNum; tidx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx_.subThreadHandles[tidx], resCtx_.mainThreadHandle, resCtx_.subNotifyIds[tidx])));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunBetweenStep(u32 neighCur, u32 neighNext)
{
    CHK_RET(MainRecordSub(1));
    CHK_RET(SubWaitMain(1));

    CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.subThreadHandles[0], channels_[neighCur].handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.subThreadHandles[0], channels_[neighCur].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[neighNext].handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[neighNext].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    CHK_RET(SubRecordMain(1));
    CHK_RET(MainWaitSub(1));

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherPower()
{
    HcclMem userMemOut = {HcclMemType::HCCL_MEM_TYPE_DEVICE, execMem_.outputPtr, totalSize_ * rankSize};

    void *remMemPtr = nullptr;
    HcclMem dst;
    HcclMem src;
    u32 dstRank;
    u32 dstGroupIdx;
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_ ));
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_ ));
    u32 revGroup = 0;
    CHK_RET(ReverseId(groupIdx, revGroup));
    CHK_RET(PrepareSliceData(revGroup, pow(base, powerSteps_ ), noPower_ * totalSize_, totalSize_ * rankSize / static_cast<u32>(pow(base, finalSteps_)), slicePower_));
    for (u32 step = 0; step < powerSteps_ - finalSteps_; step++) {
        dstGroupIdx = groupIdx  ^ (1 << (powerSteps_ - 1 - step));
        dstRank = group * pow(base, powerSteps_ ) + dstGroupIdx;
        if (step == 0) {
            CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_ACK));
            CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        }
        remMemPtr = static_cast<u8 *>(channels_[dstRank].remoteBuffer.addr) + channels_[dstRank].remoteBuffer.offset;
        HcclMem dst = HcclMemRange(execMem_.outputMem, slicePower_[step].offset, slicePower_[step].size);
        void *srcPtr = static_cast<u8 *>(remMemPtr) + slicePower_[step].offset;
        void *dstPtr = dst.addr;
        CHK_RET(HcommReadOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, dstPtr, srcPtr, slicePower_[step].size));
        if (step != (powerSteps_ - finalSteps_ - 1)) {
            CHK_RET(RunBetweenStep(dstRank,
                group * pow(base, powerSteps_ ) + (groupIdx  ^ (1 << (powerSteps_ - 1 - step - 1)))));
        } else {
            CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL));
            CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherLastTwo()
{
    //mesh
    HcclMem userMemOut = {HcclMemType::HCCL_MEM_TYPE_DEVICE, execMem_.outputPtr, totalSize_ * rankSize};
    HcclMem emptyMem = HcclMemRange(execMem_.outputMem, 0, 0);
    CHK_RET(MainRecordSub(base));
    CHK_RET(SubWaitMain(base));

    u32 subGroupSize = static_cast<u32>(pow(base, base));
    CHK_PRT_RET(SubThreadNum < (subGroupSize - 1),
        HCCL_ERROR("[AllGatherHDStage][RunAllGatherLastTwo]rank[%u] count[%llu] failed",
            rank, execMem_.count),  HCCL_E_RESERVED);
    for (u32 round = 1; round < subGroupSize ; round++) {
        u32 dstRank = rank / subGroupSize  * subGroupSize  + BackwardRank(rank % subGroupSize , subGroupSize , round);
        ThreadHandle& subThreadHandle = round == (subGroupSize  - 1) ? resCtx_.mainThreadHandle:resCtx_.subThreadHandles[round - 1];
        CHK_RET(HcommChannelNotifyRecordOnThread(subThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(subThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }

    CHK_RET(SubRecordMain(base));
    CHK_RET(MainWaitSub(base));

    CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, emptyMem.addr, emptyMem.addr, emptyMem.size));

    CHK_RET(SubWaitMain(SubThreadNum));
    CHK_RET(MainRecordSub(SubThreadNum));

    if (execMem_.outputMem.addr != userMemOut.addr) {
        HcclMem src = HcclMemRange(execMem_.outputMem, 0, totalSize_ * rankSize / subGroupSize);
        HcclMem dst = HcclMemRange(userMemOut, totalSize_ * rankSize / subGroupSize  * (resMap[rank % subGroupSize]),
            totalSize_ * rankSize / subGroupSize);
        CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst.addr, src.addr, src.size));
    }

    for (u32 round = 1; round < subGroupSize ; round++) {
        u32 dstRank = rank / subGroupSize  * subGroupSize + BackwardRank(rank % subGroupSize , subGroupSize , round);
        ThreadHandle& subThreadHandle = resCtx_.subThreadHandles[round - 1];
        void *remMemPtr = static_cast<u8 *>(channels_[dstRank].remoteBuffer.addr) + channels_[dstRank].remoteBuffer.offset;
        HcclMem dst = HcclMemRange(userMemOut, totalSize_ * rankSize / subGroupSize  * (resMap[dstRank % subGroupSize]), totalSize_ * rankSize / subGroupSize);
        void *srcPtr = static_cast<u8 *>(remMemPtr);
        CHK_RET(HcommReadOnThread(subThreadHandle, channels_[dstRank].handle, dst.addr, srcPtr, dst.size));
        CHK_RET(HcommChannelNotifyRecordOnThread(subThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(subThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
    }
    CHK_RET(SubRecordMain(SubThreadNum));
    CHK_RET(MainWaitSub(SubThreadNum));
    CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, emptyMem.addr, emptyMem.addr, emptyMem.size));
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherLastOne()
{
    HcclMem userMemOut = {HcclMemType::HCCL_MEM_TYPE_DEVICE, execMem_.outputPtr, totalSize_ * rankSize};
    u32 group = rank / static_cast<u32>(pow(base, powerSteps_ ));
    u32 groupIdx = rank % static_cast<u32>(pow(base, powerSteps_ ));
    HcclMem emptyMem = HcclMemRange(execMem_.outputMem, 0, 0);
    CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, emptyMem.addr, emptyMem.addr, emptyMem.size));

    u32 dstRank = group * pow(base, powerSteps_ ) + (groupIdx ^ (1 << 0));
    CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_ACK));
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));

    CHK_RET(MainRecordSub(1));
    CHK_RET(SubWaitMain(1));
    //本地拷贝
    if (execMem_.outputMem.addr != userMemOut.addr) {
        HcclMem src = HcclMemRange(execMem_.outputMem, 0, totalSize_ * rankSize / base);
        HcclMem dst = HcclMemRange(userMemOut, totalSize_ * rankSize / base  * (rank % base), totalSize_ * rankSize / base);
        CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst.addr, src.addr, src.size));
    }
    // 对端拷到usrout上
    void *remMemPtr = static_cast<u8 *>(channels_[dstRank].remoteBuffer.addr) + channels_[dstRank].remoteBuffer.offset;
    HcclMem dst = HcclMemRange(userMemOut, totalSize_ * rankSize / base  * (1 - (rank % base)), totalSize_ * rankSize / base);
    void *srcPtr = static_cast<u8 *>(remMemPtr) + 0;
    CHK_RET(HcommReadOnThread(resCtx_.subThreadHandles[0], channels_[dstRank].handle, dst.addr, srcPtr, dst.size));

    CHK_RET(SubRecordMain(1));
    CHK_RET(MainWaitSub(1));

    CHK_RET(HcommChannelNotifyRecordOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL));
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx_.mainThreadHandle, channels_[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));

    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherLast()
{
    HcclMem userMemOut = {HcclMemType::HCCL_MEM_TYPE_DEVICE, execMem_.outputPtr, totalSize_ * rankSize};
    if (execMem_.outputMem.addr != userMemOut.addr) {
        HcclMem src = HcclMemRange(execMem_.outputMem, 0, totalSize_ * rankSize);
        HcclMem dst = HcclMemRange(userMemOut, 0, totalSize_ * rankSize);
        CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dst.addr, src.addr, src.size));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAllGatherStage()
{
    HCCL_INFO("RunAllGatherStage run: rank[%u] totalrank[%u] outputMem[%p] count[%llu]",
        rank, rankSize, execMem_.outputMem.addr, execMem_.count);
    u32 unitSize = SIZE_TABLE[dataType_];
    totalSize_ = unitSize * execMem_.count;
    // 对应因式分解中的2的幂次部分
    powerSteps_ = static_cast<u32>(log2(rankSize & (-rankSize)));
    if (powerSteps_ >= base) {
        finalSteps_ = base;
    } else if (powerSteps_ >= 1) {
        finalSteps_ = 1;
    }
    // 对应因式分解中的奇数部分
    noPower_ = rankSize / (rankSize & (-rankSize));
    CHK_RET(RunPreCopy());
    if (noPower_ > 1) {
        CHK_RET(RunAllGatherNoPower());
    }
    if ((powerSteps_ - finalSteps_)>= 1) {
        CHK_RET(RunAllGatherPower());
    }
    if (finalSteps_ == base){
        CHK_RET(RunAllGatherLastTwo());
    } else if (finalSteps_ == 1) {
        CHK_RET(RunAllGatherLastOne());
    } else {
        CHK_RET(RunAllGatherLast());
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherHDStage::RunAsync()
{
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("AllGatherHDStage run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank, rankSize, execMem_.inputMem.addr, execMem_.outputMem.addr, execMem_.count);

    ret = RunAllGatherStage();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherHDStage][RunAsync]rank[%u] count[%llu] failed step",
            rank, execMem_.count), ret);

    HCCL_INFO("AllGatherHDStage finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
