#include "allgather_batch_small_count_executor.h"

#include <algorithm>

#include "all_gather_hd_stage_core.h"
#include "log.h"

namespace ops_hccl_allgatherbatch {

AllGatherBatchSmallCountExecutor::AllGatherBatchSmallCountExecutor(
    const OpParam &param, AlgResourceCtx &resCtx, BatchCallProfiling &profiling)
    : param_(param), resCtx_(resCtx), profiling_(profiling)
{
}

HcclResult AllGatherBatchSmallCountExecutor::Orchestrate()
{
    uint64_t startus = GetCurrentTimeUs();

    const u64 count = param_.items[0].sendCount;
    const HcclDataType dataType = param_.items[0].dataType;

    HcclResult ret = HCCL_SUCCESS;
    std::vector<ChannelResource> channels_(param_.topoInfo.rankSize);
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        ChannelResource &channel = GetChannel(resCtx_, idx);
        channels_[channel.remoteRank] = channel;
    }
    ret = RunLoop(channels_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][Orchestrate]AllGather executor kernel run failed, ret[%d]",
            ret), ret);

    HCCL_INFO("tag[%s], Allgather executor orchestrate success, take time [%llu]us",
        param_.tag, GetCurrentTimeUs() - startus);
    return HCCL_SUCCESS;
}

u64 AllGatherBatchSmallCountExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / (unitSize * param_.topoInfo.rankSize);
    // if (param_.topoInfo.rankSize % HCCL_DEVICE_NUM_FOUR == 0) {
    //     maxCountPerLoop = maxCountPerLoop * HCCL_DEVICE_NUM_FOUR;
    // } else if (param_.topoInfo.rankSize % HCCL_DEVICE_NUM_TWO == 0){
    //     maxCountPerLoop = maxCountPerLoop * HCCL_DEVICE_NUM_TWO;
    // }
    HCCL_INFO("[AllGatherBatchSmallCountExecutor][CalcLoopMaxCount]" \
        "maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}

HcclResult AllGatherBatchSmallCountExecutor::RunLoop(std::vector<ChannelResource> &channels)
{
    const BatchItemParam &item = param_.items[0];
    const u64 count = item.sendCount;
    const HcclDataType dataType = item.dataType;
    u32 unitSize = SIZE_TABLE[dataType];

    u8 *curInputPtr = static_cast<u8 *>(item.sendBuf);
    u8 *curOutputPtr = static_cast<u8 *>(item.recvBuf);
    void *commInputPtr = resCtx_.localBuffer.addr;
    u8 *commOutputPtr = static_cast<u8 *>(resCtx_.localBuffer.addr) + resCtx_.localBuffer.offset;
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);
    CHK_PTR_NULL(commInputPtr);
    CHK_PTR_NULL(commOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(resCtx_.localBuffer.size / 2, unitSize);
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[AllGatherBatchSmallCountExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
            param_.tag, param_.topoInfo.rankSize, maxCountPerLoop),
        HCCL_E_PARA);

    for (u64 countLeft = count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[AllGatherBatchSmallCountExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d]",
            param_.tag, inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, dataType);

        // 执行
        if (useCCLBuffer) {
            // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
            void *srcPtr = curInputPtr;
            void *dstPtr = commInputPtr;
            CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dstPtr, srcPtr, curSize));
            HCCL_DEBUG("[AllGatherBatchSmallCountExecutor][RunLoop]copy from user in to ccl in.");
        }

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        ExecMem execMem;
        execMem.count = curCount;
        execMem.dataType = dataType;
        execMem.inputMem = {HCCL_MEM_TYPE_DEVICE, commInputPtr, curSize};
        u32 sliceNum = param_.topoInfo.rankSize;
        execMem.outputMem = {HCCL_MEM_TYPE_DEVICE, commOutputPtr, curSize * sliceNum};
        execMem.inputPtr = useCCLBuffer ? commInputPtr : curInputPtr;
        execMem.outputPtr = useCCLBuffer ? commOutputPtr : curOutputPtr;
        HcclResult ret = HCCL_SUCCESS;
        ret = KernelRun(execMem, channels);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherBatchSmallCountExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d].",
            HCCL_ERROR_CODE(ret), param_.tag, commInputPtr, commOutputPtr,
            curCount, dataType), ret);

        if (useCCLBuffer) {
            // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
            for (u32 i = 0; i < param_.topoInfo.rankSize; i++) {
                // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
                void *dstPtr = curOutputPtr + count * unitSize * i;
                void *srcPtr = commOutputPtr + curSize * i;
                CHK_RET(HcommLocalCopyOnThread(resCtx_.mainThreadHandle, dstPtr, srcPtr, curSize));
            }
        }

        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBatchSmallCountExecutor::KernelRun(ExecMem &execMem, std::vector<ChannelResource> &channels)
{
    AllGatherHDStage hdStageCore(param_, resCtx_, execMem, channels);
    const uint64_t hdStageStartUs = GetCurrentTimeUs();
    HcclResult commRet = hdStageCore.RunAsync();
    profiling_.hdStageUs += (GetCurrentTimeUs() - hdStageStartUs);
    CHK_RET(commRet);

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl_allgatherbatch
