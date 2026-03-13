#include "task_check_op_semantics.h"

#include <queue>

#include "checker_def.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include "allreduce_semantics_checker.h"
#include "allgather_semantics_checker.h"
#include "allgather_v_semantics_checker.h"
#include "reduce_scatter_semantics_checker.h"
#include "reduce_scatter_v_semantics_checker.h"
#include "all2all_semantics_checker.h"
#include "all2allv_semantics_checker.h"
#include "send_recv_semantics_checker.h"
#include "broadcast_semantics_checker.h"
#include "reduce_semantics_checker.h"
#include "scatter_semantics_checker.h"
#include "batchsendrecv_semantics_checker.h"
#ifdef HCCL_ALG_ANALYZER_DAVID
#include "alg_adapt_v2_interface.h"
#endif

using namespace Hccl;

namespace checker {

void TaskCheckOpSemantics::InitInputBuffer()
{
    for (auto &child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        CalcInputOutputSize(*checkerOpParam_, graphHead_->children.size(), inputDataSize_, outputDataSize_, rankId);
        BufferSemantic inputInitStatus(0, inputDataSize_);
        inputInitStatus.srcBufs.insert(SrcBufDes(rankId, BufferType::INPUT, 0));
        allRankMemSemantics_[rankId][BufferType::INPUT].insert(inputInitStatus);
    }
    return;
}

void TaskCheckOpSemantics::InitInputBuffer(RankId root)
{
    for (auto &child : graphHead_->children) {
        RankId rankId = child->rankIdx;
        if (rankId == root) {
            CalcInputOutputSize(*checkerOpParam_, graphHead_->children.size(), inputDataSize_, outputDataSize_, rankId);
            BufferSemantic inputInitStatus(0, inputDataSize_);
            inputInitStatus.srcBufs.insert(SrcBufDes(rankId, BufferType::INPUT, 0));
            allRankMemSemantics_[rankId][BufferType::INPUT].insert(inputInitStatus);
        }
    }
    return;
}

bool TaskCheckOpSemantics::IsReadyForSimulate(const TaskNode *node, std::set<TaskNode *> &simulatedNodes) const
{
    for (auto &parent : node->parents) {
        if (simulatedNodes.count(parent) == 0) {
            return false;
        }
    }
    return true;
}

HcclResult TaskCheckOpSemantics::CheckBufSemantics(std::vector<BufferSemantic *> &bufSemantics, u64 startAddr,
                                                   u64 size, bool ignoreError) const
{
    if (bufSemantics.size() == 0) {
        if (!ignoreError) {
            DUMP_AND_ERROR("When check buf semantics in range, buf semantics to check is empty");
        } else {
            CHECKER_WARNING("When check buf semantics in range, buf semantics to check is empty");
        }
        return HcclResult::HCCL_E_PARA;
    }

    u64             totalSize = 0;
    BufferSemantic *pre       = bufSemantics[0];
    // 头部有空档
    if (pre->startAddr > startAddr) {
        if (!ignoreError) {
            DUMP_AND_ERROR("When check buf semantics in range, there is blank in head. "
                "startAddr should be %llu, while first semantic startAddr is %llu", startAddr, pre->startAddr);
        } else {
            CHECKER_WARNING("When check buf semantics in range, there is blank in head. "
                "startAddr should be %llu, while first semantic startAddr is %llu", startAddr, pre->startAddr);
        }
        return HcclResult::HCCL_E_PARA;
    }
    if (pre->startAddr + pre->size >= startAddr + size) {
        totalSize += size;
    } else {
        totalSize += pre->startAddr + pre->size - startAddr;
    }

    for (size_t index = 1; index < bufSemantics.size(); index++) {
        BufferSemantic *cur = bufSemantics[index];
        if (cur->startAddr != pre->startAddr + pre->size) {
            if (!ignoreError) {
                DUMP_AND_ERROR("there is blank in middle, pre semantic endAddr is %llu, cur semantic startAddr is %llu, "
                    "they should be equal", pre->startAddr + pre->size, cur->startAddr);
            } else {
                CHECKER_WARNING("there is blank in middle, pre semantic endAddr is %llu, cur semantic startAddr is %llu, "
                    "they should be equal", pre->startAddr + pre->size, cur->startAddr);
            }
            return HcclResult::HCCL_E_PARA;
        }
        if (cur->startAddr + cur->size >= startAddr + size) {
            totalSize += startAddr + size - cur->startAddr;
        } else {
            totalSize += cur->size;
        }

        pre = cur;
    }

    if (totalSize != size) {
        if (!ignoreError) {
            DUMP_AND_ERROR("When check buf semantics in range, there is blank in tail. "
                "endAddr should be %llu, while last semantic endAddr is %llu",
                startAddr + size, startAddr + totalSize);
        } else {
            CHECKER_WARNING("When check buf semantics in range, there is blank in tail. "
                "endAddr should be %llu, while last semantic endAddr is %llu",
                startAddr + size, startAddr + totalSize);
        }
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

// 获取slicePair和srcBufSemantic的交集区域
void TaskCheckOpSemantics::GetSrcIntersectionAddr(SliceOpPair &slicePair, const BufferSemantic &srcBufSemantic,
                                                  u64 &srcStartAddr, u64 &srcEndAddr) const
{
    srcStartAddr = slicePair.srcSlice.GetOffset();
    srcEndAddr   = srcStartAddr + slicePair.srcSlice.GetSize();
    if (srcBufSemantic.startAddr > srcStartAddr) {
        srcStartAddr = srcBufSemantic.startAddr;
    }
    if (srcBufSemantic.startAddr + srcBufSemantic.size < srcEndAddr) {
        srcEndAddr = srcBufSemantic.startAddr + srcBufSemantic.size;
    }
    return;
}

void TaskCheckOpSemantics::GetAffectedBufSemantics(SliceOpPair &slicePair, const BufferSemantic &srcBufSemantic,
                                                   std::vector<BufferSemantic *> &affectedDstBufSemantics)
{
    u64 srcStartAddr;
    u64 srcEndAddr;
    GetSrcIntersectionAddr(slicePair, srcBufSemantic, srcStartAddr, srcEndAddr);

    RankId     dstRank    = slicePair.dstRank;
    BufferType dstBufType = slicePair.dtsSlice.GetType();

    s64 dstSrcOffset = slicePair.dtsSlice.GetOffset() - slicePair.srcSlice.GetOffset();
    u64 dstStartAddr = srcStartAddr + dstSrcOffset;
    u64 dstEndAddr   = srcEndAddr + dstSrcOffset;

    for (auto &ele : allRankMemSemantics_[dstRank][dstBufType]) {
        if (ele.startAddr + ele.size <= dstStartAddr) {
            continue;
        }
        if (ele.startAddr >= dstEndAddr) {
            continue;
        }
        affectedDstBufSemantics.push_back(const_cast<BufferSemantic *>(&ele));
    }
    return;
}

void TaskCheckOpSemantics::GetAffectedBufSemantics(SliceOpPair &slicePair,
                                                   std::vector<BufferSemantic *> &affectedDstBufSemantics)
{
    u64 dstStartAddr = slicePair.dtsSlice.GetOffset();
    u64 dstEndAddr   = dstStartAddr + slicePair.dtsSlice.GetSize();

    RankId     dstRank    = slicePair.dstRank;
    BufferType dstBufType = slicePair.dtsSlice.GetType();

    for (auto &ele : allRankMemSemantics_[dstRank][dstBufType]) {
        if (ele.startAddr + ele.size <= dstStartAddr) {
            continue;
        }
        if (ele.startAddr >= dstEndAddr) {
            continue;
        }
        affectedDstBufSemantics.push_back(const_cast<BufferSemantic *>(&ele));
    }
    return;
}

// 因为srcBufSemantic与affectedDstBufSemantics可能会有重叠，affectedDstBufSemantics处理过程中会被修改，因此srcBufSemantic不能传引用
void TaskCheckOpSemantics::ApplyOverrideSrcBufSemantic(SliceOpPair &slicePair, const BufferSemantic srcBufSemantic)
{
    u64 srcStartAddr;
    u64 srcEndAddr;
    GetSrcIntersectionAddr(slicePair, srcBufSemantic, srcStartAddr, srcEndAddr);

    s64 dstSrcOffset = slicePair.dtsSlice.GetOffset() - slicePair.srcSlice.GetOffset();
    u64 dstStartAddr = srcStartAddr + dstSrcOffset;
    u64 dstEndAddr   = srcEndAddr + dstSrcOffset;

    RankId                    dstRank              = slicePair.dstRank;
    BufferType                dstBufType           = slicePair.dtsSlice.GetType();
    std::set<BufferSemantic> &targetBufferSemantic = allRankMemSemantics_[dstRank][dstBufType];

    BufferSemantic dstBufSemantic(dstStartAddr, dstEndAddr - dstStartAddr, srcBufSemantic.isReduce,
        srcBufSemantic.reduceType, srcBufSemantic.srcBufs);
    // 因为可能使用的是srcBufSemantic的一部分，所以这边需要处理一下源地址
    for (auto &srcBuf : dstBufSemantic.srcBufs) {
        srcBuf.srcAddr += srcStartAddr - srcBufSemantic.startAddr;
    }
    dstBufSemantic.affectedGlobalSteps.push_back(globalStep_);

    targetBufferSemantic.insert(dstBufSemantic);

    return;
}

HcclResult TaskCheckOpSemantics::ReduceToAffectedBufSemantic(const BufferSemantic         &srcBufSemantic,
                                                             std::vector<BufferSemantic *> toAddReduceInfoSemantics,
                                                             u64                           srcStartAddr)
{
    u64 srcOffset = srcStartAddr - srcBufSemantic.startAddr;
    for (auto &ele : toAddReduceInfoSemantics) {
        if (ele->srcBufs.size() == 1) {
            if (ele->isReduce == true) {
                DUMP_AND_ERROR("buffer semantic srcBufs size is 1, but isReduce is true");
                DUMP_AND_ERROR("invalid buffer semantic is %s", ele->Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            ele->isReduce   = true;
            ele->reduceType = reduceType_;
        }
        if (srcBufSemantic.srcBufs.size() > 1 &&
            ele->reduceType != srcBufSemantic.reduceType) {
            DUMP_AND_ERROR("reduceType is different");
            DUMP_AND_ERROR("src buf semantic is %s", srcBufSemantic.Describe().c_str());
            DUMP_AND_ERROR("dst buf semantic is %s", ele->Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        for (auto srcBuf : srcBufSemantic.srcBufs) {
            // 校验重复reduce的场景
            srcBuf.srcAddr += srcOffset;
            u32 beforeInsertCnt = ele->srcBufs.size();
            ele->srcBufs.insert(srcBuf);
            u32 afterInsertCnt = ele->srcBufs.size();
            if (beforeInsertCnt == afterInsertCnt) {
                DUMP_AND_ERROR("after add reduce srcBuf %s, the size of dst srcBufs not changed", srcBuf.Describe().c_str());
                DUMP_AND_ERROR("src buf semantic is %s", srcBufSemantic.Describe().c_str());
                DUMP_AND_ERROR("dst buf semantic is %s", ele->Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
        }

        // 用于图形化界面展示，添加影响的节点
        ele->affectedGlobalSteps.insert(ele->affectedGlobalSteps.end(), srcBufSemantic.affectedGlobalSteps.begin(),
            srcBufSemantic.affectedGlobalSteps.end());
        srcOffset += ele->size;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::ApplyReduceSrcBufSemantic(SliceOpPair &slicePair, const BufferSemantic &srcBufSemantic,
                                                           std::vector<BufferSemantic *> &affectedDstBufSemantics)
{
    u64 srcStartAddr;
    u64 srcEndAddr;
    GetSrcIntersectionAddr(slicePair, srcBufSemantic, srcStartAddr, srcEndAddr);

    s64 dstSrcOffset = slicePair.dtsSlice.GetOffset() - slicePair.srcSlice.GetOffset();
    u64 dstStartAddr = srcStartAddr + dstSrcOffset;
    u64 dstEndAddr   = srcEndAddr + dstSrcOffset;

    // 校验目的地是否已经有了数据
    auto ret = CheckBufSemantics(affectedDstBufSemantics, dstStartAddr, dstEndAddr - dstStartAddr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::GEN_FAILED_INCOMPLETE_SLICE);
        DUMP_AND_ERROR("failed to check dst buf semantics in src semantic range, src buf semantic is %s, "
            "affected dst buf semantics are as follows:", srcBufSemantic.Describe().c_str());
        for (auto &ele : affectedDstBufSemantics) {
            DUMP_AND_ERROR("    %s", ele->Describe().c_str());
        }
        return ret;
    }

    RankId     dstRank    = slicePair.dstRank;
    BufferType dstBufType = slicePair.dtsSlice.GetType();
    // 分割尾节点
    BufferSemantic &lastDstBuf = *(affectedDstBufSemantics.back());
    if (lastDstBuf.startAddr + lastDstBuf.size > dstEndAddr) {
        BufferSemantic dstBufSemantic(dstEndAddr, lastDstBuf.startAddr + lastDstBuf.size - dstEndAddr,
                                      lastDstBuf.isReduce, lastDstBuf.reduceType, lastDstBuf.srcBufs);
        for (auto &srcBuf : dstBufSemantic.srcBufs) {
            srcBuf.srcAddr += dstEndAddr - lastDstBuf.startAddr;
        }
        dstBufSemantic.affectedGlobalSteps = lastDstBuf.affectedGlobalSteps;

        allRankMemSemantics_[dstRank][dstBufType].insert(dstBufSemantic);

        // 修改原来的节点
        lastDstBuf.size = dstEndAddr - lastDstBuf.startAddr;
    }

    // 拆分首节点。（必须先新增尾节点，再修改首节点，因为fistDstBuf和lastDstBuf可能指向同一个对象）
    BufferSemantic &fistDstBuf = *(affectedDstBufSemantics[0]);
    if (fistDstBuf.startAddr < dstStartAddr) {
        BufferSemantic dstBufSemantic(dstStartAddr, fistDstBuf.startAddr + fistDstBuf.size - dstStartAddr,
                                      fistDstBuf.isReduce, fistDstBuf.reduceType, fistDstBuf.srcBufs);
        for (auto &srcBuf : dstBufSemantic.srcBufs) {
            srcBuf.srcAddr += dstStartAddr - fistDstBuf.startAddr;
        }
        dstBufSemantic.affectedGlobalSteps = fistDstBuf.affectedGlobalSteps;

        allRankMemSemantics_[dstRank][dstBufType].insert(dstBufSemantic);

        fistDstBuf.size = dstStartAddr - fistDstBuf.startAddr;
    }

    std::vector<BufferSemantic *> toAddReduceInfoSemantics;
    GetAffectedBufSemantics(slicePair, srcBufSemantic, toAddReduceInfoSemantics);

    ret = ReduceToAffectedBufSemantic(srcBufSemantic, toAddReduceInfoSemantics, srcStartAddr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::GEN_FAILED_MODIFY_SEMANTIC_FAILED);
        DUMP_AND_ERROR("reduce to affected buf semantics failed, src buf semantic is %s, affected dst buf semantics are as follows:", srcBufSemantic.Describe().c_str());
        for (auto &ele : toAddReduceInfoSemantics) {
            DUMP_AND_ERROR("    %s", ele->Describe().c_str());
        }
        return ret;
    }

    return HcclResult::HCCL_SUCCESS;
}

void TaskCheckOpSemantics::RemoveAffectedBufSemantics(SliceOpPair                   &slicePair,
                                                      std::vector<BufferSemantic *> &affectedDstBufSemantics)
{
    u64 dstStartAddr = slicePair.dtsSlice.GetOffset();
    u64 dstEndAddr   = dstStartAddr + slicePair.dtsSlice.GetSize();

    RankId                    dstRank              = slicePair.dstRank;
    BufferType                dstBufType           = slicePair.dtsSlice.GetType();
    std::set<BufferSemantic> &targetBufferSemantic = allRankMemSemantics_[dstRank][dstBufType];

    // 一种特殊情况，对应的dst还未创建起来
    if (affectedDstBufSemantics.size() == 0) {
        return;
    }

    // 新增尾节点
    BufferSemantic &lastDstBuf = *(affectedDstBufSemantics.back());
    if (lastDstBuf.startAddr + lastDstBuf.size > dstEndAddr) {
        BufferSemantic dstBufSemantic(dstEndAddr, lastDstBuf.startAddr + lastDstBuf.size - dstEndAddr,
                                      lastDstBuf.isReduce, lastDstBuf.reduceType, lastDstBuf.srcBufs);
        for (auto &srcBuf : dstBufSemantic.srcBufs) {
            srcBuf.srcAddr += dstEndAddr - lastDstBuf.startAddr;
        }
        dstBufSemantic.affectedGlobalSteps = lastDstBuf.affectedGlobalSteps;

        targetBufferSemantic.insert(dstBufSemantic);
        // 修改原来的节点
        lastDstBuf.size = dstEndAddr - lastDstBuf.startAddr;
    }

    // 修改首节点。（必须先新增尾节点，再修改首节点，因为fistDstBuf和lastDstBuf可能指向同一个对象）
    BufferSemantic &fistDstBuf = *(affectedDstBufSemantics[0]);
    if (fistDstBuf.startAddr < dstStartAddr) {
        fistDstBuf.size = dstStartAddr - fistDstBuf.startAddr;
    }

    for (auto ele : affectedDstBufSemantics) {
        if (ele->startAddr < dstStartAddr) {
            continue;
        }
        targetBufferSemantic.erase(*ele);
    }

    return;
}

HcclResult TaskCheckOpSemantics::ApplySrcBufSemanticsToDst(SliceOpPair                  &slicePair,
                                                           std::vector<BufferSemantic *> srcBufSemantics)
{
    SliceOp sliceOp = slicePair.sliceOp;

    if (sliceOp == SliceOp::OVERRIDE) {
        std::vector<BufferSemantic *> affectedDstBufSemantics;
        GetAffectedBufSemantics(slicePair, affectedDstBufSemantics);
        RemoveAffectedBufSemantics(slicePair, affectedDstBufSemantics);

        for (auto &ele : srcBufSemantics) {
            ApplyOverrideSrcBufSemantic(slicePair, *ele);
        }
    } else if (sliceOp == SliceOp::REDUCE) {
        for (auto &ele : srcBufSemantics) {
            std::vector<BufferSemantic *> affectedDstBufSemantics;
            GetAffectedBufSemantics(slicePair, *ele, affectedDstBufSemantics);
            auto ret = ApplyReduceSrcBufSemantic(slicePair, *ele, affectedDstBufSemantics);
            if (ret != HcclResult::HCCL_SUCCESS) {
                DUMP_AND_ERROR("fail to apply reduce for src buf semantic, which is %s, "
                    "affected dst buf semantics are as follows:", ele->Describe().c_str());
                for (auto &ele : affectedDstBufSemantics) {
                    DUMP_AND_ERROR("    %s", ele->Describe().c_str());
                }
                return ret;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::ProcessSliceOpPair(SliceOpPair &slicePair)
{
    RankId     srcRank      = slicePair.srcRank;
    BufferType srcBufType   = slicePair.srcSlice.GetType();
    u64        srcStartAddr = slicePair.srcSlice.GetOffset();
    u64        srcSize      = slicePair.srcSlice.GetSize();
    if (srcSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    // 根据源slice获取源数据片
    std::vector<BufferSemantic *> srcBufSemantics;
    for (auto &ele : allRankMemSemantics_[srcRank][srcBufType]) {
        if (ele.startAddr + ele.size <= srcStartAddr) {
            continue;
        }

        if (ele.startAddr >= srcStartAddr + srcSize) {
            continue;
        }

        srcBufSemantics.push_back(const_cast<BufferSemantic *>(&ele));
    }

    auto ret = CheckBufSemantics(srcBufSemantics, srcStartAddr, srcSize, true);
    if (ret != HcclResult::HCCL_SUCCESS) {
        // 对于reduce操作，源slice对应的语义块不能有缺失，因为随机数据做reduce，可能导致概率性溢出问题，所以需要添加拦截
        if (slicePair.sliceOp == SliceOp::REDUCE) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::GEN_FAILED_INCOMPLETE_SLICE);
            DUMP_AND_ERROR("failed to check buf semantics in src slice, size= %d src slice is %s, bufSemantics in this range are as follows:",
                srcBufSemantics.size(), slicePair.srcSlice.Describe().c_str());
            for (auto &ele : srcBufSemantics) {
                DUMP_AND_ERROR("    %s", ele->Describe().c_str());
            }

            return HCCL_E_PARA;
        } else if (slicePair.sliceOp == SliceOp::OVERRIDE) {
            CHECKER_WARNING("incomplete buf semantics in src slice, which may affected performance.");
        }
    }

    // 将源数据片应用到目标数据片
    ret = ApplySrcBufSemanticsToDst(slicePair, srcBufSemantics);
    if (ret != HcclResult::HCCL_SUCCESS) {
        DUMP_AND_ERROR("failed to apply src buf semantics to dst, src slice is %s, bufSemantics in this range are as follows:",
            slicePair.srcSlice.Describe().c_str());
        for (auto &ele : srcBufSemantics) {
            DUMP_AND_ERROR("    %s", ele->Describe().c_str());
        }
        return ret;
    }
    RankId dstRank = slicePair.dstRank;
    memSemanticsChange_[dstRank] = true;

    return HcclResult::HCCL_SUCCESS;
}

void TaskCheckOpSemantics::GetSliceOpPair(TaskNode *simNode, std::vector<SliceOpPair> &sliceOpPairs) const
{
    TaskTypeStub taskType = GetNodeType(simNode);
    if (taskType == TaskTypeStub::LOCAL_COPY) {
        const TaskStubLocalCopy *task = dynamic_cast<const TaskStubLocalCopy *>(simNode->task);
        SliceOpPair    pair;
        pair.srcRank  = simNode->rankIdx;
        pair.dstRank  = simNode->rankIdx;
        pair.srcSlice = task->GetSrcSlice();
        pair.dtsSlice = task->GetDstSlice();
        pair.sliceOp  = SliceOp::OVERRIDE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::LOCAL_REDUCE) {
        const TaskStubLocalReduce *task = dynamic_cast<const TaskStubLocalReduce *>(simNode->task);
        SliceOpPair      pair;
        pair.srcRank  = simNode->rankIdx;
        pair.dstRank  = simNode->rankIdx;
        pair.srcSlice = task->GetSrcSlice();
        pair.dtsSlice = task->GetDstSlice();
        pair.sliceOp  = SliceOp::REDUCE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::READ) {
        const TaskStubRead *readTask = dynamic_cast<const TaskStubRead *>(simNode->task);
        SliceOpPair pair;
        pair.srcRank  = readTask->GetRemoteRank();
        pair.dstRank  = simNode->rankIdx;
        pair.srcSlice = readTask->GetRemoteSlice();
        pair.dtsSlice = readTask->GetLocalSlice();
        pair.sliceOp  = SliceOp::OVERRIDE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::WRITE) {
        const TaskStubWrite *writeTask = dynamic_cast<const TaskStubWrite *>(simNode->task);
        SliceOpPair pair;
        pair.srcRank  = simNode->rankIdx;
        pair.dstRank  = writeTask->GetRemoteRank();
        pair.srcSlice = writeTask->GetLocalSlice();
        pair.dtsSlice = writeTask->GetRemoteSlice();
        pair.sliceOp  = SliceOp::OVERRIDE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::READ_REDUCE) {
        const TaskStubReadReduce  *readReduceTask = dynamic_cast<const TaskStubReadReduce *>(simNode->task);
        SliceOpPair pair;
        pair.srcRank  = readReduceTask->GetRemoteRank();
        pair.dstRank  = simNode->rankIdx;
        pair.srcSlice = readReduceTask->GetRemoteSlice();
        pair.dtsSlice = readReduceTask->GetLocalSlice();
        pair.sliceOp  = SliceOp::REDUCE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::WRITE_REDUCE) {
        const TaskStubWriteReduce  *writeReduceTask = dynamic_cast<const TaskStubWriteReduce *>(simNode->task);
        SliceOpPair pair;
        pair.srcRank  = simNode->rankIdx;
        pair.dstRank  = writeReduceTask->GetRemoteRank();
        pair.srcSlice = writeReduceTask->GetLocalSlice();
        pair.dtsSlice = writeReduceTask->GetRemoteSlice();
        pair.sliceOp  = SliceOp::REDUCE;
        sliceOpPairs.push_back(pair);
    } else if (taskType == TaskTypeStub::LOCAL_BATCH_REDUCE) {
        const TaskStubLocalBatchReduce  *batchReduceTask = dynamic_cast<const TaskStubLocalBatchReduce *>(simNode->task);
        const std::vector<DataSlice>& srcSlices = batchReduceTask->GetSrcSlices();
        for (u32 i = 0; i < srcSlices.size(); i++) {
            SliceOpPair pair;
            pair.srcRank  = simNode->rankIdx;
            pair.dstRank  = simNode->rankIdx;
            pair.srcSlice = srcSlices[i];
            pair.dtsSlice = batchReduceTask->GetDstSlice();
            pair.sliceOp  = SliceOp::REDUCE;
            sliceOpPairs.push_back(pair);
        }
    }
    return;
}

void TaskCheckOpSemantics::UpdateStep(TaskNode *simNode)
{
    RankId rankId = simNode->rankIdx;
    globalStep_++;
    if (localStep_.count(rankId) == 0) {
        localStep_[rankId] = 1;
    } else {
        localStep_[rankId] += 1;
    }

    localStep2GlobalStep_[rankId][localStep_[rankId]] = globalStep_;
    globalStep2LocalStep_[globalStep_] = LocalStep{rankId, localStep_[rankId]};

    simNode->globalStep = globalStep_;
    simNode->localStep = localStep_[rankId];
    return;
}

u32 TaskCheckOpSemantics::GetCurLocalStep()
{
    // globalStep2LocalStep_ 的key一定是递增的，所以取最后一个就是当前的
    auto iter = globalStep2LocalStep_.end();
    iter--;
    return iter->second.localStep;
}

HcclResult TaskCheckOpSemantics::DumpNodeSemantics(TaskNode *simNode)
{
    RankId rankId = simNode->rankIdx;

    bool change = false;
    if (memSemanticsChange_.count(rankId) && memSemanticsChange_[rankId]) {
        change = true;
        memSemanticsChange_[rankId] = false;
    }

    DataDumper::Global()->DumpSemanticState(rankId, localStep_[rankId], globalStep_,
                                            change, allRankMemSemantics_[rankId]);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::ProcessNodeSemantics(TaskNode *simNode)
{
    // 更新localStep与globalStep
    UpdateStep(simNode);

    std::vector<SliceOpPair> sliceOpPairs;
    GetSliceOpPair(simNode, sliceOpPairs);

    HcclResult ret;
    for (auto &ele : sliceOpPairs) {
        ret = ProcessSliceOpPair(ele);
        if (ret != HcclResult::HCCL_SUCCESS) {
            simNode->genSemanticError = true; // 该节点产生语义信息失败，后续在可视化界面进行高亮
            DUMP_AND_ERROR("process SliceOpPair failed,  SliceOpPair is %s", ele.Describe().c_str());
            return ret;
        }
    }

    // 在这边dump语义信息，给可视化界面使用
    CHK_RET(DumpNodeSemantics(simNode));

    return HcclResult::HCCL_SUCCESS;
}

void TaskCheckOpSemantics::AddChildrenToQueue(TaskNode *node, std::set<TaskNode *> &visitedNodes,
    std::queue<TaskNode *> &walkQue, std::set<TaskNode *> &simulatedNodes) const
{
#ifdef HCCL_ALG_ANALYZER_DAVID
    node = Hccl::UpdateNodeForCcuGraph(node, simulatedNodes);
#endif
    for (auto &child : node->children) {
        if (visitedNodes.count(child) != 0) {
            continue;
        }
        walkQue.push(child);
        visitedNodes.insert(child);
    }
    return;
}

void TaskCheckOpSemantics::AddAivChildrenToQueue(TaskNode *node, std::set<TaskNode *> &visitedNodes,
    std::queue<TaskNode *> &walkQue, std::set<TaskNode *> &simulatedNodes) const
{
    if (node->task != nullptr && node->task->GetType() == TaskTypeStub::AIV_TASK) {
        AivTaskStub *curTask = dynamic_cast<AivTaskStub *>(node->task);
        TaskNode *aivStartNode = curTask->GetAivStart();
        curTask->GetAivEnd()->children = node->children;
        simulatedNodes.insert(aivStartNode);
        node = aivStartNode;
    }

    for (auto &child : node->children) {
        if (visitedNodes.count(child) != 0) {
            continue;
        }
        walkQue.push(child);
        visitedNodes.insert(child);
    }
    return;
}

HcclResult TaskCheckOpSemantics::GenMemSemantics()
{
    std::set<TaskNode *>   visitedNodes;
    std::queue<TaskNode *> walkQue;
    std::set<TaskNode *>   simulatedNodes;

    // graphHead_是dummy节点，需要忽略掉的
    simulatedNodes.insert(graphHead_);
    for (auto &child : graphHead_->children) {
        visitedNodes.insert(child);
        walkQue.push(child);
    }

    while (!walkQue.empty()) {
        TaskNode *curNode = walkQue.front();
        walkQue.pop();

        // 因为Send/Recv或者SendReduce/RecvReduce必须成对执行，所以队列中有些节点可能已经执行
        if (simulatedNodes.count(curNode) != 0) {
            continue;
        }

        // 父节点还没全部执行完成，不能执行
        if (!IsReadyForSimulate(curNode, simulatedNodes)) {
            walkQue.push(curNode);
            continue;
        }

        auto ret = ProcessNodeSemantics(curNode);
        if (ret != HcclResult::HCCL_SUCCESS) {
            DUMP_AND_ERROR("Process node semantics failed. Node task is %s", curNode->task->Describe().c_str());
            return ret;
        }

        AddChildrenToQueue(curNode, visitedNodes, walkQue, simulatedNodes);
        simulatedNodes.insert(curNode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::GenAivMemSemantics()
{
    std::set<TaskNode *>   visitedNodes;
    std::queue<TaskNode *> walkQue;
    std::set<TaskNode *>   simulatedNodes;

    // graphHead_是dummy节点，需要忽略掉的
    simulatedNodes.insert(graphHead_);
    for (auto &child : graphHead_->children) {
        visitedNodes.insert(child);
        walkQue.push(child);
    }

    while (!walkQue.empty()) {
        TaskNode *curNode = walkQue.front();
        walkQue.pop();

        // 因为Send/Recv或者SendReduce/RecvReduce必须成对执行，所以队列中有些节点可能已经执行
        if (simulatedNodes.count(curNode) != 0) {
            continue;
        }

        // 父节点还没全部执行完成，不能执行
        if (!IsReadyForSimulate(curNode, simulatedNodes)) {
            walkQue.push(curNode);
            continue;
        }

        // 跳过AivStart和BlockStart
        if (curNode->task != nullptr) {
            auto ret = ProcessNodeSemantics(curNode);

            if (ret != HcclResult::HCCL_SUCCESS) {
                DUMP_AND_ERROR("Process node semantics failed. Node task is %s", curNode->task->Describe().c_str());
                return ret;
            }
        }

        AddAivChildrenToQueue(curNode, visitedNodes, walkQue, simulatedNodes);
        simulatedNodes.insert(curNode);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckOpSemantics::Execute()
{
    if (opType_ == CheckerOpType::BROADCAST || opType_ == CheckerOpType::SCATTER) {
        InitInputBuffer(root_);
    } else {
        InitInputBuffer();
    }

    DataDumper::Global()->InitSemanticState();

    HcclResult ret = HcclResult::HCCL_SUCCESS;
    if (graphHead_->hasAivTask) {
        ret = GenAivMemSemantics();
    } else {
        ret = GenMemSemantics();
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        DUMP_AND_ERROR("Generate memory semantics failed.");
        return ret;
    }
    HCCL_INFO("gen GenMemSemantics success");

    if (opType_ == CheckerOpType::ALLREDUCE) {
        ret = TaskCheckAllReduceSemantics(allRankMemSemantics_, dataSize_, reduceType_);
    } else if (opType_ == CheckerOpType::ALLGATHER) {
        ret = TaskCheckAllGatherSemantics(allRankMemSemantics_, dataSize_);
    } else if (opType_ == CheckerOpType::ALLGATHER_V) {
        ret = TaskCheckAllGatherVSemantics(allRankMemSemantics_, *checkerOpParam_);
    } else if (opType_ == CheckerOpType::REDUCE_SCATTER) {
        ret = TaskCheckReduceScatterSemantics(allRankMemSemantics_, dataSize_, reduceType_);
    } else if (opType_ == CheckerOpType::REDUCE_SCATTER_V) {
        ret = TaskCheckReduceScatterVSemantics(allRankMemSemantics_, *checkerOpParam_);
    } else if (opType_ == CheckerOpType::ALLTOALL || opType_ == CheckerOpType::ALLTOALLVC) {
        ret = TaskCheckAll2AllSemantics(allRankMemSemantics_, *checkerOpParam_);
    } else if (opType_ == CheckerOpType::ALLTOALLV) {
        ret = TaskCheckAll2AllVSemantics(allRankMemSemantics_, *checkerOpParam_);
    } else if (opType_ == CheckerOpType::SEND || opType_ == CheckerOpType::RECEIVE) {
        ret = TaskCheckSendRecvSemantics(allRankMemSemantics_, dataSize_, srcRank_, dstRank_);
    } else if (opType_ == CheckerOpType::BROADCAST) {
        ret = TaskCheckBroadcastSemantics(allRankMemSemantics_, dataSize_, root_);
    } else if (opType_ == CheckerOpType::REDUCE) {
        ret = TaskCheckReduceSemantics(allRankMemSemantics_, dataSize_, reduceType_, root_);
    } else if (opType_ == CheckerOpType::SCATTER) {
        ret = TaskCheckScatterSemantics(allRankMemSemantics_, dataSize_, root_);
    } else if (opType_ == CheckerOpType::BATCH_SEND_RECV) {
        ret = TaskCheckBatchSendRecvSemantics(allRankMemSemantics_, dataSize_);
    } else {
        DUMP_AND_ERROR("unsupported op type[%s]", opType_.Describe().c_str());
        ret = HcclResult::HCCL_E_NOT_SUPPORT;
    }

    if (ret != HcclResult::HCCL_SUCCESS) {
        DUMP_AND_ERROR("Check memory semantics failed.");
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace checker