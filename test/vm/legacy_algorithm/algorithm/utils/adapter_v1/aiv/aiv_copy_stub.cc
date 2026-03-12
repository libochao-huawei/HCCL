#include "aiv_copy_stub.h"
#include "llt_common.h"
#include "checker_data_slice.h"
#include "mem_layout.h"
#include "rank_info_recorder.h"
#include "task_stub.h"
#include "aiv_task_stub.h"
#include "aiv_task_queue_stub.h"
#include "utils_stub.h"
#include "checker_def.h"

using namespace checker;
using namespace hccl;

namespace AscendC {

CheckerReduceOp g_atomicType = CheckerReduceOp::REDUCE_RESERVED;

void DataCopyTask(const RankId curRank, const RankId srcRank, const RankId dstRank, const DataSlice &srcSlice,
    const DataSlice &dstSlice, const BlockId block, const pipe_t pipet, bool isCp2UB, bool isGenfromSync)
{
    LinkInfoStub link(LinkProtoStub::SDMA);
    CheckerDataType dataType = MemLayout::Global()->GetCheckerDataType();
    std::shared_ptr<TaskStub> task = nullptr;
    if (srcRank == dstRank) {  // 本地拷贝的场景
        if (isCp2UB || g_atomicType == CheckerReduceOp::REDUCE_RESERVED) {
            task.reset(new TaskStubLocalCopy(srcSlice, dstSlice, isGenfromSync));
        } else {
            task.reset(new TaskStubLocalReduce(srcSlice, dstSlice, dataType, g_atomicType, isGenfromSync));
        }
    } else if (curRank == srcRank) {  // 写操作
        if (isCp2UB || g_atomicType == CheckerReduceOp::REDUCE_RESERVED) {
            task.reset(new TaskStubWrite(dstRank, link, srcSlice, dstSlice, isGenfromSync));
        } else {
            task.reset(new TaskStubWriteReduce(dstRank, link, srcSlice, dstSlice, dataType, g_atomicType, isGenfromSync));
        }
    } else if (curRank == dstRank) {  // 读操作
        if (isCp2UB || g_atomicType == CheckerReduceOp::REDUCE_RESERVED) {
            task.reset(new TaskStubRead(srcRank, link, dstSlice, srcSlice, isGenfromSync));
        } else {
            task.reset(new TaskStubReadReduce(dstRank, link, srcSlice, dstSlice, dataType, g_atomicType, isGenfromSync));
        }
    }
    AivTaskQueueStub::AppendAivTask(curRank, block, pipet, task);
}

template <typename T>
__aicore__ void DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                const uint32_t calCount, bool isGenfromSync)
{
    RankId srcRank = 0;
    DataSlice srcSlice;
    BlockId block = 0;
    RankId dstRank = 0;
    DataSlice dstSlice;
    u64 size = calCount * sizeof(T);

    CHK_RET_NULL(MemLayout::Global()->AivGetSlice((char*)srcLocal.ptr_, size, srcSlice, &srcRank, &block));
    CHK_RET_NULL(MemLayout::Global()->GetSlice((char*)dstGlobal.ptr_, size, dstSlice, &dstRank));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET_NULL(hccl::CheckCurRankId(curRank, srcRank, dstRank));

    // 忽略空拷贝操作
    if (calCount == 0) {
        return;
    }

    DataCopyTask(curRank, srcRank, dstRank, srcSlice, dstSlice, block, PIPE_MTE3, false, isGenfromSync);
}

template <typename T>
__aicore__ void DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                const uint32_t calCount, bool isGenfromSync)
{
    if (MemLayout::Global()->GetBufferType((u64)srcGlobal.ptr_) == BufferType::AIV_COMMINFO) {
        u64 dstAddr = 0;
        u64 dstSize = 0;
        if (MemLayout::Global()->GetRealAddr((u64)dstLocal.ptr_, dstAddr, dstSize) == HCCL_E_PARA) {
            MemLayout::Global()->MemAlloc((u64)dstLocal.ptr_, (u64)(calCount * sizeof(T)));
        }
        MemLayout::Global()->GetRealAddr((u64)dstLocal.ptr_, dstAddr, dstSize);
        u64 srcAddr = 0;
        u64 srcSize = 0;
        MemLayout::Global()->GetRealAddr((u64)srcGlobal.ptr_, srcAddr, srcSize);

        if (memcpy_s((char *)dstAddr, dstSize, (char *)srcAddr, calCount * sizeof(T)) != 0) {
            HCCL_ERROR("DataCopy failed!");
            return;
        }
        return;
    }

    RankId srcRank = 0;
    DataSlice srcSlice;
    BlockId block = 0;
    RankId dstRank = 0;
    DataSlice dstSlice;
    u64 size = calCount * sizeof(T);

    CHK_RET_NULL(MemLayout::Global()->GetSlice((char*)srcGlobal.ptr_, size, srcSlice, &srcRank));
    CHK_RET_NULL(MemLayout::Global()->AivGetSlice((char*)dstLocal.ptr_, size, dstSlice, &dstRank, &block));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET_NULL(hccl::CheckCurRankId(curRank, srcRank, dstRank));

    // 忽略空拷贝操作
    if (calCount == 0) {
        return;
    }

    DataCopyTask(curRank, srcRank, dstRank, srcSlice, dstSlice, block, PIPE_MTE2, true, isGenfromSync);
}

template <typename T>
__aicore__  void DataCopyPad(const GlobalTensor<T>& dstGlobal,
                                    const LocalTensor<T>& srcLocal,
                                    const DataCopyExtParams& dataCopyParams,
                                    bool isGenfromSync)
{
    RankId srcRank = 0;
    DataSlice srcSlice;
    BlockId block = 0;
    RankId dstRank = 0;
    DataSlice dstSlice;
    u64 size = dataCopyParams.blockCount * dataCopyParams.blockLen;

    CHK_RET_NULL(MemLayout::Global()->AivGetSlice((char*)srcLocal.ptr_, size, srcSlice, &srcRank, &block));
    CHK_RET_NULL(MemLayout::Global()->GetSlice((char*)dstGlobal.ptr_, size, dstSlice, &dstRank));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET_NULL(hccl::CheckCurRankId(curRank, srcRank, dstRank));

    // 忽略空拷贝操作
    if (dataCopyParams.blockCount == 0 || dataCopyParams.blockLen == 0) {
        return;
    }

    DataCopyTask(curRank, srcRank, dstRank, srcSlice, dstSlice, block, PIPE_MTE3, false, isGenfromSync);
}

template <typename T>
__aicore__  void DataCopyPad(const LocalTensor<T>& dstLocal,
                                    const GlobalTensor<T>& srcGlobal,
                                    const DataCopyExtParams& dataCopyParams,
                                    const DataCopyPadExtParams<T>& padParams,
                                    bool isGenfromSync)
{
    RankId srcRank = 0;
    DataSlice srcSlice;
    BlockId block = 0;
    RankId dstRank = 0;
    DataSlice dstSlice;
    u64 size = dataCopyParams.blockCount * dataCopyParams.blockLen;

    CHK_RET_NULL(MemLayout::Global()->GetSlice((char*)srcGlobal.ptr_, size, srcSlice, &srcRank));
    CHK_RET_NULL(MemLayout::Global()->AivGetSlice((char*)dstLocal.ptr_, size, dstSlice, &dstRank, &block));

    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    CHK_RET_NULL(hccl::CheckCurRankId(curRank, srcRank, dstRank));

    // 忽略空拷贝操作
    if (dataCopyParams.blockCount == 0 || dataCopyParams.blockLen == 0) {
        return;
    }

    DataCopyTask(curRank, srcRank, dstRank, srcSlice, dstSlice, block, PIPE_MTE2, true, isGenfromSync);
}

__aicore__ void SetAtomicNone()
{
    if(g_atomicType == CheckerReduceOp::REDUCE_RESERVED){
        HCCL_ERROR("SetAtomicNone Error, AtomicNone has been set.");
        return;
    }
    g_atomicType = CheckerReduceOp::REDUCE_RESERVED;
    return;
}

// set_atomic_add
template <typename T>
__aicore__ void SetAtomicAdd()
{
    if(g_atomicType != CheckerReduceOp::REDUCE_RESERVED){
        HCCL_ERROR("SetAtomicAdd Error, flag has been set:%d", g_atomicType);
        return;
    }
    g_atomicType = CheckerReduceOp::REDUCE_SUM;
    return;
}

// set_atomic_max
template <typename T>
__aicore__ void SetAtomicMax()
{
    if(g_atomicType != CheckerReduceOp::REDUCE_RESERVED){
        HCCL_ERROR("SetAtomicMax Error, flag has been set:%d", g_atomicType);
        return;
    }
    g_atomicType = CheckerReduceOp::REDUCE_MAX;
    return;

}

// set_atomic_min
template <typename T>
__aicore__ void SetAtomicMin()
{
    if(g_atomicType != CheckerReduceOp::REDUCE_RESERVED){
        HCCL_ERROR("SetAtomicMin Error, flag has been set:%d", g_atomicType);
        return;
    }
    g_atomicType = CheckerReduceOp::REDUCE_MIN;
    return;
}

template <typename T>
__aicore__ void Duplicate(const LocalTensor<T>& dstLocal, const T& scalar, const int32_t& count)
{

}

template __aicore__  void DataCopyPad<signed char>(const GlobalTensor<signed char>& dstGlobal,
                                                          const LocalTensor<signed char>& srcLocal,
                                                          const DataCopyExtParams& dataCopyParams,
                                                          bool isGenfromSync);
template __aicore__  void DataCopyPad<float>(const GlobalTensor<float>& dstGlobal,
                                                    const LocalTensor<float>& srcLocal,
                                                    const DataCopyExtParams& dataCopyParams,
                                                    bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned char>(const GlobalTensor<unsigned char>& dstGlobal,
                                                            const LocalTensor<unsigned char>& srcLocal,
                                                            const DataCopyExtParams& dataCopyParams,
                                                            bool isGenfromSync);
template __aicore__  void DataCopyPad<int>(const GlobalTensor<int>& dstGlobal,
                                                  const LocalTensor<int>& srcLocal,
                                                  const DataCopyExtParams& dataCopyParams,
                                                  bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned int>(const GlobalTensor<unsigned int>& dstGlobal,
                                                           const LocalTensor<unsigned int>& srcLocal,
                                                           const DataCopyExtParams& dataCopyParams,
                                                           bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned long>(const GlobalTensor<unsigned long>& dstGlobal,
                                                            const LocalTensor<unsigned long>& srcLocal,
                                                            const DataCopyExtParams& dataCopyParams,
                                                            bool isGenfromSync);
template __aicore__  void DataCopyPad<short>(const GlobalTensor<short>& dstGlobal,
                                                    const LocalTensor<short>& srcLocal,
                                                    const DataCopyExtParams& dataCopyParams,
                                                    bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned short>(const GlobalTensor<unsigned short>& dstGlobal,
                                                             const LocalTensor<unsigned short>& srcLocal,
                                                             const DataCopyExtParams& dataCopyParams,
                                                             bool isGenfromSync);

template __aicore__  void DataCopyPad<signed char>(const LocalTensor<signed char>& dstLocal,
                                                          const GlobalTensor<signed char>& srcGlobal,
                                                          const DataCopyExtParams& dataCopyParams,
                                                          const DataCopyPadExtParams<signed char>& padParams,
                                                          bool isGenfromSync);
template __aicore__  void DataCopyPad<float>(const LocalTensor<float>& dstLocal,
                                                    const GlobalTensor<float>& srcGlobal,
                                                    const DataCopyExtParams& dataCopyParams,
                                                    const DataCopyPadExtParams<float>& padParams,
                                                    bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned char>(const LocalTensor<unsigned char>& dstLocal,
                                                            const GlobalTensor<unsigned char>& srcGlobal,
                                                            const DataCopyExtParams& dataCopyParams,
                                                            const DataCopyPadExtParams<unsigned char>& padParams,
                                                            bool isGenfromSync);
template __aicore__  void DataCopyPad<int>(const LocalTensor<int>& dstLocal,
                                                  const GlobalTensor<int>& srcGlobal,
                                                  const DataCopyExtParams& dataCopyParams,
                                                  const DataCopyPadExtParams<int>& padParams,
                                                  bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned int>(const LocalTensor<unsigned int>& dstLocal,
                                                           const GlobalTensor<unsigned int>& srcGlobal,
                                                           const DataCopyExtParams& dataCopyParams,
                                                           const DataCopyPadExtParams<unsigned int>& padParams,
                                                           bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned long>(const LocalTensor<unsigned long>& dstLocal,
                                                            const GlobalTensor<unsigned long>& srcGlobal,
                                                            const DataCopyExtParams& dataCopyParams,
                                                            const DataCopyPadExtParams<unsigned long>& padParams,
                                                            bool isGenfromSync);
template __aicore__  void DataCopyPad<short>(const LocalTensor<short>& dstLocal,
                                                    const GlobalTensor<short>& srcGlobal,
                                                    const DataCopyExtParams& dataCopyParams,
                                                    const DataCopyPadExtParams<short>& padParams,
                                                    bool isGenfromSync);
template __aicore__  void DataCopyPad<unsigned short>(const LocalTensor<unsigned short>& dstLocal,
                                                             const GlobalTensor<unsigned short>& srcGlobal,
                                                             const DataCopyExtParams& dataCopyParams,
                                                             const DataCopyPadExtParams<unsigned short>& padParams,
                                                             bool isGenfromSync);

template __aicore__ void DataCopy<signed char>(const GlobalTensor<signed char>& dstGlobal,
                                                      const LocalTensor<signed char>& srcLocal,
                                                      const uint32_t calCount,
                                                      bool isGenfromSync);
template __aicore__ void DataCopy<float>(const GlobalTensor<float>& dstGlobal,
                                                const LocalTensor<float>& srcLocal,
                                                const uint32_t calCount,
                                                bool isGenfromSync);
template __aicore__ void DataCopy<unsigned char>(const GlobalTensor<unsigned char>& dstGlobal,
                                                        const LocalTensor<unsigned char>& srcLocal,
                                                        const uint32_t calCount,
                                                        bool isGenfromSync);
template __aicore__ void DataCopy<int>(const GlobalTensor<int>& dstGlobal,
                                              const LocalTensor<int>& srcLocal,
                                              const uint32_t calCount,
                                              bool isGenfromSync);
template __aicore__ void DataCopy<unsigned int>(const GlobalTensor<unsigned int>& dstGlobal,
                                                       const LocalTensor<unsigned int>& srcLocal,
                                                       const uint32_t calCount,
                                                       bool isGenfromSync);
template __aicore__ void DataCopy<unsigned long>(const GlobalTensor<unsigned long>& dstGlobal,
                                                        const LocalTensor<unsigned long>& srcLocal,
                                                        const uint32_t calCount,
                                                        bool isGenfromSync);
template __aicore__ void DataCopy<short>(const GlobalTensor<short>& dstGlobal,
                                                const LocalTensor<short>& srcLocal,
                                                const uint32_t calCount,
                                                bool isGenfromSync);
template __aicore__ void DataCopy<unsigned short>(const GlobalTensor<unsigned short>& dstGlobal,
                                                         const LocalTensor<unsigned short>& srcLocal,
                                                         const uint32_t calCount,
                                                         bool isGenfromSync);

template __aicore__ void DataCopy<signed char>(const LocalTensor<signed char>& dstLocal,
                                                      const GlobalTensor<signed char>& srcGlobal,
                                                      const uint32_t calCount,
                                                      bool isGenfromSync);
template __aicore__ void DataCopy<float>(const LocalTensor<float>& dstLocal,
                                                const GlobalTensor<float>& srcGlobal,
                                                const uint32_t calCount,
                                                bool isGenfromSync);
template __aicore__ void DataCopy<unsigned char>(const LocalTensor<unsigned char>& dstLocal,
                                                        const GlobalTensor<unsigned char>& srcGlobal,
                                                        const uint32_t calCount,
                                                        bool isGenfromSync);
template __aicore__ void DataCopy<int>(const LocalTensor<int>& dstLocal,
                                              const GlobalTensor<int>& srcGlobal,
                                              const uint32_t calCount,
                                              bool isGenfromSync);
template __aicore__ void DataCopy<unsigned int>(const LocalTensor<unsigned int>& dstLocal,
                                                       const GlobalTensor<unsigned int>& srcGlobal,
                                                       const uint32_t calCount,
                                                       bool isGenfromSync);
template __aicore__ void DataCopy<unsigned long>(const LocalTensor<unsigned long>& dstLocal,
                                                        const GlobalTensor<unsigned long>& srcGlobal,
                                                        const uint32_t calCount,
                                                        bool isGenfromSync);
template __aicore__ void DataCopy<short>(const LocalTensor<short>& dstLocal,
                                                const GlobalTensor<short>& srcGlobal,
                                                const uint32_t calCount,
                                                bool isGenfromSync);
template __aicore__ void DataCopy<unsigned short>(const LocalTensor<unsigned short>& dstLocal,
                                                         const GlobalTensor<unsigned short>& srcGlobal,
                                                         const uint32_t calCount,
                                                         bool isGenfromSync);

template void SetAtomicAdd<signed char>();
template void SetAtomicAdd<float>();
template void SetAtomicAdd<unsigned char>();
template void SetAtomicAdd<int>();
template void SetAtomicAdd<unsigned int>();
template void SetAtomicAdd<short>();
template void SetAtomicAdd<unsigned short>();

template void SetAtomicMax<signed char>();
template void SetAtomicMax<float>();
template void SetAtomicMax<unsigned char>();
template void SetAtomicMax<int>();
template void SetAtomicMax<unsigned int>();
template void SetAtomicMax<short>();
template void SetAtomicMax<unsigned short>();

template void SetAtomicMin<signed char>();
template void SetAtomicMin<float>();
template void SetAtomicMin<unsigned char>();
template void SetAtomicMin<int>();
template void SetAtomicMin<unsigned int>();
template void SetAtomicMin<short>();
template void SetAtomicMin<unsigned short>();

template void Duplicate<int>(const LocalTensor<int>& dstLocal, const int& scalar, const int32_t& count);

}