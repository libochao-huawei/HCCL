#include <mutex>
#include "mem_layout.h"
#include "aiv_memory_stub.h"
#include "aiv_sync_stub.h"

namespace AscendC {

template <typename T>
__inout_pipe__(S) T LocalTensor<T>::GetValue(const uint32_t offset) const
{
    HCCL_DEBUG("getvalue:offset %d, ptr %p", offset, ptr_);
    std::unique_ptr<T> ret = GetBufValue<T>(offset * sizeof(T) + (uint64_t)ptr_);
    if (ret) {
        HCCL_DEBUG("get value *ret:%d", *ret);
        return *ret;
    }

    u64 realAddr = 0;
    u64 size = 0;
    if (checker::MemLayout::Global()->GetRealAddr((u64)(offset * sizeof(T) + (uint64_t)ptr_), realAddr, size) ==
        HCCL_SUCCESS) {
        return *(T *)realAddr;
    }

    return 0;
}

template <typename T>
template <typename T1>
__inout_pipe__(S) void LocalTensor<T>::SetValue(const uint32_t index, const T1 value) const
{
    HCCL_DEBUG("setvalue:offset %d, ptr %d", index, value);
    SetBufValue<T>(index * sizeof(T) + (uint64_t)ptr_, value);
    return;
}

template <typename T>
LocalTensor<T> LocalTensor<T>::operator[](const uint32_t offset) const
{
    LocalTensor<T> tmp;
    tmp.ptr_ = (void *)((offset * sizeof(T)) + (uint64_t)ptr_);
    tmp.size_ = this->size_ - offset * sizeof(T);
    return tmp;
}

template <typename T>
__aicore__ void GlobalTensor<T>::SetGlobalBuffer(__gm__ T* buffer, uint64_t bufferSize)
{
    this->ptr_ = buffer;
    this->size_ = bufferSize * sizeof(T);
    HCCL_DEBUG("buffer %p, bufferSize %ld", ptr_, size_);
    CHK_RET_NULL(checker::MemLayout::Global()->SetGlobalBuffer((char *)this->ptr_, this->size_));
}

template <typename T>
__aicore__ GlobalTensor<T> GlobalTensor<T>::operator[](const uint64_t offset) const
{
    GlobalTensor<T> tmp;
    tmp.ptr_ = (void *)((offset * sizeof(T)) + (uint64_t)ptr_);
    tmp.size_ = this->size_ - offset * sizeof(T);
    return tmp;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ inline LocalTensor<T> TQueBind<src, dst, depth, mask>::AllocTensor()
{
    LocalTensor<T> tmp;
    for (auto i = 0; i < num; i++) {
        if (this->tensor[i].isUse == 0 && this->tensor[i].inQue == 0) {
            this->tensor[i].isUse = 1;
            tmp.ptr_ = this->tensor[i].ptr;
            tmp.size_ = this->tensor[i].size;
            if (this->tensor[i].needWait) {
                pipe_t pSrc = PIPE_S;
                pipe_t pDes = PIPE_S;
                GetTPipe(dstPostion, pSrc, pDes);
                wait_flag(pDes, pSrc, (event_t)0, true);
            }
            tensor[i].needWait = true;
            break;
        }
    }
    return tmp;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ void TQueBind<src, dst, depth, mask>::FreeTensor(LocalTensor<T> &tensor)
{
    for (auto i = 0; i < num; i++) {
        if (this->tensor[i].ptr == tensor.ptr_ && this->tensor[i].isUse == 1) {
            this->tensor[i].isUse = 0;
            break;
        }
    }
    tensor.ptr_ = 0;
    tensor.size_ = 0;
    pipe_t pSrc = PIPE_S;
    pipe_t pDes = PIPE_S;
    GetTPipe(dstPostion, pSrc, pDes);
    set_flag(pDes, pSrc, (event_t)0, true);
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ bool TQueBind<src, dst, depth, mask>::EnQue(const LocalTensor<T> &tensor)
{
    for (auto i = 0; i < num; i++) {
        if (this->tensor[i].ptr == tensor.ptr_ && this->tensor[i].inQue == 0) {
            this->tensor[i].inQue = 1;
            pipe_t pSrc = PIPE_S;
            pipe_t pDes = PIPE_S;
            GetTPipe(dstPostion, pSrc, pDes);
            set_flag(pSrc, pDes, (event_t)0);
            return true;
        }
    }
    return false;
}

template <TPosition src, TPosition dst, int32_t depth, auto mask>
template <typename T>
__aicore__ LocalTensor<T> TQueBind<src, dst, depth, mask>::DeQue()
{
    LocalTensor<T> tmp;
    for (auto i = 0; i < num; i++) {
        if (this->tensor[i].isUse == 1 && this->tensor[i].inQue == 1) {
            this->tensor[i].inQue = 0;
            tmp.ptr_ = this->tensor[i].ptr;
            tmp.size_ = this->tensor[i].size;
            break;
        }
    }
    pipe_t pSrc = PIPE_S;
    pipe_t pDes = PIPE_S;
    GetTPipe(dstPostion, pSrc, pDes);
    wait_flag(pSrc, pDes, (event_t)0);
    return tmp;
}

template <TPosition pos>
template <typename T>
__aicore__ LocalTensor<T> TBuf<pos>::Get()
{
    LocalTensor<T> tmp;
    tmp.ptr_ = this->ptr_;
    tmp.size_ = this->size_;
    return tmp;
}

template <TPosition pos>
template <typename T>
__aicore__ LocalTensor<T> TBuf<pos>::GetWithOffset(uint32_t size, uint32_t bufOffset)
{
    LocalTensor<T> tmp;
    if(size*sizeof(T) + bufOffset > this->size_){
        return tmp;
    }
    tmp.ptr_ = (void *)(((uint64_t)this->ptr_) + bufOffset);
    tmp.size_ = size * sizeof(T);
    return tmp;
}

__aicore__ TPipe::TPipe()
{
    CHK_PRT(checker::MemLayout::Global()->TpipeInit(startPtr, endPtr, GetBlockIdx()));
    curPtr = startPtr;
    usedLen = 0;
    usedNum = 0;
}
__aicore__ TPipe::~TPipe()
{

}

template <class T>
__aicore__ bool TPipe::InitBuffer(T &que, uint8_t num, uint32_t len)
{
    if ((usedLen + num * len) > ((uint64_t)endPtr - (uint64_t)startPtr) || usedNum + num > MAX_QUE_NUM) {
        return false;
    }

    que.ptr_ = curPtr;
    que.size_ = len * num;
    que.num = num;

    uint64_t tensorLen = len;
    for (auto i = 0; i < que.num; i++) {
        TensorMem tmp = {(void *)((uint64_t)curPtr + i * tensorLen), tensorLen, 0, 0, false};
        que.tensor[i] = tmp;
    }

    uint64_t usedPtr = (uint64_t)curPtr + num * len;
    curPtr = (void *)usedPtr;
    usedLen += num * len;
    usedNum += num;
    return true;
}

template <TPosition pos>
__aicore__ bool TPipe::InitBuffer(TBuf<pos>& buf, uint32_t len)
{
    if ((usedLen + len) > ((uint64_t)endPtr - (uint64_t)startPtr) || usedNum + 1 > MAX_QUE_NUM) {
        return false;
    }

    buf.ptr_ = curPtr;
    buf.size_ = len;
    buf.num = 1;
    TensorMem tmp = {curPtr, len, 0, 0, false};
    buf.tensor[0] = tmp;

    uint64_t usedPtr = (uint64_t)curPtr + buf.size_;
    curPtr = (void *)usedPtr;
    usedLen += buf.size_;
    usedNum++;
    return true;
}

__aicore__ TEventID TPipe::FetchEventID(HardEvent evt)
{
    return (TEventID)0;
}

__inline__ std::unique_ptr<TPipe> g_tPipePtr;
__aicore__ TPipe* GetTPipePtr() {
    if (g_tPipePtr == nullptr) {
        static std::mutex mutex;  // 静态mutex，避免每次调用重新创建
        std::lock_guard<std::mutex> lock(mutex);
        g_tPipePtr = std::make_unique<TPipe>();
    }
    return g_tPipePtr.get();
}

template <typename T>
std::map<uint64_t, T> g_valueMap;

template <typename T>
std::unique_ptr<T> GetBufValue(uint64_t addr)
{
    auto it = g_valueMap<T>.find(addr);
    if (it != g_valueMap<T>.end()) {
        return std::make_unique<T>(it->second);
    } else {
        return nullptr;
    }
}

template <typename T>
void SetBufValue(uint64_t addr, T value)
{
    g_valueMap<T>[addr] = value;
}

void GetTPipe(TPosition pos, pipe_t &src, pipe_t &dst)
{
    switch (pos) {
        case TPosition::GM:
            src = pipe_t::PIPE_S;
            dst = pipe_t::PIPE_MTE3;
            return;
        case TPosition::VECIN:
            src = pipe_t::PIPE_MTE2;
            dst = pipe_t::PIPE_S;
            return;
        case TPosition::VECOUT:
            src = pipe_t::PIPE_MTE2;
            dst = pipe_t::PIPE_MTE3;
            return;
        default:
            HCCL_ERROR("Position not supported:%d.", pos);
            return;
    }
}

template __inout_pipe__(S) int LocalTensor<int>::GetValue(const uint32_t offset) const;
template __inout_pipe__(S) unsigned long LocalTensor<unsigned long>::GetValue(const uint32_t offset) const;

template __inout_pipe__(S) void LocalTensor<int>::SetValue<int>(const uint32_t index, const int value) const;
template __inout_pipe__(S) void LocalTensor<unsigned long>::SetValue<unsigned long>(const uint32_t index, const unsigned long value) const;

template LocalTensor<int> LocalTensor<int>::operator[](const uint32_t offset) const;
template LocalTensor<unsigned int> LocalTensor<unsigned int>::operator[](const uint32_t offset) const;
template LocalTensor<unsigned long> LocalTensor<unsigned long>::operator[](const uint32_t offset) const;

template __aicore__ void GlobalTensor<signed char>::SetGlobalBuffer(__gm__ signed char* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<float>::SetGlobalBuffer(__gm__ float* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<unsigned char>::SetGlobalBuffer(__gm__ unsigned char* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<int>::SetGlobalBuffer(__gm__ int* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<unsigned int>::SetGlobalBuffer(__gm__ unsigned int* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<unsigned long>::SetGlobalBuffer(__gm__ unsigned long* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<short>::SetGlobalBuffer(__gm__ short* buffer, uint64_t bufferSize);
template __aicore__ void GlobalTensor<unsigned short>::SetGlobalBuffer(__gm__ unsigned short* buffer, uint64_t bufferSize);

template __aicore__ GlobalTensor<signed char> GlobalTensor<signed char>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<float> GlobalTensor<float>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<unsigned char> GlobalTensor<unsigned char>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<int> GlobalTensor<int>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<unsigned int> GlobalTensor<unsigned int>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<unsigned long> GlobalTensor<unsigned long>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<short> GlobalTensor<short>::operator[](const uint64_t offset) const;
template __aicore__ GlobalTensor<unsigned short> GlobalTensor<unsigned short>::operator[](const uint64_t offset) const;

template __aicore__ LocalTensor<int> TQueBind<TPosition::GM, TPosition::VECIN, 1, 0>::AllocTensor<int>();
template __aicore__ LocalTensor<unsigned long> TQueBind<TPosition::GM, TPosition::VECIN, 1, 0>::AllocTensor<unsigned long>();
template __aicore__ LocalTensor<int> TQueBind<TPosition::VECOUT, TPosition::GM, 1, 0>::AllocTensor<int>();
template __aicore__ LocalTensor<unsigned long> TQueBind<TPosition::VECOUT, TPosition::GM, 1, 0>::AllocTensor<unsigned long>();
template __aicore__ LocalTensor<signed char> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<signed char>();
template __aicore__ LocalTensor<float> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<float>();
template __aicore__ LocalTensor<unsigned char> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<unsigned char>();
template __aicore__ LocalTensor<int> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<int>();
template __aicore__ LocalTensor<unsigned int> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<unsigned int>();
template __aicore__ LocalTensor<short> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<short>();
template __aicore__ LocalTensor<unsigned short> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::AllocTensor<unsigned short>();

template __aicore__ void TQueBind<TPosition::GM, TPosition::VECIN, 1, 0>::FreeTensor<int>(LocalTensor<int>& tensor);
template __aicore__ void TQueBind<TPosition::GM, TPosition::VECIN, 1, 0>::FreeTensor<unsigned long>(LocalTensor<unsigned long>& tensor);
template __aicore__ void TQueBind<TPosition::VECOUT, TPosition::GM, 1, 0>::FreeTensor<int>(LocalTensor<int>& tensor);
template __aicore__ void TQueBind<TPosition::VECOUT, TPosition::GM, 1, 0>::FreeTensor<unsigned long>(LocalTensor<unsigned long>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<signed char>(LocalTensor<signed char>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<float>(LocalTensor<float>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<unsigned char>(LocalTensor<unsigned char>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<int>(LocalTensor<int>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<unsigned int>(LocalTensor<unsigned int>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<short>(LocalTensor<short>& tensor);
template __aicore__ void TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::FreeTensor<unsigned short>(LocalTensor<unsigned short>& tensor);

template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<signed char>(const LocalTensor<signed char>& tensor);
template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<float>(const LocalTensor<float>& tensor);
template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<unsigned char>(const LocalTensor<unsigned char>& tensor);
template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<int>(const LocalTensor<int>& tensor);
template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<unsigned int>(const LocalTensor<unsigned int>& tensor);
template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<short>(const LocalTensor<short>& tensor);
template __aicore__ bool TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::EnQue<unsigned short>(const LocalTensor<unsigned short>& tensor);

template __aicore__ LocalTensor<signed char> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<signed char>();
template __aicore__ LocalTensor<float> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<float>();
template __aicore__ LocalTensor<unsigned char> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<unsigned char>();
template __aicore__ LocalTensor<int> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<int>();
template __aicore__ LocalTensor<unsigned int> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<unsigned int>();
template __aicore__ LocalTensor<short> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<short>();
template __aicore__ LocalTensor<unsigned short> TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>::DeQue<unsigned short>();

template __aicore__ LocalTensor<int> TBuf<TPosition::VECCALC>::Get<int>();
template __aicore__ LocalTensor<unsigned long> TBuf<TPosition::VECCALC>::Get<unsigned long>();
template __aicore__ LocalTensor<int> TBuf<TPosition::VECCALC>::GetWithOffset<int>(uint32_t size, uint32_t bufOffset);

template __aicore__ bool TPipe::InitBuffer<TQue<TPosition::VECIN, 1, 0>>(TQue<TPosition::VECIN, 1, 0>& que, uint8_t num, uint32_t len);
template __aicore__ bool TPipe::InitBuffer<TQue<TPosition::VECOUT, 1, 0>>(TQue<TPosition::VECOUT, 1, 0>& que, uint8_t num, uint32_t len);
template __aicore__ bool TPipe::InitBuffer<TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>>(TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, 0>& que,
                                                                                                       uint8_t num, uint32_t len);
template __aicore__ bool TPipe::InitBuffer<TPosition::VECCALC>(TBuf<TPosition::VECCALC>& buf, uint32_t len);

}