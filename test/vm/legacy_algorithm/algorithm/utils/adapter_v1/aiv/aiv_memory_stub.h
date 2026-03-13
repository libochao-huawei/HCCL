#ifndef AIV_MEMORY_STUB_H
#define AIV_MEMORY_STUB_H
#include <map>
#include <memory>
#include "aiv_base_stub.h"

namespace AscendC {

using TBufHandle = uint8_t*;

enum class TBufState : uint8_t {
    FREE = 0,
    OCCUPIED,
    ENQUE,
    DEQUE,
};

enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    C2PIPE2LOCAL,
    MAX,
};

using QuePosition = TPosition;

constexpr uint32_t MAX_QUE_NUM = 64;  // 一个kernel中所有的buffer num之和不能超过64

struct TensorMem{
    void *ptr; 
    uint64_t size; 
    uint32_t isUse;
    uint32_t inQue;
    bool needWait;
};

struct TBuffAddr {
    uint32_t dataLen;
    uint32_t bufferAddr;
    TBufHandle bufferHandle;
    uint8_t logicPos;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    uint8_t* absAddr;
#endif
};

// Tensor打桩
template <typename T> class LocalTensor {
public:
    __inout_pipe__(S) T GetValue(const uint32_t offset) const;
    template <typename T1> __inout_pipe__(S) void SetValue(const uint32_t index, const T1 value) const;
    LocalTensor operator[](const uint32_t offset) const;

    void *ptr_ = nullptr;
    uint64_t size_ = 0;
};

template <typename T> class GlobalTensor {
public:
    __aicore__ void SetGlobalBuffer(__gm__ T* buffer, uint64_t bufferSize);
    __aicore__ GlobalTensor operator[](const uint64_t offset) const;
        
    void *ptr_ = nullptr;
    uint64_t size_ = 0;
};

__aicore__ constexpr TPosition GetBufferLogicPos(TPosition pos, bool isSrc)
{
    if (pos == TPosition::VECIN) {
        return isSrc ? TPosition::GM : TPosition::VECIN;
    } else if (pos == TPosition::VECOUT) {
        return isSrc ? TPosition::VECOUT : TPosition::GM;
    }
    return TPosition::MAX;
};

template <TPosition src, TPosition dst, int32_t depth, auto mask = 0> class TQueBind {
public:
    __aicore__ TQueBind() 
        : ptr_(0), 
          size_(0), 
          num(0), 
          srcPostion(src), 
          dstPostion(dst) {
    }
    template <typename T> __aicore__ LocalTensor<T> AllocTensor();
    template <typename T> __aicore__ void FreeTensor(LocalTensor<T>& tensor);
    template <typename T> __aicore__ bool EnQue(const LocalTensor<T>& tensor);
    template <typename T> __aicore__ LocalTensor<T> DeQue();

    void *ptr_; 
    uint64_t size_; 
    uint32_t num;
    std::map<uint32_t, TensorMem> tensor;
    TPosition srcPostion;
    TPosition dstPostion;
};

template <TPosition pos, int32_t depth, auto mask = 0>
class TQue : public TQueBind<GetBufferLogicPos(pos, true), GetBufferLogicPos(pos, false), depth, mask> {
public:
    __aicore__ TQue() = default;
private:
    friend class TPipe;
    template<TPosition bufPos, uint32_t bufIDSize> friend class TBufPool;
    static constexpr bool isTQue = true;
};

template <TPosition pos = TPosition::LCM> class TBuf : public TQueBind<pos, pos, 0, 0> {
public:
    __aicore__ TBuf() = default;
    template <typename T> __aicore__ LocalTensor<T> Get();
    template <typename T> __aicore__ LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset);
};

class TPipeBase {
public:
};

using TEventID = int8_t;

class TPipe : public TPipeBase {
public:
    __aicore__ TPipe();
    __aicore__ ~TPipe();
    __aicore__ void Init();
    template <class T> __aicore__ bool InitBuffer(T& que, uint8_t num, uint32_t len);
    template <TPosition pos> __aicore__ bool InitBuffer(TBuf<pos>& buf, uint32_t len);
    __aicore__ TEventID FetchEventID(HardEvent evt);

protected:
    template <TPosition src, TPosition dst, int32_t depth, auto mask> friend class TQueBind;
    template <TPosition pos, int32_t depth, auto mask> friend class TQue;
    template <TPosition pos> friend class TBuf;

    void *startPtr = 0;
    void *endPtr = 0;
    void *curPtr = 0;
    uint64_t usedLen = 0;
    uint32_t usedNum = 0;
};

extern __inline__ std::unique_ptr<TPipe> g_tPipePtr;
__aicore__ TPipe* GetTPipePtr();

template <typename T>
std::unique_ptr<T> GetBufValue(uint64_t addr);
template <typename T>
void SetBufValue(uint64_t addr, T value);

void GetTPipe(TPosition pos, pipe_t &src, pipe_t &dst);
}  // namespace AscendC

#endif