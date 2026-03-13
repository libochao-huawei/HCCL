#ifndef AIV_SYNC_INTERFACE_STUB_H
#define AIV_SYNC_INTERFACE_STUB_H

namespace AscendC {

constexpr uint64_t UB_FLAG_PAD_COUNT = 8;
constexpr uint64_t UB_ADDRESS_PAD_COUNT = 4;

template<HardEvent event> __aicore__ void SyncFunc();

__aicore__ void SetSignalValue(__gm__ int32_t* gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value);

__aicore__ void AddSignalValue(__gm__ int32_t* gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value);

__aicore__ void WaitSignalValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t expectedValue);

__aicore__ void WaitSignalGEValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t value);

__aicore__ void SetFlagBatchValue(__gm__ int32_t *ctrlFlagGM, TQue<QuePosition::VECOUT, 1> &batchQue, int32_t setValue, int32_t count);

__aicore__ int32_t GetSignalValue(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor);

__aicore__ int32_t GetSignalValueWithExpected(__gm__ int32_t *gmSignalAddr, LocalTensor<int32_t>& localTensor, int32_t expectedValue);

}

#endif