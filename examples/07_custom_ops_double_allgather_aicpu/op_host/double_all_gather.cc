#include "hccl_custom_double_allgather.h"
#include "common.h"
#include "resource.h"
#include "launch_kernel.h"

#include <cstdint>`r`n#include <cstdio>

namespace ops_hccl_double_allgather {

static bool IsOverlap(const void *lhs, size_t lhsBytes, const void *rhs, size_t rhsBytes)
{
    uintptr_t lhsBegin = reinterpret_cast<uintptr_t>(lhs);
    uintptr_t lhsEnd = lhsBegin + lhsBytes;
    uintptr_t rhsBegin = reinterpret_cast<uintptr_t>(rhs);
    uintptr_t rhsEnd = rhsBegin + rhsBytes;
    return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

static HcclResult ValidateArgs(const void *sendBuf0, void *recvBuf0, uint64_t sendCount0, HcclDataType dataType0,
    const void *sendBuf1, void *recvBuf1, uint64_t sendCount1, HcclDataType dataType1, HcclComm comm,
    aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf0);
    CHK_PTR_NULL(recvBuf0);
    CHK_PTR_NULL(sendBuf1);
    CHK_PTR_NULL(recvBuf1);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(sendCount0 == 0 || sendCount1 == 0, HCCL_ERROR("count must be positive"), HCCL_E_PARA);
    CHK_PRT_RET(!IsSupportedDataType(dataType0) || !IsSupportedDataType(dataType1),
        HCCL_ERROR("unsupported dtype"), HCCL_E_PARA);

    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    CHK_PRT_RET(rankSize == 0, HCCL_ERROR("invalid rankSize"), HCCL_E_PARA);

    size_t sendBytes0 = 0;
    size_t sendBytes1 = 0;
    CHK_PRT_RET(!GetBytesByCount(sendCount0, dataType0, sendBytes0) ||
        !GetBytesByCount(sendCount1, dataType1, sendBytes1), HCCL_ERROR("count overflow"), HCCL_E_PARA);
    CHK_PRT_RET(sendBytes0 > SIZE_MAX / rankSize || sendBytes1 > SIZE_MAX / rankSize,
        HCCL_ERROR("recv size overflow"), HCCL_E_PARA);

    const size_t recvBytes0 = sendBytes0 * rankSize;
    const size_t recvBytes1 = sendBytes1 * rankSize;

    CHK_PRT_RET(IsOverlap(sendBuf0, sendBytes0, recvBuf0, recvBytes0), HCCL_ERROR("gather0 inplace unsupported"), HCCL_E_PARA);
    CHK_PRT_RET(IsOverlap(sendBuf1, sendBytes1, recvBuf1, recvBytes1), HCCL_ERROR("gather1 inplace unsupported"), HCCL_E_PARA);
    CHK_PRT_RET(IsOverlap(sendBuf0, sendBytes0, sendBuf1, sendBytes1), HCCL_ERROR("send buffers overlap"), HCCL_E_PARA);
    CHK_PRT_RET(IsOverlap(sendBuf0, sendBytes0, recvBuf1, recvBytes1), HCCL_ERROR("send0 overlaps recv1"), HCCL_E_PARA);
    CHK_PRT_RET(IsOverlap(recvBuf0, recvBytes0, sendBuf1, sendBytes1), HCCL_ERROR("recv0 overlaps send1"), HCCL_E_PARA);
    CHK_PRT_RET(IsOverlap(recvBuf0, recvBytes0, recvBuf1, recvBytes1), HCCL_ERROR("recv buffers overlap"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

}

extern "C" HcclResult HcclDoubleAllGatherCustom(
    const void *sendBuf0,
    void *recvBuf0,
    uint64_t sendCount0,
    HcclDataType dataType0,
    const void *sendBuf1,
    void *recvBuf1,
    uint64_t sendCount1,
    HcclDataType dataType1,
    HcclComm comm,
    aclrtStream stream)
{
    using namespace ops_hccl_double_allgather;

    CHK_RET(ValidateArgs(sendBuf0, recvBuf0, sendCount0, dataType0,
        sendBuf1, recvBuf1, sendCount1, dataType1, comm, stream));

    DoubleAllGatherParam param = {};
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_double_allgather");
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);
    CHK_RET(HcclGetCommName(comm, param.commName));
    CHK_RET(HcclGetRankId(comm, &param.rank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    param.gather0.inputPtr = const_cast<void *>(sendBuf0);
    param.gather0.outputPtr = recvBuf0;
    param.gather0.count = sendCount0;
    param.gather0.dataType = dataType0;
    param.gather0.elemBytes = GetElemBytes(dataType0);

    param.gather1.inputPtr = const_cast<void *>(sendBuf1);
    param.gather1.outputPtr = recvBuf1;
    param.gather1.count = sendCount1;
    param.gather1.dataType = dataType1;
    param.gather1.elemBytes = GetElemBytes(dataType1);

    CHK_RET(PrepareResources(comm, param, stream));
    return LaunchKernel(param, stream);
}

