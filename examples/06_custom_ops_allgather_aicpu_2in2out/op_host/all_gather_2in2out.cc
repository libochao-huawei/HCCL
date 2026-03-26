#include <cstdio>
#include <cstring>
#include <vector>
#include "hccl_custom_allgather_2in2out.h"
#include "common.h"
#include "topology.h"
#include "small_count_policy.h"
#include "resource.h"
#include "launch_kernel.h"

using namespace ops_hccl_allgather_2in2out;

namespace {

HcclResult CheckParams(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream)
{
    // 这一步负责把最基础的非法输入挡在入口层，避免后续流程带着坏参数继续执行。
    CHK_PTR_NULL(sendBuf0);
    CHK_PTR_NULL(sendBuf1);
    CHK_PTR_NULL(recvBuf0);
    CHK_PTR_NULL(recvBuf1);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    if (sendCount0 == 0 || sendCount1 == 0) {
        HCCL_ERROR("sendCount0[%llu] sendCount1[%llu] is invalid",
            static_cast<unsigned long long>(sendCount0),
            static_cast<unsigned long long>(sendCount1));
        return HCCL_E_PARA;
    }
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("dataType[%u] is invalid", static_cast<uint32_t>(dataType));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildCommMeta(HcclComm comm, CommMeta &meta)
{
    CHK_RET(HcclGetRankId(comm, &meta.rankId));
    CHK_RET(HcclGetRankSize(comm, &meta.rankSize));
    CHK_RET(HcclGetCommName(comm, meta.commName));

    // 当前样例先保守固定为 OP_BASE；后续如果需要支持别的 workflow，再补真实查询。
    meta.workflowMode = kWorkflowModeOpBase;
    meta.deviceType = 0;
    meta.topologyType = DetectTopologyType(meta);
    return HCCL_SUCCESS;
}

HcclResult RunFallbackNative(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream)
{
    // fused 条件不满足时的保底路径：直接顺序调用两次原生 HcclAllGather。
    CHK_RET(HcclAllGather(sendBuf0, recvBuf0, sendCount0, dataType, comm, stream));
    CHK_RET(HcclAllGather(sendBuf1, recvBuf1, sendCount1, dataType, comm, stream));
    return HCCL_SUCCESS;
}

HcclResult RunFusedKernel(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    const SmallCountDecision &decision,
    const CommMeta &meta,
    const std::vector<uint32_t> &peers,
    HcclComm comm,
    aclrtStream stream)
{
    // 第 5 阶段开始真正走 fused kernel：Host 只负责准备参数和资源，
    // 两路 allgather 都在同一次 AICPU kernel 中顺序完成。
    OpParam param {};
    CHK_RET(BuildFusedOpParam(sendBuf0, sendBuf1,
                              recvBuf0, recvBuf1,
                              sendCount0, sendCount1,
                              dataType, decision, meta, peers, comm, param));

    HCCL_INFO("[Stage5] fused small-count path selected, launch one AICPU kernel for route0 + route1");
    return LaunchKernel(param, stream);
}

} // namespace

extern "C" HcclResult HcclAllGather2In2OutAicpuCustom(
    void *sendBuf0,
    void *sendBuf1,
    void *recvBuf0,
    void *recvBuf1,
    uint64_t sendCount0,
    uint64_t sendCount1,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream)
{
    // 这是整个自定义算子的 Host 主入口：
    // 1. 检查参数
    // 2. 获取通信域元数据
    // 3. 计算 small-count 资格
    // 4. 命中资格时准备 AICPU 资源并只 launch 一次 fused kernel
    // 5. 不满足资格时回退到两次原生 allgather
    CHK_RET(CheckParams(sendBuf0, sendBuf1, recvBuf0, recvBuf1,
                        sendCount0, sendCount1, dataType, comm, stream));

    CommMeta meta {};
    CHK_RET(BuildCommMeta(comm, meta));

    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    (void)cclBufferAddr;

    const SmallCountDecision decision = CheckFusedSmallCountEligible(
        meta, sendCount0, sendCount1, dataType, cclBufferSize);
    const std::vector<uint32_t> peers = BuildPeerOrder(
        meta.rankId, meta.rankSize, static_cast<TopologyType>(meta.topologyType));

    HCCL_INFO("[Stage5] rank[%u] rankSize[%u] topoType[%u] peerNum[%u] loopMaxCount[%llu] smallCountEligible[%d] reason[%s]",
        meta.rankId,
        meta.rankSize,
        meta.topologyType,
        static_cast<uint32_t>(peers.size()),
        static_cast<unsigned long long>(decision.loopMaxCount),
        static_cast<int>(decision.eligible),
        decision.reason);

    if (decision.eligible) {
        return RunFusedKernel(sendBuf0, sendBuf1,
                              recvBuf0, recvBuf1,
                              sendCount0, sendCount1,
                              dataType, decision, meta, peers, comm, stream);
    }

    HCCL_INFO("[Stage5] fallback path selected, reason[%s]", decision.reason);
    return RunFallbackNative(sendBuf0, sendBuf1,
                             recvBuf0, recvBuf1,
                             sendCount0, sendCount1,
                             dataType, comm, stream);
}
