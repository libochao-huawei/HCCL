#include "task_exception_info.h"

#include <sstream>

namespace ops_hccl_allgatherbatch {

namespace {

constexpr char kAllGatherBatchAlgTag[] = "allgatherbatch_hd_stage";
constexpr char kAllGatherBatchAlgName[] = "allgatherbatch";
constexpr char kAllGatherBatchOpName[] = "AllGatherBatch";

}  // namespace

HcclResult CreateAllGatherBatchOpInfo(const OpParam &param, AllGatherBatchOpInfo &opInfo)
{
    errno_t ret = strncpy_s(opInfo.algTag, sizeof(opInfo.algTag), kAllGatherBatchAlgTag, sizeof(opInfo.algTag) - 1);
    CHK_PRT_RET(ret != EOK,
        HCCL_ERROR("[CreateAllGatherBatchOpInfo] copy algTag failed, ret=%d", static_cast<int>(ret)),
        HCCL_E_MEMORY);

    ret = strncpy_s(opInfo.algName, sizeof(opInfo.algName), kAllGatherBatchAlgName, sizeof(opInfo.algName) - 1);
    CHK_PRT_RET(ret != EOK,
        HCCL_ERROR("[CreateAllGatherBatchOpInfo] copy algName failed, ret=%d", static_cast<int>(ret)),
        HCCL_E_MEMORY);

    ret = strncpy_s(opInfo.tag, sizeof(opInfo.tag), param.tag, sizeof(opInfo.tag) - 1);
    CHK_PRT_RET(ret != EOK,
        HCCL_ERROR("[CreateAllGatherBatchOpInfo] copy tag failed, ret=%d", static_cast<int>(ret)),
        HCCL_E_MEMORY);

    ret = strncpy_s(opInfo.commName, sizeof(opInfo.commName), param.commName, sizeof(opInfo.commName) - 1);
    CHK_PRT_RET(ret != EOK,
        HCCL_ERROR("[CreateAllGatherBatchOpInfo] copy commName failed, ret=%d", static_cast<int>(ret)),
        HCCL_E_MEMORY);

    opInfo.rank = param.topoInfo.rank;
    opInfo.rankSize = param.topoInfo.rankSize;
    opInfo.itemCount = param.itemCount;
    opInfo.opType = param.opType;
    opInfo.commMode = param.commMode;
    opInfo.totalInputBytes = param.totalInputBytes;
    return HCCL_SUCCESS;
}

void GetAllGatherBatchOpInfo(const void *opInfo, char *output, size_t size)
{
    if (opInfo == nullptr || output == nullptr || size == 0) {
        return;
    }

    const AllGatherBatchOpInfo *info = reinterpret_cast<const AllGatherBatchOpInfo *>(opInfo);
    std::stringstream ss;
    ss << "op=" << kAllGatherBatchOpName << ", ";
    ss << "tag=" << info->tag << ", ";
    ss << "alg=" << info->algTag << ", ";
    ss << "comm=" << info->commName << ", ";
    ss << "rank=" << info->rank << "/" << info->rankSize << ", ";
    ss << "items=" << info->itemCount << ", ";
    ss << "bytes=" << info->totalInputBytes << ", ";
    ss << "mode=" << ToCommModeString(info->commMode) << ", ";
    ss << "opType=" << static_cast<uint32_t>(info->opType) << ".";

    const std::string text = ss.str();
    errno_t ret = strncpy_s(output, size, text.c_str(), size - 1);
    if (ret != EOK) {
        HCCL_ERROR("[GetAllGatherBatchOpInfo] copy output failed, ret=%d, size=%zu, textSize=%zu",
            static_cast<int>(ret),
            size,
            text.size());
    }
}

}  // namespace ops_hccl_allgatherbatch
