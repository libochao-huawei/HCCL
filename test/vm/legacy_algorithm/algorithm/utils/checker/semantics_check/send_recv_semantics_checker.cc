/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Reduce语义校验实现类
 * Author: huangweihao
 * Create: 2024-10-10
 */
#include "send_recv_semantics_checker.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include <map>
#include "base.h"
#include "log.h"

using namespace Hccl;

namespace checker {

HcclResult TaskCheckSendRecvSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                      RankId srcRank, RankId dstRank)
{
    u32 rankSize = allRankMemSemantics.size();

    // 对应的rank不存在需要报错
    if (allRankMemSemantics.size() != 2) {
        DUMP_AND_ERROR("sendrecv only support 2 ranks");
        return HcclResult::HCCL_E_PARA;
    }

    u64 totalSize = 0;
    for (auto &ele : allRankMemSemantics[dstRank][BufferType::OUTPUT]) {
        if (ele.startAddr != totalSize) {
            DataDumper::Global()->AddMissingSemantic(dstRank, BufferType::OUTPUT, totalSize);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Missing buffer semantic: "
                "exepected startAddr is %llu, while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                dstRank, totalSize, ele.startAddr, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() != 1) {
            DataDumper::Global()->MarkInvalidSemantic(dstRank, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                "while cur buffer semantic is %s", dstRank, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->rankId != srcRank) {
            DataDumper::Global()->MarkInvalidSemantic(dstRank, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Buffer semantic srcBuf rank[%u] is not from srcRank[%u], "
                "cur buffer semantic is %s",
                dstRank, ele.srcBufs.begin()->rankId, srcRank, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
            DataDumper::Global()->MarkInvalidSemantic(dstRank, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s",
                dstRank, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }
        if (ele.srcBufs.begin()->srcAddr != totalSize) {
            DataDumper::Global()->MarkInvalidSemantic(dstRank, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, "
                "while it is %llu, cur buffer semantic is %s",
                dstRank, totalSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }
        totalSize += ele.size;
    }
    if (totalSize != dataSize) {
        DataDumper::Global()->AddMissingSemantic(dstRank, BufferType::OUTPUT, totalSize);
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
        DUMP_AND_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
            "which should be %llu", dstRank, totalSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker