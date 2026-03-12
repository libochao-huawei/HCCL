/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: AllGatherV语义校验实现类
 */
#include "allgather_v_semantics_checker.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include <map>
#include "llt_common.h"
#include "log.h"

using namespace Hccl;

namespace checker {
HcclResult TaskCheckAllGatherVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, CheckerOpParam &opParam)
{
    u32 rankSize = allRankMemSemantics.size();

    u64 outputSize = 0;
    // AllGatherV 输入不等长
    for (u32 i = 0; i < rankSize; i++) {
        u64 curCounts = opParam.VDataDes.counts[i];
        u64 curLength = curCounts * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
        outputSize += curLength;
    }

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        u64    totalSize   = 0;
        RankId curRankId   = 0;
        u64    curDataSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            u64 inputSize = opParam.VDataDes.counts[curRankId] * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
            while (!inputSize) {
                curRankId++;
                inputSize = opParam.VDataDes.counts[curRankId] * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
            }

            if (ele.startAddr != totalSize) {
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Missing buffer semantic: "
                "exepected startAddr is %llu, while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                    rankId, totalSize, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                    "while cur buffer semantic is %s", rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != curRankId) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic should come from rank %u, while it come from rank %u, "
                    "cur buffer semantic is %s", rankId, curRankId, ele.srcBufs.begin()->rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != curDataSize) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, cur buffer semantic is %s",
                    rankId, curDataSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;

            if (curDataSize == inputSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > inputSize) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Accumulated semantic size from rank %u is %llu, greater than expected %llu",
                    rankId, curRankId, curDataSize, inputSize);
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != outputSize) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "while outputSize is %llu, rankSize is %u",
                rankId, totalSize, outputSize, rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker