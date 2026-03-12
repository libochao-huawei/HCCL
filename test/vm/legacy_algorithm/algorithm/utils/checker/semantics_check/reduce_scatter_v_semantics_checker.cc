/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ReduceScatterV语义校验实现类
 */
#include "reduce_scatter_v_semantics_checker.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include <map>
#include "base.h"
#include "log.h"

using namespace Hccl;

namespace checker {

HcclResult TaskCheckReduceScatterVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                            CheckerOpParam &opParam)
{
    CheckerReduceOp reduceType = opParam.reduceType;
    u32 rankSize = allRankMemSemantics.size();
    u64 outputSize = 0;

    for (u32 rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        u64 totalSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            if (ele.startAddr != totalSize) {
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Missing buffer semantic: exepected startAddr is %llu, "
                    "while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                    rankId, totalSize, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]cur buffer semantic reduceType %d is unequal to expected reduceType %d, "
                    "cur buffer semantic is %s",
                    rankId, ele.reduceType, reduceType, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]buffer semantic srcBufs size %u is unequal to rankSize %u, "
                    "cur buffer semantic is %s",
                    rankId, ele.srcBufs.size(), rankSize, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != static_cast<int>(rankSize - 1)) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]cur buffer semantic srcBufs is invalid, cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto &srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                    DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                    DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, "
                        "cur buffer semantic is %s",
                        rankId, ele.Describe().c_str());
                    HcclResult::HCCL_E_PARA;
                }

                const u64 outputOffset = opParam.VDataDes.displs[rankId] * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];
                if (srcBuf.srcAddr != outputOffset + totalSize) {
                    DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                    DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                    DUMP_AND_ERROR("[rankId:%u]Exepected semantic srcBuf srcAddr is %llu, "
                        "while cur srcBuf srcAddr is %llu, cur buffer semantic is %s",
                        rankId, outputOffset + totalSize, srcBuf.srcAddr, ele.Describe().c_str());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        outputSize = opParam.VDataDes.counts[rankId] * CHECK_SIZE_TABLE[opParam.VDataDes.dataType];

        if (totalSize != outputSize) {
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "while expected output size is %lu", rankId, totalSize, outputSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker