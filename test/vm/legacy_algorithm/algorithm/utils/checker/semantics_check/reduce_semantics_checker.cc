/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Reduce语义校验实现类
 * Author: huangweihao
 * Create: 2024-10-10
 */
#include "reduce_semantics_checker.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include <map>
#include "base.h"
#include "log.h"

using namespace Hccl;

namespace checker {

HcclResult TaskCheckReduceSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, u64 dataSize,
                                    CheckerReduceOp reduceType, RankId root)
{
    u32 rankSize = allRankMemSemantics.size();
    if (rankSize == 0) {
        return HCCL_SUCCESS;
    }

    // 对应的rank不存在需要报错
    if (allRankMemSemantics.count(root) == 0 && dataSize > 0) {
        DataDumper::Global()->AddMissingSemantic(root, BufferType::OUTPUT, 0);
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
        DUMP_AND_ERROR("Missing rank %d mem semantics", root);
        return HcclResult::HCCL_E_PARA;
    }

    u64 totalSize = 0;
    for (auto &ele : allRankMemSemantics[root][BufferType::OUTPUT]) {
        if (ele.startAddr != totalSize) {
            DataDumper::Global()->AddMissingSemantic(root, BufferType::OUTPUT, totalSize);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Missing buffer semantic: exepected startAddr is %llu, "
                "while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                root, totalSize, ele.startAddr, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
            DataDumper::Global()->MarkInvalidSemantic(root, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]cur buffer semantic reduceType %d is unequal to expected reduceType %d, "
                "cur buffer semantic is %s",
                root, ele.reduceType, reduceType, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.size() != rankSize) {
            DataDumper::Global()->MarkInvalidSemantic(root, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]buffer semantic srcBufs size %u is unequal to rankSize %u, "
                "cur buffer semantic is %s",
                root, ele.srcBufs.size(), rankSize, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        if (ele.srcBufs.begin()->rankId != 0 or ele.srcBufs.rbegin()->rankId != (rankSize - 1)) {
            DataDumper::Global()->MarkInvalidSemantic(root, BufferType::OUTPUT, ele);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]cur buffer semantic srcBufs is invalid, cur buffer semantic is %s",
                root, ele.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }

        for (auto &srcBuf : ele.srcBufs) {
            if (srcBuf.bufType != BufferType::INPUT) {
                DataDumper::Global()->MarkInvalidSemantic(root, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s",
                    root, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != totalSize) {
                DataDumper::Global()->MarkInvalidSemantic(root, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Exepected semantic srcBuf srcAddr is %llu, "
                    "while cur srcBuf srcAddr is %llu, cur buffer semantic is %s",
                    root, totalSize, srcBuf.srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
        }
        totalSize += ele.size;
    }
    if (totalSize != dataSize) {
        DataDumper::Global()->AddMissingSemantic(root, BufferType::OUTPUT, totalSize);
        DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
        DUMP_AND_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
            "while exepected total size is %llu", root, totalSize, dataSize);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker