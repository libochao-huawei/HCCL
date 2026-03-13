/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: All2All语义校验函数实现
 * Author: yinding
 * Create: 2024-09-30
 */
#include "all2all_semantics_checker.h"
#include "data_dumper.h"
#include "analysis_result.pb.h"
#include "semantics_utils.h"

#include <map>
#include <vector>
#include "llt_common.h"
#include "log.h"

using namespace Hccl;

namespace checker {

void GenSendOffsetForOneRank(CheckerOpParam &opParam, u32 rankSize, RankId targetRank, std::vector<u64> &sendOffsets)
{
    for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
        u64 offset = 0;
        for (RankId dstRank = 0; dstRank < targetRank; dstRank++) {
            u64 count = opParam.All2AllDataDes.sendCountMatrix[srcRank * rankSize + dstRank];
            u64 size = count * CHECK_SIZE_TABLE[opParam.All2AllDataDes.recvType];
            offset += size;
        }
        sendOffsets.push_back(offset);
    }
    return;
}

HcclResult TaskCheckAll2AllSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                     CheckerOpParam &opParam)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            DataDumper::Global()->AddMissingSemantic(rankId, BufferType::OUTPUT, 0);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        std::vector<u64> sendOffsets;
        GenSendOffsetForOneRank(opParam, rankSize, rankId, sendOffsets);

        u64    totalSize   = 0;
        RankId curRankId   = 0;
        u64    curDataSize = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            if (ele.startAddr != totalSize) {
                DataDumper::Global()->AddMissingSemantic(rankId, BufferType::OUTPUT, totalSize);
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

            if (ele.srcBufs.begin()->srcAddr != sendOffsets[curRankId] + curDataSize) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, cur buffer semantic is %s",
                    rankId, sendOffsets[curRankId] + curDataSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            u64 recvCountFromCurRank = opParam.All2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId];
            u64 recvDataSizeFromCurRank = recvCountFromCurRank * CHECK_SIZE_TABLE[opParam.All2AllDataDes.recvType];
            if (curDataSize == recvDataSizeFromCurRank) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > recvDataSizeFromCurRank) {
                DataDumper::Global()->MarkInvalidSemantic(rankId, BufferType::OUTPUT, ele);
                DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_UNEXPECTED_SEMANTIC);
                DUMP_AND_ERROR("[rankId:%u]Accumulated semantic size from rank %u is %llu, greater than expected %llu",
                    rankId, curRankId, curDataSize, recvDataSizeFromCurRank);
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankId等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankId != rankSize) {
            DataDumper::Global()->AddMissingSemantic(rankId, BufferType::OUTPUT, totalSize);
            DataDumper::Global()->SetResultStatus(gui::ResultStatus::CHECK_FAILED_MISSING_SEMANTIC);
            DUMP_AND_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "accumulated semantic size from rank %u is %llu, while rankSize is %u",
                rankId, totalSize, curRankId, curDataSize, rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}


} // namespace checker