/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: All2All语义校验函数实现
 * Author: yinding
 * Create: 2024-09-30
 */
#include "all2all_semantics_checker.h"
 
#include <map>
#include <vector>
#include "checker_def.h"
#include "log.h"

namespace HcclSim {

void GenSendOffsetForOneRank(All2AllDataDesTagInner &all2AllDataDes, u32 rankSize, RankId targetRank, std::vector<u64> &sendOffsets)
{
    for (RankId srcRank = 0; srcRank < rankSize; srcRank++) {
        u64 offset = 0;
        for (RankId dstRank = 0; dstRank < targetRank; dstRank++) {
            u64 count = all2AllDataDes.sendCountMatrix[srcRank * rankSize + dstRank];
            u64 size = count * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            offset += size;
        }
        sendOffsets.push_back(offset);
    }
    return;
}

HcclResult TaskCheckAll2AllSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics,
                                     All2AllDataDesTagInner &all2AllDataDes)
{
    u32 rankSize = allRankMemSemantics.size();

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        std::vector<u64> sendOffsets;
        GenSendOffsetForOneRank(all2AllDataDes, rankSize, rankId, sendOffsets);

        u64    totalSize   = 0;
        RankId curRankId   = 0;
        u64    curDataSize = 0;

        // curRankId向rankId发送的数据量是0 直接跳过
        while (curRankId < rankSize && all2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId] == 0) {
            curRankId++;
        }

        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            if (ele.startAddr != totalSize) {
                HCCL_ERROR("[rankId:%u]Missing buffer semantic: "
                    "exepected startAddr is %llu, while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                    rankId, totalSize, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                    "while cur buffer semantic is %s", rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (curRankId >= rankSize) {
                HCCL_ERROR("[rankId:%u]Extra buffer semantic found", rankId);
                return HcclResult::HCCL_E_PARA;
            }
 
            if (ele.srcBufs.begin()->rankId != curRankId) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic should come from rank %u, while it come from rank %u, "
                    "cur buffer semantic is %s", rankId, curRankId, ele.srcBufs.begin()->rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != sendOffsets[curRankId] + curDataSize) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, cur buffer semantic is %s",
                    rankId, sendOffsets[curRankId] + curDataSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            u64 recvCountFromCurRank = all2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId];
            u64 recvDataSizeFromCurRank = recvCountFromCurRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            if (curDataSize == recvDataSizeFromCurRank) {
                curDataSize = 0;
                curRankId++;
                while (curRankId < rankSize && all2AllDataDes.sendCountMatrix[curRankId * rankSize + rankId] == 0) {
                    // 跳过后续所有发送量为0的rank
                    curRankId++;
                }
            } else if (curDataSize > recvDataSizeFromCurRank) {
                HCCL_ERROR("[rankId:%u]Accumulated semantic size from rank %u is %llu, greater than expected %llu",
                    rankId, curRankId, curDataSize, recvDataSizeFromCurRank);
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankId等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankId != rankSize) {
            HCCL_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "accumulated semantic size from rank %u is %llu, while rankSize is %u",
                rankId, totalSize, curRankId, curDataSize, rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}


} // namespace HcclSim