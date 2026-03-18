/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_batch_send_recv_executor.h"
#include "ins_temp_batch_send_recv_mesh_1D.h"
#ifndef AICPU_COMPILE
#include "ccu_temp_batch_send_recv_mesh_1D.h"
#endif

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2BatchSendRecvSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    std::vector<std::vector<u32>> tempAlgHierarchyInfo = algHierarchyInfo.infos[0];
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierarchyInfo);
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortSendItems(
    const HcclSendRecvItem *a, const HcclSendRecvItem *b) const
{
    u32 aFlag = (a->remoteRank <= static_cast<uint32_t>(myRank_)) ?
        (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank <= static_cast<uint32_t>(myRank_)) ?
        (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag != bFlag) {
        return aFlag > bFlag;
    } else if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortRecvItems(
    const HcclSendRecvItem *a, const HcclSendRecvItem *b) const
{
    u32 aFlag = (a->remoteRank < static_cast<uint32_t>(myRank_)) ?
        (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank < static_cast<uint32_t>(myRank_)) ?
        (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag != bFlag) {
        return aFlag < bFlag;
    } else if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortSelfItems(
    const HcclSendRecvItem *a, const HcclSendRecvItem *b) const
{
    if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetPairWiseList(
    const HcclSendRecvItem *sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] index is %u, itemNum is %u, "\
            "localRankID is %d, remoteRank is %u, sendRecvType is %u, rankSize is %u.",
            i, itemNum, myRank_, sendRecvInfo->remoteRank,
            static_cast<u32>(sendRecvInfo->sendRecvType), rankSize_);
        CHK_PTR_NULL(sendRecvInfo->buf);

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] sendRecvType wrong, "\
                "sendrecvType is %d, rankID is %d, remoteRank is %u.",
                sendRecvInfo->sendRecvType, myRank_, sendRecvInfo->remoteRank);
            return HcclResult::HCCL_E_PARA;
        }
        sendRecvInfo++;
    }

    // pair-wise 排序
    auto sendCompare = [this](const HcclSendRecvItem *a, const HcclSendRecvItem *b) {
        return this->SortSendItems(a, b);
    };
    auto recvCompare = [this](const HcclSendRecvItem *a, const HcclSendRecvItem *b) {
        return this->SortRecvItems(a, b);
    };
    std::stable_sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::stable_sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    // 筛选自收发任务
    while ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_)) &&
        (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        sendToSelfDeque_.push_back(sendDeque_.front());
        recvFromSelfDeque_.push_back(recvDeque_.front());
        sendDeque_.pop_front();
        recvDeque_.pop_front();
    }
    auto selfDequeCompare = [this](const HcclSendRecvItem *a, const HcclSendRecvItem *b) {
        return this->SortSelfItems(a, b);
    };
    std::stable_sort(sendToSelfDeque_.begin(), sendToSelfDeque_.end(), selfDequeCompare);
    std::stable_sort(recvFromSelfDeque_.begin(), recvFromSelfDeque_.end(), selfDequeCompare);

    // 自发自收任务未完全匹配
    if ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_)) ||
        (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] SendTask and RecvTask to rank itself "\
            "do not match, please check the task list.");
        return HCCL_E_PARA;
    }

    // 校验自收发配对并转换为 SendRecvSlice
    CHK_RET(CalcSelfSlices());

    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcSendSlices()
{
    while (!sendDeque_.empty()) {
        const HcclSendRecvItem *sendItem = sendDeque_.front();
        HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][CalcSendSlices] remoteRank[%u], buf[%p], count[%llu], "\
            "dataType[%u], sendRecvType[%d].", sendItem->remoteRank, sendItem->buf,
            sendItem->count, sendItem->dataType, sendItem->sendRecvType);
        dataTypeSize_ = DATATYPE_SIZE_TABLE[sendItem->dataType];
        CHK_PRT_RET(dataTypeSize_ == 0,
            HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][CalcSendSlices] dataTypeSize is zero."),
            HCCL_E_PARA);
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        u8 *curInputPtr = static_cast<u8 *>(sendItem->buf);
        CHK_PTR_NULL(curInputPtr);

        u64 curOffset = 0;
        u64 resDataCount = sendItem->count;
        while (resDataCount > 0) {
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curInputPtr = static_cast<u8 *>(sendItem->buf) + curOffset;
            sendDataSlices_.emplace_back(static_cast<void *>(curInputPtr), transferSize, sendItem->remoteRank);
            HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][CalcSendSlices] slice curOffset[%llu], "\
                "slice size[%llu], curInputPtr[%p].", curOffset, transferSize, curInputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        sendDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRecvSlices()
{
    while (!recvDeque_.empty()) {
        const HcclSendRecvItem *recvItem = recvDeque_.front();
        HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][CalcRecvSlices] remoteRank[%u], buf[%p], count[%llu], "\
            "dataType[%u], sendRecvType[%d].", recvItem->remoteRank, recvItem->buf,
            recvItem->count, recvItem->dataType, recvItem->sendRecvType);
        dataTypeSize_ = DATATYPE_SIZE_TABLE[recvItem->dataType];
        CHK_PRT_RET(dataTypeSize_ == 0,
            HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][CalcRecvSlices] dataTypeSize is zero."),
            HCCL_E_PARA);
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        u8 *curOutputPtr = static_cast<u8 *>(recvItem->buf);
        CHK_PTR_NULL(curOutputPtr);

        u64 curOffset = 0;
        u64 resDataCount = recvItem->count;
        while (resDataCount > 0) {
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curOutputPtr = static_cast<u8 *>(recvItem->buf) + curOffset;
            recvDataSlices_.emplace_back(static_cast<void *>(curOutputPtr), transferSize, recvItem->remoteRank);
            HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][CalcRecvSlices] slice curOffset[%llu], "\
                "slice size[%llu], curOutputPtr[%p].", curOffset, transferSize, curOutputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        recvDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcSelfSlices()
{
    while (!sendToSelfDeque_.empty() && !recvFromSelfDeque_.empty()) {
        const HcclSendRecvItem *sendItem = sendToSelfDeque_.front();
        const HcclSendRecvItem *recvItem = recvFromSelfDeque_.front();
        if (sendItem->count != recvItem->count || sendItem->dataType != recvItem->dataType) {
            HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][CalcSelfSlices] Send task and recv task to self : "
                "count or dataType is not equal, please check the task list.");
            return HCCL_E_PARA;
        }
        u64 dataTypeSize = DATATYPE_SIZE_TABLE[sendItem->dataType];
        CHK_PRT_RET(dataTypeSize == 0,
            HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][CalcSelfSlices] dataTypeSize is zero."),
            HCCL_E_PARA);
        u64 dataSize = sendItem->count * dataTypeSize;
        selfSendSlices_.emplace_back(sendItem->buf, dataSize, static_cast<u32>(myRank_));
        selfRecvSlices_.emplace_back(recvItem->buf, dataSize, static_cast<u32>(myRank_));
        HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][CalcSelfSlices] inputData[%p], "
            "outputData[%p], dataSize[%llu]", sendItem->buf, recvItem->buf, dataSize);
        sendToSelfDeque_.pop_front();
        recvFromSelfDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ParseAndOrganize(
    const HcclSendRecvItem *itemPtr, u32 itemNum)
{
    // 清空状态
    sendDeque_.clear();
    recvDeque_.clear();
    sendToSelfDeque_.clear();
    recvFromSelfDeque_.clear();
    sendDataSlices_.clear();
    recvDataSlices_.clear();
    selfSendSlices_.clear();
    selfRecvSlices_.clear();

    // pair-wise 排序，识别自发自收
    CHK_RET(GetPairWiseList(itemPtr, itemNum));
    // 任务切片
    CHK_RET(CalcSendSlices());
    CHK_RET(CalcRecvSlices());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][Orchestrate] Orchestrate Start.");

    maxTmpMemSize_ = resCtx.cclMem.size > 0 ?
        std::min(resCtx.cclMem.size, UB_MAX_DATA_SIZE) : UB_MAX_DATA_SIZE;
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;

    // 解析并组织任务
    CHK_RET(ParseAndOrganize(itemPtr_, itemNum_));

    // 构建 BatchSendRecvInfo
    BatchSendRecvInfo batchInfo;
    batchInfo.sendSlices = std::move(sendDataSlices_);
    batchInfo.recvSlices = std::move(recvDataSlices_);
    batchInfo.sendToSelfSlices = std::move(selfSendSlices_);
    batchInfo.recvFromSelfSlices = std::move(selfRecvSlices_);

    // 构建 Template
    std::vector<std::vector<u32>> tempAlgHierarchyInfo = resCtx.algHierarchyInfo.infos[0];
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, tempAlgHierarchyInfo);
    algTemplate->SetBatchSendRecvInfo(batchInfo);

    // 准备 TemplateResource
    TemplateResource templateAlgRes;
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo;
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo));
        if (remoteRankToChannelInfo.size() > 0) {
            templateAlgRes.channels = remoteRankToChannelInfo[0];
        }
    }
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }
    templateAlgRes.threads = resCtx.threads;

    // 准备 TemplateDataParams
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;

    // 执行
    HcclResult ret = algTemplate->KernelRun(param, tempAlgParams, templateAlgRes);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][Orchestrate]errNo[0x%016llx] "\
            "BatchSendRecv executor kernel run failed", HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][Orchestrate] Orchestrate End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, InsBatchSendRecv,
    InsV2BatchSendRecvSoleExecutor, TopoMatch1D, InsTempBatchSendRecvMesh1D);

#ifndef AICPU_COMPILE
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, CcuBatchSendRecvMesh1D,
    InsV2BatchSendRecvSoleExecutor, TopoMatch1D, CcuTempBatchSendRecvMesh1D);
#endif

} // namespace ops_hccl
