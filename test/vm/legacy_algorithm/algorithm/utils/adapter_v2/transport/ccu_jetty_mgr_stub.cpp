/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu jetty manager
 * Create: 2025-08-11
 */

#include "ccu_jetty_mgr.h"

#include <unordered_set>

#include "ccu_dev_mgr.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"

namespace Hccl {

constexpr uint32_t CCU_DEFAULT_REQUEST_SQ_SIZE = 128;
constexpr uint32_t CCU_DEFAULT_REQUEST_CHANNEL_NUM = 1;
constexpr uint32_t CCU_DEFAULT_REQUEST_JETTY_NUM = 0; // 申请数量为0时，由平台层决定提供数量

CcuJettyMgr::CcuJettyMgr(int32_t devLogicId): devLogicId_(devLogicId)
{
}

CcuJettyMgr::~CcuJettyMgr()
{
    // 对象析构时清空多个map，ccuJettyMap_清空触发ccuJetty析构释放
    DECTOR_TRY_CATCH("CcuJettyMgr", ReleaseConfirmedChannelRes());
}

HcclResult CcuJettyMgr::PrepareCreate(const std::vector<LinkData> &links)
{
    CHK_PRT_RET(links.empty(),
        HCCL_INFO("[CcuJettyMgr][%s] passed, links is empty, devLogicId[%d].",
            __func__, devLogicId_),
        HcclResult::HCCL_SUCCESS);

    for (const auto &link : links) {
        auto it = allocatedChannelIdMap_.find(link);
        if (it != allocatedChannelIdMap_.end()) {
            continue;
        }

        const auto &locAddr = link.GetLocalAddr();
        ResourceBatch *batchPtr = nullptr;
        auto ret = GetAvailableBatch(locAddr, batchPtr);
        CHK_PRT_RET(ret == HcclResult::HCCL_E_UNAVAIL,
            HCCL_WARNING("[CcuJettyMgr][%s] alloc ccu channels failed, ccu resources "
                "are unavaialble, locAddr[%s], devLogicId[%d].",
                __func__, locAddr.Describe().c_str(), devLogicId_),
            ret);
        CHK_RET(ret);
    
        ChannelIdKey channelIdKey = batchPtr->availableChannelIdKeys.back();
        batchPtr->availableChannelIdKeys.pop_back();
        unconfirmedRecord_.allocations.emplace_back(Allocation{link, channelIdKey, batchPtr});
        allocatedChannelIdMap_[link] = channelIdKey;
        channelRemoteRankIdMap_[channelIdKey] = link.GetRemoteRankId();
    }

    isReleased = false;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyMgr::GetAvailableBatch(const BatchKey &batchKey, ResourceBatch *&batchPtr)
{
    // 当前以locAddr作为batchKey，不同本端不能复用资源
    if (FindAvailableBatch(batchKey, batchPtr)) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 已有的资源不足，需要新增资源，获取的channel数量可能超过申请数量
    const CcuChannelPara channelPara{batchKey, CCU_DEFAULT_REQUEST_CHANNEL_NUM,
            CCU_DEFAULT_REQUEST_JETTY_NUM, CCU_DEFAULT_REQUEST_SQ_SIZE};
    std::vector<CcuChannelInfo> channelInfos;
    auto ret = CcuAllocChannels(devLogicId_, channelPara, channelInfos);
    CHK_PRT_RET(ret == HcclResult::HCCL_E_UNAVAIL,
        HCCL_WARNING("[CcuJettyMgr][%s] alloc ccu channels failed, ccu resources "
            "are unavaialble, locAddr[%s] devLogicId[%d].", __func__,
            batchKey.Describe().c_str(), devLogicId_),
        ret);
    CHK_RET(ret);
    // 如果新增资源保存失败，手动释放避免泄露
    ret = CreateAndSaveNewBatch(batchKey, channelInfos, batchPtr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CcuJettyMgr][%s] failed, release temp ccu resources, locAddr[%s], "
            "devLogicId[%d], .", __func__, batchKey.Describe().c_str(), devLogicId_);
        for (const auto &channelInfo : channelInfos) {
            const auto dieId = channelInfo.dieId;
            const auto channelId = channelInfo.channelId;
            CHK_RET(CcuReleaseChannel(devLogicId_, dieId, channelId));
        }
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJettyMgr::CreateAndSaveNewBatch(const BatchKey &batchKey,
    const std::vector<CcuChannelInfo> channelInfos, ResourceBatch *&batchPtr)
{
    // 该流程如果抛异常，可能资源还未记录，析构无法释放导致资源泄露，故捕获异常处理
    TRY_CATCH_RETURN(
        auto &batches = batchMap_[batchKey];
        auto newBatch = std::make_unique<ResourceBatch>(batchKey, channelInfos);
        for (const auto &channelInfo : channelInfos) {
            const auto dieId = channelInfo.dieId;
            const auto channelIdKey = std::make_pair(dieId, channelInfo.channelId);

            std::vector<CcuJetty *> jettys;
            for (const auto &jettyInfo : channelInfo.jettyInfos) {
                const auto jettyIdKey = std::make_pair(dieId, jettyInfo.taJettyId);
                jettys.emplace_back(newBatch->jettys[jettyIdKey].get());
            }

            channelJettyInfoMap_.emplace(channelIdKey, std::make_pair(channelInfo, jettys));
            usedChannelCntMap_[dieId] += 1;
        }

        batches.push_back(std::move(newBatch));
        ResourceBatch *rawBatch = batches.back().get();
        
        unconfirmedRecord_.newBatchSet.insert(rawBatch);
        batchPtr = rawBatch;
    );
    return HcclResult::HCCL_SUCCESS;
}

bool CcuJettyMgr::FindAvailableBatch(const BatchKey &batchKey, ResourceBatch *&batchPtr) const
{
    auto it = batchMap_.find(batchKey);
    if (it == batchMap_.end()) {
        return false;
    }

    auto &batches = it->second;
    if (batches.empty()) {
        return false;
    }
    // 当前分配逻辑只有最后一个batch可能还有空闲资源
    auto &lastBatch = batches.back();
    if (lastBatch->availableChannelIdKeys.empty()) {
        return false;
    }

    batchPtr = lastBatch.get();
    return true;
}

CcuJettyMgr::CcuChannelJettyInfo CcuJettyMgr::GetChannelJettys(const LinkData &link) const
{
    const auto &it = allocatedChannelIdMap_.find(link);
    CHK_RET_THROW(InternalException,
        StringFormat("[CcuJettyMgr][%s] failed to find allocated channelId of link[%s], ",
            "devLogicId[%d].", __func__, link.Describe().c_str(), devLogicId_),
        it == allocatedChannelIdMap_.end());
    // 内部维护数据保证channelJettyInfoMap_记录的资源存在
    return channelJettyInfoMap_.at(it->second);
}

void CcuJettyMgr::Confirm()
{
    unconfirmedRecord_.Clear();
}

void CcuJettyMgr::Fallback()
{
    FallbackAndRemoveBatches();
    FallbackAllocatedChannelJettyInfo();
    Confirm();
}

void CcuJettyMgr::FallbackAndRemoveBatches()
{
    using BatchSetKey = std::unordered_map<BatchKey,
        std::unordered_set<ResourceBatch *>>;
    BatchSetKey batchesToRemoveByKey;
    for (const auto &allocation : unconfirmedRecord_.allocations) {
        ResourceBatch *batchPtr = allocation.batchPtr;
        if (!batchPtr) {
            continue;
        }

        const auto &newBatchSet = unconfirmedRecord_.newBatchSet;
        bool isNewBatch = newBatchSet.find(batchPtr) != newBatchSet.end();
        if (!isNewBatch) { // 如果是之前申请的资源需要回退到待分配状态
            batchPtr->availableChannelIdKeys.push_back(allocation.channelIdKey);
            continue;
        }

        batchesToRemoveByKey[batchPtr->key].insert(batchPtr);
    }
    // 本轮新申请的资源需要调用接口释放，使用set去重
    for (const auto &keyEntry : batchesToRemoveByKey) {
        const auto &key = keyEntry.first;
        const auto &batchesToRemove = keyEntry.second;

        for (const auto &batchPtr: batchesToRemove) {
            for (const auto &channelIdKey : batchPtr->channelIdKeys) {
                const auto dieId = channelIdKey.first;
                const auto channelId = channelIdKey.second;
                CHK_RET_THROW(InternalException,
                    StringFormat("[CcuJettyMgr][%s] failed to release channelId[%u] of "
                        "die[%u], devLogicId[%d].", __func__, channelId, dieId, devLogicId_),
                    CcuReleaseChannel(devLogicId_, dieId, channelId));

                usedChannelCntMap_[dieId] -= 1;
            }
        }
        // 从batchMap_中清理新增的batch
        auto &batches = batchMap_[key];
        batches.erase(std::remove_if(batches.begin(), batches.end(),
            [&batchesToRemove](const auto &batchPtr) {
                return batchesToRemove.find(batchPtr.get()) != batchesToRemove.end();
            }), batches.end());
    }
}

void CcuJettyMgr::FallbackAllocatedChannelJettyInfo()
{
    // 清理非batch粒度记录的需回退的资源
    for (const auto &allocation : unconfirmedRecord_.allocations) {
        allocatedChannelIdMap_.erase(allocation.link);
        channelJettyInfoMap_.erase(allocation.channelIdKey);
        channelRemoteRankIdMap_.erase(allocation.channelIdKey);
    }
}

void CcuJettyMgr::Clean()
{
    ReleaseConfirmedChannelRes();
    // n秒快恢需要记录链路信息，故仅清空channel信息
    for (auto &linkEntry : allocatedChannelIdMap_) {
        linkEntry.second = {0, 0};
    }

    batchMap_.clear();
    channelJettyInfoMap_.clear();
    channelRemoteRankIdMap_.clear();
    usedChannelCntMap_.clear();
    Confirm();
}

void CcuJettyMgr::ReleaseConfirmedChannelRes()
{
    isReleased = true;
    for (const auto &infoEntry : channelJettyInfoMap_) {
        const auto &channelIdKey = infoEntry.first;
        const auto dieId = channelIdKey.first;
        const auto channelId = channelIdKey.second;
        CHK_RET_THROW(InternalException,
            StringFormat("[CcuJettyMgr][%s] failed to release channelId[%u] of "
                "die[%u], devLogicId[%d].", __func__, channelId, dieId, devLogicId_),
            CcuReleaseChannel(devLogicId_, dieId, channelId));
    }
}

void CcuJettyMgr::Resume()
{
    if (!isReleased) {
        THROW<InternalException>("[CcuJettyMgr][%s] failed, the ccu resource "
            "has not been released yet, devLogicId[%d].", __func__, devLogicId_);
    }

    std::vector<LinkData> links;
    links.reserve(allocatedChannelIdMap_.size());
    for (const auto &linkEntry : allocatedChannelIdMap_) {
        links.push_back(linkEntry.first);
    }
    allocatedChannelIdMap_.clear();

    CHK_RET_THROW(InternalException,
        StringFormat("[CcuJettyMgr][%s] failed to resume ccu jettys, devLogicId[%d].",
            __func__, devLogicId_),
        PrepareCreate(links));
}

uint32_t CcuJettyMgr::GetUsedChannelCount(const uint8_t dieId)
{
    const auto &dieIter = usedChannelCntMap_.find(dieId);
    if (dieIter != usedChannelCntMap_.end()) {
        return dieIter->second;
    }
    return 0;
}

RankId CcuJettyMgr::GetRemoteRankIdByChannelId(const uint8_t dieId, const uint32_t channelId)
{
    const auto &iter = channelRemoteRankIdMap_.find({dieId, channelId});
    if (iter == channelRemoteRankIdMap_.end()) {
        THROW<InvalidParamsException>("[CcuJettyMgr][%s] failed to find remoteRankId by "
            "dieId[%u] channelId[%u], devLogicId[%d].", __func__, dieId, channelId,
            devLogicId_);
    }

    HCCL_INFO("[CcuJettyMgr][%s] find remoteRankId[%d] by dieId[%u] channelId[%u], "
        "all size[%zu], devLogicId[%d].", __func__, iter->second, dieId, channelId,
        channelRemoteRankIdMap_.size(), devLogicId_);
    return iter->second;
}

} // namespace Hccl