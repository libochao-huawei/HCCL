/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: ccu jetty manager header file
 * Create: 2025-08-11
 */

#ifndef HCCL_CCU_JETTY_MGR_H
#define HCCL_CCU_JETTY_MGR_H

#include <vector>
#include <unordered_map>

#include "ccu_jetty.h"
#include "hash_utils.h"
#include "ip_address.h"
#include "virtual_topo.h"

namespace Hccl {

class CcuJettyMgr final {
public:
    explicit CcuJettyMgr(int32_t devLogicId);
    ~CcuJettyMgr();

    HcclResult PrepareCreate(const std::vector<LinkData> &links);
    std::pair<CcuChannelInfo, std::vector<CcuJetty *>> GetChannelJettys(const LinkData &link) const;

    void Confirm();
    void Fallback();
    void Clean();
    void Resume();

    uint32_t GetUsedChannelCount(const uint8_t dieId);
    RankId GetRemoteRankIdByChannelId(const uint8_t dieId, const uint32_t channelId);

private:
    int32_t devLogicId_{0};
    bool    isReleased{true};

    struct ResIdHash {
        std::size_t operator()(const std::pair<uint8_t, uint32_t>& p) const
        {
            return HashCombine({p.first, p.second});
        }
    };

    using CcuJettyPtr = CcuJetty*;
    using BatchKey = IpAddress; // srcIpAddress;
    using ResIdkey = std::pair<uint8_t, uint32_t>;
    using ChannelIdKey = ResIdkey;
    using JettyIdKey = ResIdkey;

    struct ResourceBatch { // 记录该批次申请到的所有channel资源信息
        BatchKey key;
        std::vector<ChannelIdKey> channelIdKeys;
        std::vector<ChannelIdKey> availableChannelIdKeys;
        std::unordered_map<JettyIdKey, std::unique_ptr<CcuJetty>, ResIdHash> jettys;
    
        ResourceBatch(const BatchKey &batchKey, const std::vector<CcuChannelInfo> &channelInfos)
            : key(batchKey)
        {
            const uint32_t channelNum = channelInfos.size();
            channelIdKeys.reserve(channelNum);
            availableChannelIdKeys.reserve(channelNum);
            for (const auto &channelInfo : channelInfos) {
                const auto dieId = channelInfo.dieId;
                const auto channelId = channelInfo.channelId;
                channelIdKeys.emplace_back(dieId, channelId);
                availableChannelIdKeys.emplace_back(dieId, channelId);

                for (const auto &jettyInfo : channelInfo.jettyInfos) {
                    const auto taJettyId = jettyInfo.taJettyId;
                    const auto jettyIdKey = std::make_pair(dieId, taJettyId);
                    if (jettys.find(jettyIdKey) != jettys.end()) {
                        continue;
                    }

                    std::unique_ptr<CcuJetty> ccuJetty;
                    CHK_RET_THROW(InternalException,
                        StringFormat("[CcuJettyMgr][%s] failed to create ccu jetty, locAddr[%s] "
                            "dieId[%u] taJettyId[%u].", __func__, key.Describe().c_str(),
                            dieId, taJettyId),
                        CcuCreateJetty(key, jettyInfo, ccuJetty));

                    jettys[jettyIdKey] = std::move(ccuJetty);
                }
            }
        }
    };

    struct Allocation {
        LinkData link;
        ChannelIdKey channelIdKey;
        ResourceBatch *batchPtr;
    };

    struct UnconfirmedRecord {
        std::vector<Allocation> allocations; // 记录从已申请的资源中的分配操作
        std::unordered_set<ResourceBatch *> newBatchSet; // 记录新申请资源的操作

        void Clear() {
            allocations.clear();
            newBatchSet.clear();
        }
    };

    // 本轮下发算子新增分配记录
    UnconfirmedRecord unconfirmedRecord_;
    // 各资源申请记录，当前按SrcIpAddr粒度申请和管理
    std::unordered_map<BatchKey, std::vector<std::unique_ptr<ResourceBatch>>> batchMap_;
    // 各link已分配的channel资源Id信息
    std::unordered_map<LinkData, ChannelIdKey> allocatedChannelIdMap_;
    // 全部已申请的channel资源信息，资源申请成功后将要记录到该map中
    using CcuChannelJettyInfo = std::pair<CcuChannelInfo, std::vector<CcuJetty *>>;
    std::unordered_map<ChannelIdKey, CcuChannelJettyInfo, ResIdHash> channelJettyInfoMap_;
    // 以die粒度记录已分配channel资源, index: dieId
    std::unordered_map<uint8_t, uint32_t> usedChannelCntMap_;
    // 记录channel与对端rank的映射关系, index: (die, channelId)
    std::unordered_map<ChannelIdKey, RankId, ResIdHash> channelRemoteRankIdMap_;

    HcclResult GetAvailableBatch(const BatchKey &batchKey, ResourceBatch *&batchPtr);
    bool FindAvailableBatch(const BatchKey &batchKey, ResourceBatch *&batchPtr) const;
    HcclResult CreateAndSaveNewBatch(const BatchKey &batchKey,
        const std::vector<CcuChannelInfo> channelInfos, ResourceBatch *&batchPtr);
    void FallbackAndRemoveBatches();
    void FallbackAllocatedChannelJettyInfo();
    void ReleaseConfirmedChannelRes();
};

} // namespace Hccl

#endif // HCCL_CCU_JETTY_MGR_H