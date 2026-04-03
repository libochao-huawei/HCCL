#include "all_gather_hd_stage_core.h"
#include <vector>
#include "all_gather_nhr_core.h"
#include "log.h"
namespace ops_hccl_allgatherbatch {
namespace {
uint32_t CalcTrailingPowerSteps(uint32_t rankSize)
{
    uint32_t steps = 0;
    while ((rankSize & 1U) == 0U && rankSize > 1U) {
        rankSize >>= 1U;
        ++steps;
    }
    return steps;
}
uint32_t CalcFinalSteps(uint32_t powerSteps)
{
    if (powerSteps >= 2U) {
        return 2U;
    }
    if (powerSteps >= 1U) {
        return 1U;
    }
    return 0U;
}
}  // namespace
AllGatherHDStageCore::AllGatherHDStageCore(const OpParam &param, AlgResourceCtx &resCtx, uint64_t packedBytes)
    : param_(param), resCtx_(resCtx), packedBytes_(packedBytes)
{
}
// Stage ??????????????? HDStage??????????? NHR ?????????????
HcclResult AllGatherHDStageCore::ValidateStageInput() const
{
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    if (!IsValidCommMode(param_.commMode)) {
        HCCL_ERROR("HDStage commMode is invalid");
        return HCCL_E_INTERNAL;
    }
    if (!HasConsistentRankDistribution(param_)) {
        HCCL_ERROR("HDStage rank distribution is inconsistent, commMode=%s, intra=%u, cross=%u, rankSize=%u",
            ToCommModeString(param_.commMode),
            param_.intraServerRankCount,
            param_.crossServerRankCount,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (param_.topoInfo.rankSize == 0) {
        HCCL_ERROR("HDStage rankSize is zero");
        return HCCL_E_PARA;
    }
    if (packedBytes_ == 0) {
        HCCL_ERROR("HDStage packedBytes is zero");
        return HCCL_E_PARA;
    }
    if (param_.windowBytes == 0) {
        HCCL_ERROR("HDStage windowBytes is zero");
        return HCCL_E_INTERNAL;
    }
    if (packedBytes_ > param_.windowBytes) {
        HCCL_ERROR("HDStage packedBytes=%llu exceeds param windowBytes=%llu",
            static_cast<unsigned long long>(packedBytes_),
            static_cast<unsigned long long>(param_.windowBytes));
        return HCCL_E_INTERNAL;
    }
    if (packedBytes_ > stats.maxWindowBytes) {
        HCCL_ERROR("HDStage packedBytes=%llu exceeds maxWindowBytes=%llu",
            static_cast<unsigned long long>(packedBytes_),
            static_cast<unsigned long long>(stats.maxWindowBytes));
        return HCCL_E_INTERNAL;
    }
    const uint64_t totalBytes = packedBytes_ * param_.topoInfo.rankSize;
    if (resCtx_.localBuffer.addr == nullptr || resCtx_.localBuffer.size < totalBytes) {
        HCCL_ERROR("HDStage localBuffer is too small, need=%llu, actual=%llu",
            static_cast<unsigned long long>(totalBytes),
            static_cast<unsigned long long>(resCtx_.localBuffer.size));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::BuildStagePlan(HDStagePlan &plan) const
{
    const uint32_t rankSize = param_.topoInfo.rankSize;
    if (rankSize == 0) {
        HCCL_ERROR("rankSize is zero");
        return HCCL_E_PARA;
    }
    // ?????????????????
    // 1. noPower ??? 2 ????????????? NHR?
    // 2. power ??? 2 ?????? final ??????
    // 3. final ???? 0/1/2 ???????
    plan.powerSteps = CalcTrailingPowerSteps(rankSize);
    plan.powerFactor = (plan.powerSteps == 0U) ? 1U : (1U << plan.powerSteps);
    plan.noPower = rankSize / plan.powerFactor;
    plan.finalSteps = CalcFinalSteps(plan.powerSteps);
    plan.remainingPowerSteps = (plan.powerSteps > plan.finalSteps) ? (plan.powerSteps - plan.finalSteps) : 0U;
    plan.needNoPowerPath = (plan.noPower > 1U);
    plan.needPowerPath = (plan.remainingPowerSteps >= 1U);
    plan.needFinalPath = (rankSize > 1U);
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::ValidateStagePlan(const HDStagePlan &plan) const
{
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    if (plan.noPower == 0) {
        HCCL_ERROR("HDStage noPower is zero");
        return HCCL_E_INTERNAL;
    }
    if ((plan.noPower * plan.powerFactor) != param_.topoInfo.rankSize) {
        HCCL_ERROR("HDStage plan mismatch, noPower=%u, powerFactor=%u, rankSize=%u",
            plan.noPower,
            plan.powerFactor,
            param_.topoInfo.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (plan.finalSteps > plan.powerSteps) {
        HCCL_ERROR("HDStage finalSteps exceeds powerSteps, finalSteps=%u, powerSteps=%u",
            plan.finalSteps,
            plan.powerSteps);
        return HCCL_E_INTERNAL;
    }
    if (plan.remainingPowerSteps + plan.finalSteps != plan.powerSteps) {
        HCCL_ERROR("HDStage power split mismatch, remainingPowerSteps=%u, finalSteps=%u, powerSteps=%u",
            plan.remainingPowerSteps,
            plan.finalSteps,
            plan.powerSteps);
        return HCCL_E_INTERNAL;
    }
    if (stats.intraServerChannels + stats.crossServerChannels != resCtx_.channelCount) {
        HCCL_ERROR("HDStage channel scope split is inconsistent, intra=%u, cross=%u, channelCount=%u",
            stats.intraServerChannels,
            stats.crossServerChannels,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kSingleServer && stats.crossServerChannels != 0) {
        HCCL_ERROR("HDStage single-server mode unexpectedly has cross-server channels=%u", stats.crossServerChannels);
        return HCCL_E_INTERNAL;
    }
    if (param_.commMode == BatchCommMode::kCrossServer && stats.crossServerChannels == 0) {
        HCCL_ERROR("HDStage cross-server mode has no cross-server channels");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::ValidateProtocolDistribution() const
{
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    if (stats.recognizedProtocols != resCtx_.channelCount) {
        HCCL_ERROR("HDStage protocol distribution mismatch, recognized=%u, channelCount=%u",
            stats.recognizedProtocols,
            resCtx_.channelCount);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::BuildNHRSubgroupCtx(const HDStagePlan &plan, NHRSubgroupCtx &subgroupCtx) const
{
    if (!plan.needNoPowerPath) {
        HCCL_ERROR("HDStage noPower subgroup is requested while noPower path is disabled");
        return HCCL_E_INTERNAL;
    }
    if (plan.noPower <= 1U || plan.powerFactor == 0U) {
        HCCL_ERROR("HDStage noPower subgroup is invalid, noPower=%u, powerFactor=%u",
            plan.noPower,
            plan.powerFactor);
        return HCCL_E_INTERNAL;
    }
    const uint32_t rank = param_.topoInfo.rank;
    const uint32_t group = rank / plan.powerFactor;
    const uint32_t groupIdx = rank % plan.powerFactor;
    subgroupCtx.subgroupRank = group;
    subgroupCtx.subgroupSize = plan.noPower;
    subgroupCtx.baseOffset = 0;
    subgroupCtx.subgroupRanks.clear();
    subgroupCtx.subgroupRanks.reserve(plan.noPower);
    for (uint32_t idx = 0; idx < plan.noPower; ++idx) {
        subgroupCtx.subgroupRanks.push_back(idx * plan.powerFactor + groupIdx);
    }
    if (subgroupCtx.subgroupRank >= subgroupCtx.subgroupSize) {
        HCCL_ERROR("HDStage subgroupRank=%u is out of range, subgroupSize=%u",
            subgroupCtx.subgroupRank,
            subgroupCtx.subgroupSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage built NHR subgroup: rank=%u, group=%u, groupIdx=%u, subgroupRank=%u, subgroupSize=%u",
        rank,
        group,
        groupIdx,
        subgroupCtx.subgroupRank,
        subgroupCtx.subgroupSize);
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunPreCopy() const
{
    // ?? custom-op ?????????????? Pack ?????
    // HDStage ?????? PreCopy ?????????????????????
    uint8_t *localRankBase = static_cast<uint8_t *>(resCtx_.localBuffer.addr) +
        (packedBytes_ * param_.topoInfo.rank);
    if (localRankBase == nullptr) {
        HCCL_ERROR("HDStage pre-copy base is null");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("HDStage pre-copy ready: rank=%u, packedBytes=%llu, localRankBase=%p",
        param_.topoInfo.rank,
        static_cast<unsigned long long>(packedBytes_),
        localRankBase);
    return HCCL_SUCCESS;
}
// HDStage ? NoPower ???????????????????????? NHR?
HcclResult AllGatherHDStageCore::RunNHR(const char *pathTag, const NHRSubgroupCtx &subgroupCtx) const
{
    HCCL_INFO("HDStage delegates subgroup to NHR: path=%s, rank=%u, packedBytes=%llu, subgroupRank=%u, subgroupSize=%u",
        pathTag,
        param_.topoInfo.rank,
        static_cast<unsigned long long>(packedBytes_),
        subgroupCtx.subgroupRank,
        subgroupCtx.subgroupSize);
    AllGatherNHRCore nhrCore(param_, resCtx_, packedBytes_, subgroupCtx);
    return nhrCore.RunAsync();
}
const ChannelResource *AllGatherHDStageCore::FindChannel(uint32_t remoteRank) const
{
    for (uint32_t idx = 0; idx < resCtx_.channelCount; ++idx) {
        const ChannelResource &channel = GetChannel(resCtx_, idx);
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}
HcclResult AllGatherHDStageCore::BuildPartnerRanksForBit(
    const HDStagePlan &plan, uint32_t bit, std::vector<uint32_t> &partnerRanks) const
{
    if (plan.powerFactor == 0U || bit >= plan.powerSteps) {
        HCCL_ERROR("HDStage power bit is invalid, bit=%u, powerSteps=%u, powerFactor=%u",
            bit,
            plan.powerSteps,
            plan.powerFactor);
        return HCCL_E_INTERNAL;
    }
    const uint32_t groupIdx = param_.topoInfo.rank % plan.powerFactor;
    const uint32_t lowMask = (bit == 0U) ? 0U : ((1U << bit) - 1U);
    const uint32_t flippedBit = (((groupIdx >> bit) & 1U) ^ 1U);
    partnerRanks.clear();
    for (uint32_t column = 0; column < plan.powerFactor; ++column) {
        if (bit != 0U && ((column & lowMask) != (groupIdx & lowMask))) {
            continue;
        }
        if (((column >> bit) & 1U) != flippedBit) {
            continue;
        }
        for (uint32_t group = 0; group < plan.noPower; ++group) {
            partnerRanks.push_back(group * plan.powerFactor + column);
        }
    }
    if (partnerRanks.empty()) {
        HCCL_ERROR("HDStage partner rank set is empty, bit=%u, groupIdx=%u", bit, groupIdx);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::ReadPartnerRanks(
    uint32_t partnerRank, const std::vector<uint32_t> &partnerRanks, const char *stageTag) const
{
    const ChannelResource *channel = FindChannel(partnerRank);
    if (channel == nullptr) {
        HCCL_ERROR("HDStage %s channel to partnerRank=%u is missing", stageTag, partnerRank);
        return HCCL_E_NOT_FOUND;
    }
    if (channel->remoteBuffer.addr == nullptr ||
        channel->remoteBuffer.size < (packedBytes_ * param_.topoInfo.rankSize)) {
        HCCL_ERROR("HDStage %s remote buffer is too small, partnerRank=%u, need=%llu, actual=%llu",
            stageTag,
            partnerRank,
            static_cast<unsigned long long>(packedBytes_ * param_.topoInfo.rankSize),
            static_cast<unsigned long long>(channel->remoteBuffer.size));
        return HCCL_E_INTERNAL;
    }
    int32_t ret = HcommChannelNotifyRecordOnThread(
        resCtx_.threadHandle,
        channel->handle,
        channel->remoteNotifyIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify record failed, partnerRank=%u, ret=%d", stageTag, partnerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    ret = HcommChannelNotifyWaitOnThread(
        resCtx_.threadHandle,
        channel->handle,
        channel->localNotifyIdx,
        kAllGatherBatchCustomTimeoutMs);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HDStage %s notify wait failed, partnerRank=%u, ret=%d", stageTag, partnerRank, ret);
        return static_cast<HcclResult>(ret);
    }
    for (uint32_t remoteRank : partnerRanks) {
        void *dst = static_cast<uint8_t *>(resCtx_.localBuffer.addr) + (packedBytes_ * remoteRank);
        const void *src = static_cast<const uint8_t *>(channel->remoteBuffer.addr) + (packedBytes_ * remoteRank);
        ret = HcommReadOnThread(resCtx_.threadHandle, channel->handle, dst, src, packedBytes_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("HDStage %s remote read failed, partnerRank=%u, remoteRank=%u, ret=%d",
                stageTag,
                partnerRank,
                remoteRank,
                ret);
            return static_cast<HcclResult>(ret);
        }
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunPowerBit(const HDStagePlan &plan, uint32_t bit, const char *stageTag) const
{
    const uint32_t rank = param_.topoInfo.rank;
    const uint32_t group = rank / plan.powerFactor;
    const uint32_t groupIdx = rank % plan.powerFactor;
    const uint32_t partnerGroupIdx = groupIdx ^ (1U << bit);
    const uint32_t partnerRank = group * plan.powerFactor + partnerGroupIdx;
    std::vector<uint32_t> partnerRanks;
    HCCL_CHK_RET(BuildPartnerRanksForBit(plan, bit, partnerRanks));
    HCCL_INFO("HDStage %s bit exchange: rank=%u, bit=%u, partnerRank=%u, partnerRanks=%u, packedBytes=%llu",
        stageTag,
        rank,
        bit,
        partnerRank,
        static_cast<uint32_t>(partnerRanks.size()),
        static_cast<unsigned long long>(packedBytes_));
    return ReadPartnerRanks(partnerRank, partnerRanks, stageTag);
}
HcclResult AllGatherHDStageCore::RunAllGatherNoPower(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage noPower stage: rank=%u, noPower=%u, powerSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.noPower,
        plan.powerSteps,
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needNoPowerPath) {
        HCCL_INFO("HDStage noPower stage is skipped because noPower <= 1");
        return HCCL_SUCCESS;
    }
    NHRSubgroupCtx subgroupCtx;
    HCCL_CHK_RET(BuildNHRSubgroupCtx(plan, subgroupCtx));
    return RunNHR("noPower", subgroupCtx);
}
HcclResult AllGatherHDStageCore::RunAllGatherPower(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage power stage: rank=%u, rankSize=%u, commMode=%s, remainingPowerSteps=%u, finalSteps=%u, packedBytes=%llu",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        plan.remainingPowerSteps,
        plan.finalSteps,
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needPowerPath) {
        HCCL_INFO("HDStage power stage is skipped because remainingPowerSteps is zero");
        return HCCL_SUCCESS;
    }
    for (uint32_t step = 0; step < plan.remainingPowerSteps; ++step) {
        const uint32_t bit = plan.powerSteps - 1U - step;
        HCCL_CHK_RET(RunPowerBit(plan, bit, "power"));
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunAllGatherLast(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last stage: rank=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    // finalSteps == 0 ???? power bit ??????? NoPower/Power ?????
    // ????????????????????????????
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunAllGatherLastOne(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last-one stage: rank=%u, powerSteps=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needFinalPath) {
        return HCCL_SUCCESS;
    }
    HCCL_CHK_RET(RunPowerBit(plan, 0U, "lastOne"));
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunAllGatherLastTwo(const HDStagePlan &plan) const
{
    HCCL_INFO("HDStage last-two stage: rank=%u, powerSteps=%u, needFinal=%s, packedBytes=%llu",
        param_.topoInfo.rank,
        plan.powerSteps,
        plan.needFinalPath ? "true" : "false",
        static_cast<unsigned long long>(packedBytes_));
    if (!plan.needFinalPath) {
        return HCCL_SUCCESS;
    }
    HCCL_CHK_RET(RunPowerBit(plan, 1U, "lastTwo-1"));
    HCCL_CHK_RET(RunPowerBit(plan, 0U, "lastTwo-0"));
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunAllGatherFinal(const HDStagePlan &plan) const
{
    if (!plan.needFinalPath) {
        HCCL_INFO("HDStage final stage is skipped because rankSize <= 1");
        return HCCL_SUCCESS;
    }
    if (plan.finalSteps >= 2U) {
        return RunAllGatherLastTwo(plan);
    }
    if (plan.finalSteps == 1U) {
        return RunAllGatherLastOne(plan);
    }
    return RunAllGatherLast(plan);
}
HcclResult AllGatherHDStageCore::RunAllGatherStage(const HDStagePlan &plan) const
{
    HCCL_CHK_RET(RunPreCopy());
    if (plan.needNoPowerPath) {
        HCCL_CHK_RET(RunAllGatherNoPower(plan));
    }
    if (plan.needPowerPath) {
        HCCL_CHK_RET(RunAllGatherPower(plan));
    }
    if (plan.needFinalPath) {
        HCCL_CHK_RET(RunAllGatherFinal(plan));
    }
    return HCCL_SUCCESS;
}
HcclResult AllGatherHDStageCore::RunAsync()
{
    if (param_.topoInfo.rankSize == 1) {
        HCCL_INFO("HDStage fast path: rankSize=1, no staged communication needed");
        return HCCL_SUCCESS;
    }
    HCCL_CHK_RET(ValidateStageInput());
    HDStagePlan plan;
    HCCL_CHK_RET(BuildStagePlan(plan));
    HCCL_CHK_RET(ValidateStagePlan(plan));
    HCCL_CHK_RET(ValidateProtocolDistribution());
    const ResourceStats stats = CollectResourceStats(param_, resCtx_);
    HCCL_INFO("HDStage plan ready: rank=%u, rankSize=%u, commMode=%s, serverIdx=%u, intraServerRankCount=%u, crossServerRankCount=%u, noPower=%u, powerFactor=%u, powerSteps=%u, remainingPowerSteps=%u, finalSteps=%u, packedBytes=%llu, paramWindowBytes=%llu, maxWindowBytes=%llu, intraServerChannels=%u, crossServerChannels=%u, hccs=%u, roce=%u, pcie=%u, sio=%u",
        param_.topoInfo.rank,
        param_.topoInfo.rankSize,
        ToCommModeString(param_.commMode),
        param_.topoInfo.serverIdx,
        param_.intraServerRankCount,
        param_.crossServerRankCount,
        plan.noPower,
        plan.powerFactor,
        plan.powerSteps,
        plan.remainingPowerSteps,
        plan.finalSteps,
        static_cast<unsigned long long>(packedBytes_),
        static_cast<unsigned long long>(param_.windowBytes),
        static_cast<unsigned long long>(stats.maxWindowBytes),
        stats.intraServerChannels,
        stats.crossServerChannels,
        stats.hccsChannels,
        stats.roceChannels,
        stats.pcieChannels,
        stats.sioChannels);
    HCCL_CHK_RET(RunAllGatherStage(plan));
    HCCL_INFO("HDStage finished: rank=%u, noPower=%s, power=%s, final=%s",
        param_.topoInfo.rank,
        plan.needNoPowerPath ? "true" : "false",
        plan.needPowerPath ? "true" : "false",
        plan.needFinalPath ? "true" : "false");
    return HCCL_SUCCESS;
}
}  // namespace ops_hccl_allgatherbatch
