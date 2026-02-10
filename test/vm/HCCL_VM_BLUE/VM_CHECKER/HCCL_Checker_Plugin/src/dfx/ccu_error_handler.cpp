/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: CcuErrorHandler
 * Author: cuizheng
 * Create: 2025-04-13
 */

#include "ccu_error_handler.h"
#include "ccu_context_mgr_imp.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"

namespace Hccl {
using namespace std;
using namespace CcuRep;

const map<uint8_t, string> MISSION_STATUS_MAP{
    {0x01, "Unsupported Opcode(0x01)"},      {0x02, "Local Operation Error(0x02)"},
    {0x03, "Remote Operation Error(0x03)"},  {0x04, "Transaction Retry Counter Exceeded(0x04)"},
    {0x05, "Transaction ACK Timeout(0x05)"}, {0x06, "Jetty Work Request Flushed(0x06)"},
    {0x07, "CCUA Alg Task Error(0x07)"},     {0x08, "Memory ECC Error(0x08)"},
    {0x09, "CCUM Execute Error(0x09)"},      {0x0A, "CCUA Execute Error(0x0A)"},
};

const map<uint8_t, map<uint8_t, string>> MISSION_SUB_STATUS_MAP{
    {0x02,
     {{0x01, "Local Length Error(0x01)"},
      {0x02, "Local Access Error(0x02)"},
      {0x03, "Remote Response Length Error(0x03)"},
      {0x04, "Local Data Poison(0x04)"}}},
    {0x03,
     {{0x01, "Remote Unsupported Request(0x01)"},
      {0x02, "Remote Access Abort(0x02)"},
      {0x04, "Remote Data Poison(0x04)"}}},
    {0x09, {{0x01, "SQE instr and key not match(0x01)"}, {0x02, "CCU Mission Task Killed(0x02)"}}},
    {0x0A,
     {{0x01, "EXOKAY(0x01)"},
      {0x11, "EXOKAY(0x11)"},
      {0x02, "SLVERR(0x02)"},
      {0x12, "SLVERR(0x12)"},
      {0x03, "DECERR(0x03)"},
      {0x13, "DECERR(0x13)"},
      {0x04, "abort(0x04)"},
      {0x14, "abort(0x14)"},
      {0x05, "write permission err(0x05)"},
      {0x15, "write permission err(0x15)"},
      {0x06, "read permission err(0x06)"},
      {0x16, "read permission err(0x16)"},
      {0x07, "atomic permission err(0x07)"},
      {0x17, "atomic permission err(0x17)"},
      {0x08, "tokenval err(0x08)"},
      {0x18, "tokenval err(0x18)"},
      {0x09, "page fault(0x09)"},
      {0x0a, "page fault(0x0a)"},
      {0x0b, "page fault(0x0b)"},
      {0x19, "page fault(0x19)"},
      {0x1a, "page fault(0x1a)"},
      {0x1b, "page fault(0x1b)"},
      {0x0c, "read local mem poison(0x0c)"}}},
};

void CcuErrorHandler::GetCcuErrorMsg(int32_t deviceId, const ParaCcu &ccuTaskParam, vector<CcuErrorInfo> &errorInfo)
{
    // 检查mission status
    const auto missionContext = GetCcuMissionContext(deviceId, ccuTaskParam.dieId, ccuTaskParam.execMissionId);
    const uint16_t missionStatus = missionContext.GetStatus();
    if (missionStatus == 0) {
        HCCL_INFO("[CcuErrorHandler][%s] no err found, mission status is 0, deviceId[%d], dieId[%u], execMissionId[%u]",
            __func__, deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.execMissionId));
        return;
    }

    // 获取异常指令对应的Rep
    CcuRepContext *ctx
        = CtxMgrImp::GetInstance(deviceId).GetCtx(ccuTaskParam.executeId, ccuTaskParam.dieId, ccuTaskParam.missionId);
    if (ctx == nullptr) {
        THROW<CcuApiException>("CcuContext not found, deviceId[%d], dieId[%u], missionId[%u], executeId[%llu]",
                               deviceId, static_cast<u32>(ccuTaskParam.dieId), static_cast<u32>(ccuTaskParam.missionId),
                               ccuTaskParam.executeId);
    }
    const uint16_t currIns = missionContext.GetCurrentIns();
    auto           rep     = ctx->GetRepByInstrId(currIns);
    if (rep == nullptr) {
        HCCL_WARNING("[CcuErrorHandler][%s] cannot find REP from current CcuContext, instrId[%u]", __func__, currIns);
        return;
    }

    // 处理Rep为FUNC_BLOCK的场景
    while (rep->Type() == CcuRepType::FUNC_BLOCK) {
        auto blockRep = static_pointer_cast<CcuRepBlock>(rep);
        rep           = blockRep->GetRepByInstrId(currIns);
        if (rep == nullptr) {
            THROW<CcuApiException>("Failed to find REP from FuncBlock, instrId[%u], FuncBlock[%s]", currIns,
                                   blockRep->GetLabel().c_str());
        }
    }

    // 分类处理Rep, 返回异常信息
    ErrorInfoBase baseInfo{deviceId, ccuTaskParam.dieId, ccuTaskParam.missionId, currIns, missionStatus};
    GenStatusInfo(baseInfo, errorInfo);
    if (rep->Type() == CcuRepType::LOOPGROUP) {
        // 处理LoopGroup
        GenErrorInfoLoopGroup(baseInfo, rep, *ctx, errorInfo);
    } else {
        // 处理可直接解析的Rep
        GenErrorInfoByRepType(baseInfo, rep, errorInfo);
    }
}

static string StatusCode2Str(uint8_t highPart, uint8_t lowPart)
{
    HCCL_INFO("Mission Status Code: highPart[0x%02x], lowPart[0x%02x]", highPart, lowPart);
    const auto status = MISSION_STATUS_MAP.find(highPart);
    if (status == MISSION_STATUS_MAP.end()) {
        return "Unknown Status";
    }
    stringstream result;
    result << status->second;

    const auto lowMap = MISSION_SUB_STATUS_MAP.find(highPart);
    if (lowMap == MISSION_SUB_STATUS_MAP.end()) {
        return result.str();
    }

    const auto subStatus = lowMap->second.find(lowPart);
    const string subStatusMsg = subStatus == lowMap->second.end() ? "Unknown Status" : subStatus->second;
    result << ", " << subStatusMsg;
    return result.str();
}

void CcuErrorHandler::GenStatusInfo(const ErrorInfoBase &baseInfo, vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type = CcuErrorType::MISSION;
    errorMsg.SetBaseInfo(CcuRepType::BASE, baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const uint8_t highPart  = (baseInfo.status >> 8) & 0xFF; // 高8位
    const uint8_t lowPart   = baseInfo.status & 0xFF;        // 低8位
    const string  statusMsg = StatusCode2Str(highPart, lowPart);
    const auto    sRet
        = strncpy_s(errorMsg.msg.mission.missionError, MISSION_STATUS_MSG_LEN, statusMsg.c_str(), statusMsg.length());
    if (sRet != EOK) {
        HCCL_ERROR("[CcuErrorHandler][%s] strcpy failed, statusMsg: %s.", __func__, statusMsg.c_str());
    }

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLoopGroup(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                            CcuRepContext &ctx, vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::LOOP_GROUP;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto  rep              = static_pointer_cast<CcuRepLoopGroup>(repBase);
    const auto  startLoopInstrId = rep->GetStartLoopInstrId();
    LoopGroupXn loopGroupXn{};
    loopGroupXn.value                     = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->parallelParam.Id());
    errorMsg.msg.loopGroup.startLoopInsId = startLoopInstrId;
    errorMsg.msg.loopGroup.loopInsCnt     = static_cast<uint16_t>(loopGroupXn.loopInsCnt);
    errorMsg.msg.loopGroup.expandOffset   = static_cast<uint16_t>(loopGroupXn.expandOffset);
    errorMsg.msg.loopGroup.expandCnt      = static_cast<uint16_t>(loopGroupXn.expandCnt);

    errorInfo.push_back(errorMsg);

    // 处理loop
    for (uint16_t i = 0; i < loopGroupXn.loopInsCnt; ++i) {
        uint16_t      loopInsId = startLoopInstrId + i;
        ErrorInfoBase loopErrInfoBase{baseInfo.deviceId, baseInfo.dieId, baseInfo.missionId, loopInsId,
                                      baseInfo.status};
        GenErrorInfoLoop(loopErrInfoBase, ctx, errorInfo);
    }
}

void CcuErrorHandler::GenErrorInfoLoop(const ErrorInfoBase &baseInfo, CcuRepContext &ctx,
                                       vector<CcuErrorInfo> &errorInfo)
{
    // 找LoopRep
    auto repBase = ctx.GetRepByInstrId(baseInfo.currentInsId);
    if (repBase == nullptr || repBase->Type() != CcuRepType::LOOP) {
        THROW<CcuApiException>("Failed to find Loop REP from CcuContext, instrId[%u]", baseInfo.currentInsId);
    }
    const auto rep = static_pointer_cast<CcuRepLoop>(repBase);

    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::LOOP;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    LoopXm loopXm{};
    loopXm.value                     = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->loopParam.Id());
    const auto ccuLoopContext        = GetCcuLoopContext(baseInfo.deviceId, baseInfo.dieId, loopXm.loopCtxId);
    errorMsg.msg.loop.startInstrId   = rep->loopBlock->StartInstrId();
    errorMsg.msg.loop.endInstrId     = rep->loopBlock->StartInstrId() + rep->loopBlock->InstrCount() - 1;
    errorMsg.msg.loop.loopCnt        = static_cast<uint16_t>(loopXm.loopCnt);
    errorMsg.msg.loop.loopCurrentCnt = ccuLoopContext.GetCurrentCnt();
    errorMsg.msg.loop.addrStride     = ccuLoopContext.GetAddrStride();

    errorInfo.push_back(errorMsg);

    // 解析Loop内的异常Rep
    const auto loopCurrentIns = ccuLoopContext.GetCurrentIns();
    auto       inLoopExRep    = rep->loopBlock->GetRepByInstrId(loopCurrentIns);
    if (inLoopExRep == nullptr) {
        THROW<CcuApiException>("Failed to find REP from Loop, instrId[%u], Loop[%s]",
                loopCurrentIns, rep->GetLabel().c_str());
    }
    ErrorInfoBase loopErrBase{baseInfo.deviceId, baseInfo.dieId, baseInfo.missionId, loopCurrentIns, baseInfo.status};
    GenErrorInfoByRepType(loopErrBase, inLoopExRep, errorInfo);
}

void CcuErrorHandler::GenErrorInfoByRepType(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                            vector<CcuErrorInfo> &errorInfo)
{
    static const set<CcuRepType> exceptionType{
        CcuRepType::BASE,       CcuRepType::BLOCK,          CcuRepType::NOP,  CcuRepType::FUNC_CALL,
        CcuRepType::FUNC_BLOCK, CcuRepType::LOOP_CALL,      CcuRepType::LOOP, CcuRepType::LOOPGROUP,
        CcuRepType::LOOP_BLOCK, CcuRepType::LOOPGROUP_BLOCK};
    if (exceptionType.count(repBase->Type()) != 0) {
        THROW<CcuApiException>("Rep type error, RepType[%u]", static_cast<uint32_t>(repBase->Type()));
    }

    using GenErrorInfoFunc = void (*)(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                                       vector<CcuErrorInfo> &errorInfo);
    static const map<CcuRepType, GenErrorInfoFunc> handlerMap {
        // WAIT_SIGNAL
        {CcuRepType::LOC_POST_SEM, &CcuErrorHandler::GenErrorInfoLocPostSem},
        {CcuRepType::LOC_WAIT_SEM, &CcuErrorHandler::GenErrorInfoLocWaitSem},
        {CcuRepType::REM_POST_SEM, &CcuErrorHandler::GenErrorInfoRemPostSem},
        {CcuRepType::REM_WAIT_SEM, &CcuErrorHandler::GenErrorInfoRemWaitSem},
        {CcuRepType::REM_POST_VAR, &CcuErrorHandler::GenErrorInfoRemPostVar},
        {CcuRepType::REM_WAIT_GROUP, &CcuErrorHandler::GenErrorInfoRemWaitGroup},
        {CcuRepType::POST_SHARED_VAR, &CcuErrorHandler::GenErrorInfoPostSharedVar},
        {CcuRepType::POST_SHARED_SEM, &CcuErrorHandler::GenErrorInfoPostSharedSem},
        // TRANS_MEM
        {CcuRepType::READ, &CcuErrorHandler::GenErrorInfoRead},
        {CcuRepType::WRITE, &CcuErrorHandler::GenErrorInfoWrite},
        {CcuRepType::LOCAL_CPY, &CcuErrorHandler::GenErrorInfoLocalCpy},
        {CcuRepType::LOCAL_REDUCE, &CcuErrorHandler::GenErrorInfoLocalReduce},
        // BUF_TRANS_MEM
        {CcuRepType::BUF_READ, &CcuErrorHandler::GenErrorInfoBufRead},
        {CcuRepType::BUF_WRITE, &CcuErrorHandler::GenErrorInfoBufWrite},
        {CcuRepType::BUF_LOC_READ, &CcuErrorHandler::GenErrorInfoBufLocRead},
        {CcuRepType::BUF_LOC_WRITE, &CcuErrorHandler::GenErrorInfoBufLocWrite},
        // BUF_REDUCE
        {CcuRepType::BUF_REDUCE, &CcuErrorHandler::GenErrorInfoBufReduce}
    };
    const auto funcIt = handlerMap.find(repBase->Type());
    if (funcIt == handlerMap.end()) {
        // DEFAULT, chip error
        GenErrorInfoDefault(baseInfo, repBase, errorInfo);
    } else {
        (funcIt->second)(baseInfo, repBase, errorInfo);
    }
}

void CcuErrorHandler::GenErrorInfoDefault(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                          vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::DEFAULT;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);
    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocPostSem(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                             vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepLocPostSem>(repBase);
    errorMsg.msg.waitSignal.signalId    = rep->sem.Id();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->sem.Id());
    errorMsg.msg.waitSignal.signalMask  = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocWaitSem(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                             vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepLocWaitSem>(repBase);
    errorMsg.msg.waitSignal.signalId    = rep->sem.Id();
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, rep->sem.Id());
    errorMsg.msg.waitSignal.signalMask  = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemPostSem(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                             vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                     = static_pointer_cast<CcuRepRemPostSem>(repBase);
    errorMsg.msg.waitSignal.signalId   = rep->transport.GetRmtCntCkeByIndex(rep->semIndex);
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    (void)memset_s(errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
                   sizeof(errorMsg.msg.waitSignal.channelId));
    errorMsg.msg.waitSignal.channelId[0] = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemWaitSem(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                             vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                     = static_pointer_cast<CcuRepRemWaitSem>(repBase);
    errorMsg.msg.waitSignal.signalId    = rep->transport.GetLocCntCkeByIndex(rep->semIndex);
    errorMsg.msg.waitSignal.signalValue = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, errorMsg.msg.waitSignal.signalId);
    errorMsg.msg.waitSignal.signalMask  = rep->mask;
    (void)memset_s(errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
                   sizeof(errorMsg.msg.waitSignal.channelId));
    errorMsg.msg.waitSignal.channelId[0] = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemPostVar(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                             vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                     = static_pointer_cast<CcuRepRemPostVar>(repBase);
    errorMsg.msg.waitSignal.signalId   = rep->transport.GetRmtCntCkeByIndex(rep->semIndex);
    errorMsg.msg.waitSignal.signalMask = rep->mask;
    (void)memset_s(errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
                   sizeof(errorMsg.msg.waitSignal.channelId));
    errorMsg.msg.waitSignal.channelId[0] = rep->transport.GetChannelId();
    errorMsg.msg.waitSignal.paramId      = rep->transport.GetRmtXnByIndex(rep->paramIndex);
    errorMsg.msg.waitSignal.paramValue   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->param.Id());

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRemWaitGroup(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                               vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                           = static_pointer_cast<CcuRepWaitGroup>(repBase);
    errorMsg.msg.waitSignal.signalId         = rep->transportGroup.GetCntCkeId(rep->semIndex);
    errorMsg.msg.waitSignal.signalValue      = GetCcuCKEValue(baseInfo.deviceId, baseInfo.dieId, errorMsg.msg.waitSignal.signalId);
    errorMsg.msg.waitSignal.signalMask       = rep->mask;
    const vector<CcuTransport*> &transports  = rep->transportGroup.GetTransports();
    (void)memset_s(errorMsg.msg.waitSignal.channelId, sizeof(errorMsg.msg.waitSignal.channelId), 0xFF,
                   sizeof(errorMsg.msg.waitSignal.channelId));
    for (uint32_t i = 0; i < transports.size() && i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        errorMsg.msg.waitSignal.channelId[i] = transports[i]->GetChannelId();
    }

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoPostSharedVar(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                                vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepPostSharedVar>(repBase);
    errorMsg.msg.waitSignal.signalId    = rep->sem.Id();
    errorMsg.msg.waitSignal.signalMask  = rep->mask;
    errorMsg.msg.waitSignal.paramId     = rep->dstVar.Id();
    errorMsg.msg.waitSignal.paramValue  = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->srcVar.Id());

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoPostSharedSem(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                                vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::WAIT_SIGNAL;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepPostSharedSem>(repBase);
    errorMsg.msg.waitSignal.signalId    = rep->sem.Id();
    errorMsg.msg.waitSignal.signalMask  = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoRead(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                       vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                   = static_pointer_cast<CcuRepRead>(repBase);
    errorMsg.msg.transMem.locAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.addr.Id());
    errorMsg.msg.transMem.locToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.token.Id());
    errorMsg.msg.transMem.rmtAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.addr.Id());
    errorMsg.msg.transMem.rmtToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.token.Id());
    errorMsg.msg.transMem.signalId   = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;
    errorMsg.msg.transMem.channelId  = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoWrite(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                        vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                   = static_pointer_cast<CcuRepWrite>(repBase);
    errorMsg.msg.transMem.locAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.addr.Id());
    errorMsg.msg.transMem.locToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->loc.token.Id());
    errorMsg.msg.transMem.rmtAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.addr.Id());
    errorMsg.msg.transMem.rmtToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->rem.token.Id());
    errorMsg.msg.transMem.signalId   = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;
    errorMsg.msg.transMem.channelId  = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocalCpy(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                           vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                   = static_pointer_cast<CcuRepLocCpy>(repBase);
    errorMsg.msg.transMem.locAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.transMem.locToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.transMem.rmtAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.transMem.rmtToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.transMem.signalId   = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoLocalReduce(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                              vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                   = static_pointer_cast<CcuRepLocCpy>(repBase);
    errorMsg.msg.transMem.locAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.transMem.locToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.transMem.rmtAddr    = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.transMem.rmtToken   = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.transMem.signalId   = rep->sem.Id();
    errorMsg.msg.transMem.signalMask = rep->mask;
    errorMsg.msg.transMem.opType     = rep->opType;
    errorMsg.msg.transMem.dataType   = rep->dataType;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufRead(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                          vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                    = static_pointer_cast<CcuRepBufRead>(repBase);
    errorMsg.msg.bufTransMem.bufId    = rep->dst.Id();
    errorMsg.msg.bufTransMem.addr     = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.bufTransMem.token    = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.bufTransMem.signalId = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;
    errorMsg.msg.bufTransMem.channelId  = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufWrite(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                           vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepBufWrite>(repBase);
    errorMsg.msg.bufTransMem.bufId      = rep->src.Id();
    errorMsg.msg.bufTransMem.addr       = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.bufTransMem.token      = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.bufTransMem.signalId   = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;
    errorMsg.msg.bufTransMem.channelId  = rep->transport.GetChannelId();

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufLocRead(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                             vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepBufLocRead>(repBase);
    errorMsg.msg.bufTransMem.bufId      = rep->dst.Id();
    errorMsg.msg.bufTransMem.addr       = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->src.addr.Id());
    errorMsg.msg.bufTransMem.token      = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->src.token.Id());
    errorMsg.msg.bufTransMem.signalId   = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufLocWrite(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                              vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::BUF_TRANS_MEM;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                      = static_pointer_cast<CcuRepBufLocWrite>(repBase);
    errorMsg.msg.bufTransMem.bufId      = rep->src.Id();
    errorMsg.msg.bufTransMem.addr       = GetCcuGSAValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.addr.Id());
    errorMsg.msg.bufTransMem.token      = GetCcuXnValue(baseInfo.deviceId, baseInfo.dieId, rep->dst.token.Id());
    errorMsg.msg.bufTransMem.signalId   = rep->sem.Id();
    errorMsg.msg.bufTransMem.signalMask = rep->mask;

    errorInfo.push_back(errorMsg);
}

void CcuErrorHandler::GenErrorInfoBufReduce(const ErrorInfoBase &baseInfo, shared_ptr<CcuRepBase> repBase,
                                            vector<CcuErrorInfo> &errorInfo)
{
    CcuErrorInfo errorMsg{};
    errorMsg.type    = CcuErrorType::BUF_REDUCE;
    errorMsg.SetBaseInfo(repBase->Type(), baseInfo.dieId, baseInfo.missionId, baseInfo.currentInsId);

    const auto rep                        = static_pointer_cast<CcuRepBufReduce>(repBase);
    errorMsg.msg.bufReduce.count          = rep->count;
    errorMsg.msg.bufReduce.dataType       = rep->dataType;
    errorMsg.msg.bufReduce.outputDataType = rep->outputDataType;
    errorMsg.msg.bufReduce.opType         = rep->opType;
    errorMsg.msg.bufReduce.signalId       = rep->sem.Id();
    errorMsg.msg.bufReduce.signalMask     = rep->mask;
    errorMsg.msg.bufReduce.xnIdLength     = rep->xnIdLength_.Id();
    const auto &buffs                     = rep->mem;
    (void)memset_s(errorMsg.msg.bufReduce.bufIds, sizeof(errorMsg.msg.bufReduce.bufIds), 0xFF,
                   sizeof(errorMsg.msg.bufReduce.bufIds));
    for (uint32_t i = 0; i < buffs.size() && i < BUF_REDUCE_ID_SIZE; ++i) {
        errorMsg.msg.bufReduce.bufIds[i] = buffs[i].Id();
    }

    errorInfo.push_back(errorMsg);
}

CcuMissionContext CcuErrorHandler::GetCcuMissionContext(int32_t deviceId, uint32_t dieId, uint32_t missionId)
{
    HRaInfo                      info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByIndex(deviceId));
    struct CustomChannelInfoIn  inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_GET_MISSION_CTX;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.offsetStartIdx              = missionId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个MissionContext
    inBuff.data.dataInfo.dataLen       = sizeof(CcuMissionContext) * inBuff.data.dataInfo.dataArraySize;

    HrtRaCustomChannel(info, &inBuff, &outBuff);

    CcuMissionContext missionCtx{};
    (void)memcpy_s(&missionCtx, sizeof(missionCtx), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return missionCtx;
}

CcuLoopContext CcuErrorHandler::GetCcuLoopContext(int32_t deviceId, uint32_t dieId, uint32_t loopCtxId)
{
    HRaInfo                      info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByIndex(deviceId));
    struct CustomChannelInfoIn  inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_GET_LOOP_CTX;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.offsetStartIdx              = loopCtxId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个LoopContext
    inBuff.data.dataInfo.dataLen       = sizeof(CcuLoopContext) * inBuff.data.dataInfo.dataArraySize;

    HrtRaCustomChannel(info, &inBuff, &outBuff);

    CcuLoopContext loopCtx{};
    (void)memcpy_s(&loopCtx, sizeof(loopCtx), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return loopCtx;
}

uint64_t CcuErrorHandler::GetCcuXnValue(int32_t deviceId, uint32_t dieId, uint32_t xnId)
{
    HRaInfo                      info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByIndex(deviceId));
    struct CustomChannelInfoIn  inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_GET_XN;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.offsetStartIdx              = xnId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个Xn
    inBuff.data.dataInfo.dataLen       = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HrtRaCustomChannel(info, &inBuff, &outBuff);

    uint64_t xnVal{0};
    (void)memcpy_s(&xnVal, sizeof(xnVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return xnVal;
}

uint64_t CcuErrorHandler::GetCcuGSAValue(int32_t deviceId, uint32_t dieId, uint32_t gsaId)
{
    HRaInfo                      info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByIndex(deviceId));
    struct CustomChannelInfoIn  inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_GET_GSA;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.offsetStartIdx              = gsaId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个GSA
    inBuff.data.dataInfo.dataLen       = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HrtRaCustomChannel(info, &inBuff, &outBuff);

    uint64_t gsaVal{0};
    (void)memcpy_s(&gsaVal, sizeof(gsaVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return gsaVal;
}

uint16_t CcuErrorHandler::GetCcuCKEValue(int32_t deviceId, uint32_t dieId, uint32_t ckeId)
{
    HRaInfo                      info(HrtNetworkMode::HDC, HrtGetDevicePhyIdByIndex(deviceId));
    struct CustomChannelInfoIn  inBuff;
    struct CustomChannelInfoOut outBuff;

    inBuff.op                          = CcuOpcodeType::CCU_U_OP_GET_CKE;
    inBuff.data.dataInfo.udieIdx       = dieId;
    inBuff.offsetStartIdx              = ckeId;
    inBuff.data.dataInfo.dataArraySize = 1; // 读1个CKE
    inBuff.data.dataInfo.dataLen       = sizeof(uint64_t) * inBuff.data.dataInfo.dataArraySize;

    HrtRaCustomChannel(info, &inBuff, &outBuff);

    uint64_t ckeVal{0};
    (void)memcpy_s(&ckeVal, sizeof(ckeVal), outBuff.data.dataInfo.dataArray, inBuff.data.dataInfo.dataLen);
    return static_cast<uint16_t>(ckeVal);
}

} // namespace Hccl