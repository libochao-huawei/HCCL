/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu instruction transform to checker task
 * Author: huangweihao
 * Create: 2025-06-19
 */

#include "ccu_task_transform.h"
#include "transport_utils.h"
#include "type_conversion.h"
#include "mem_layout.h"
#include "task_graph_generator.h"
#include "ccu_all_rank_param_recorder.h"
#include "ccu_task_common.h"
#include "rank_info_recorder.h"

using namespace checker;

namespace {
constexpr uint16_t LOAD_TYPE   = 0x0;
constexpr uint16_t CTRL_TYPE   = 0x1;
constexpr uint16_t TRANS_TYPE  = 0x2;
constexpr uint16_t REDUCE_TYPE = 0x3;

constexpr uint16_t LOADSQEARGSTOGSA_CODE = 0x0;
constexpr uint16_t LOADSQEARGSTOXN_CODE  = 0x1;
constexpr uint16_t LOADIMDTOGSA_CODE     = 0x2;
constexpr uint16_t LOADIMDTOXN_CODE      = 0x3;
constexpr uint16_t LOADGSAXN_CODE        = 0x4;
constexpr uint16_t LOADGSAGSA_CODE       = 0x5;
constexpr uint16_t LOADXX_CODE           = 0x6;

constexpr uint16_t LOOP_CODE      = 0x0;
constexpr uint16_t LOOPGROUP_CODE = 0x1;
constexpr uint16_t SETCKE_CODE    = 0x2;
constexpr uint16_t CLEARCKE_CODE  = 0x4;
constexpr uint16_t JMP_CODE       = 0x5;

constexpr uint16_t TRANSLOCMEMTOLOCMS_CODE  = 0x0;
constexpr uint16_t TRANSRMTMEMTOLOCMS_CODE  = 0x1;
constexpr uint16_t TRANSLOCMSTOLOCMEM_CODE  = 0x2;
constexpr uint16_t TRANSLOCMSTORMTMEM_CODE  = 0x3;
constexpr uint16_t TRANSRMTMSTOLOCMEM_CODE  = 0x4;
constexpr uint16_t TRANSLOCMSTOLOCMS_CODE   = 0x5;
constexpr uint16_t TRANSRMTMSTOLOCMS_CODE   = 0x6;
constexpr uint16_t TRANSLOCMSTORMTMS_CODE   = 0x7;
constexpr uint16_t TRANSRMTMEMTOLOCMEM_CODE = 0x8;
constexpr uint16_t TRANSLOCMEMTORMTMEM_CODE = 0x9;
constexpr uint16_t TRANSLOCMEMTOLOCMEM_CODE = 0xa;
constexpr uint16_t SYNCCKE_CODE             = 0xb;
constexpr uint16_t SYNCGSA_CODE             = 0xc;
constexpr uint16_t SYNCXN_CODE              = 0xd;

constexpr uint16_t ADD_CODE = 0x0;
constexpr uint16_t MAX_CODE = 0x1;
constexpr uint16_t MIN_CODE = 0x2;

constexpr uint64_t UB_MAX_SIZE = 256 * 1024 * 1024;
} // namespace

namespace Hccl {
static std::map<uint16_t, uint16_t> ccuReduceTypeMap = {
    {10, CcuRep::CCU_REDUCE_SUM},
    { 9, CcuRep::CCU_REDUCE_MIN},
    { 8, CcuRep::CCU_REDUCE_MAX}
};

HcclResult IsLoopGroupParamNull(LoopGroupParam* loopGroupParam)
{
    if (loopGroupParam != nullptr) {
        HCCL_ERROR("this ins do not support loop");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransformLoadSqeArgsToGSAInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t sqeArgsId = instr->v1.loadSqeArgsToGSA.sqeArgsId;
    uint16_t gsaId = instr->v1.loadSqeArgsToGSA.gsaId;
    uint64_t argVal = 0;
    curCcuTask->GetSqe(queId, sqeArgsId, argVal);

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsaId, argVal));
    HCCL_INFO("Load SqeArg[%u](%llu) to GSA[%u]", sqeArgsId, argVal, gsaId);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadSqeArgsToXnInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t sqeArgsId = instr->v1.loadSqeArgsToXn.sqeArgsId;
    uint16_t xnId = instr->v1.loadSqeArgsToXn.xnId;
    uint64_t argVal = 0;
    curCcuTask->GetSqe(queId, sqeArgsId, argVal);

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xnId, argVal));
    HCCL_INFO("Load SqeArg[%u](%llu) to Xn[%u]", sqeArgsId, argVal, xnId);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadImdToGSAInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t gsaId = instr->v1.loadImdToGSA.gsaId;
    uint64_t immediate = instr->v1.loadImdToGSA.immediate;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsaId, immediate));
    HCCL_INFO("Load immediate[%llu] to GSA[%u]", immediate, gsaId);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadImdToXnInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t xnId = instr->v1.loadImdToXn.xnId;
    uint64_t immediate = instr->v1.loadImdToXn.immediate;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xnId, immediate));
    HCCL_INFO("Load immediate[%llu] to Xn[%u]", immediate, xnId);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadGSAXnInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t gsAmId = instr->v1.loadGSAXn.gsAmId;
    uint16_t xnId   = instr->v1.loadGSAXn.xnId;
    uint16_t gsAdId = instr->v1.loadGSAXn.gsAdId;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint64_t      argVal1= 0;
    uint64_t      argVal2= 0;
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, gsAmId, argVal1));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, argVal2));
    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsAdId, argVal1 + argVal2));
    HCCL_INFO("Load GSA[%hu](%llu) + Xn[%hu](%llu) to GSA[%hu](%llu)", gsAmId, argVal1, xnId, argVal2, gsAdId, argVal1 + argVal2);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadGSAGSAInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t gsAmId = instr->v1.loadGSAGSA.gsAmId;
    uint16_t gsAnId = instr->v1.loadGSAGSA.gsAnId;
    uint16_t gsAdId = instr->v1.loadGSAGSA.gsAdId;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint64_t argVal1 = 0;
    uint64_t argVal2 = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, gsAmId, argVal1));
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, gsAnId, argVal2));
    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rankId, dieId, gsAdId, argVal1 + argVal2));
    HCCL_INFO("Load GSA[%hu](%llu) + GSA[%hu](%llu) to GSA[%hu](%llu)", gsAmId, argVal1, gsAnId, argVal2, gsAdId, argVal1 + argVal2);
    return HCCL_SUCCESS;
}

HcclResult TransformLoadXXInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t xmId = instr->v1.loadXX.xmId;
    uint16_t xnId = instr->v1.loadXX.xnId;
    uint16_t xdId = instr->v1.loadXX.xdId;
    uint64_t argVal1= 0;
    uint64_t argVal2= 0;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, argVal1));
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, argVal2));
    CHK_RET(AllRankParamRecorder::Global()->SetXn(rankId, dieId, xdId, argVal1 + argVal2));
    HCCL_INFO("Load Xn[%hu](%llu) + Xn[%hu](%llu) to Xn[%hu](%llu)", xmId, argVal1, xnId, argVal2, xdId, argVal1 + argVal2);
    return HCCL_SUCCESS;
}

u32 GetTopicId(checker::TaskNode* post)
{
    if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
        TaskStubLocalPostTo *postTask = dynamic_cast<TaskStubLocalPostTo *>(post->task);
        return postTask->GetTopicId();
    }
    TaskStubPost *postTask = dynamic_cast<TaskStubPost *>(post->task);
    return postTask->GetTopicId();
}

void SetTopicId(checker::TaskNode* post, u32 topicId)
{
    if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
        TaskStubLocalPostTo *postTask = dynamic_cast<TaskStubLocalPostTo *>(post->task);
        postTask->SetTopicId(topicId);
        return;
    }
    TaskStubPost *postTask = dynamic_cast<TaskStubPost *>(post->task);
    postTask->SetTopicId(topicId);
    return;
}

void GenWaitNode(TaskStubCcuGraph *curCcuTask, uint32_t queId, uint16_t waitCKEId, uint16_t waitCKEMask)
{
    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint16_t localWaitMask = 0;
    uint16_t remoteWaitMask = 0;
    TaskNode* localWaitNode = nullptr;
    TaskNode* remoteWaitNode = nullptr;

    std::set<checker::TaskNode*>& seenPosts = AllRankParamRecorder::Global()->seenPost[rankId][dieId][waitCKEId];
    std::vector<TaskNodePtr> localPosts;
    for (auto& post : seenPosts) {
        u32 topicId = GetTopicId(post);
        uint16_t curPostMask = topicId & waitCKEMask;
        if (curPostMask == 0) {
            continue;
        }

        if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            localWaitMask = localWaitMask | curPostMask;
            localPosts.push_back(post);
        } else {
            remoteWaitMask = remoteWaitMask | curPostMask;
        }
    }

    if (remoteWaitMask != 0) {
        remoteWaitNode = AddWait(rankId, queId, curCcuTask, remoteWaitMask);
    }

    if (localWaitMask != 0) {
        localWaitNode = AddLocalWait(rankId, queId, curCcuTask, localWaitMask);
        // ccu子图中，local post和local wait是多对一的关系
        for (auto &locPost : localPosts) {
            curCcuTask->localPostWaitPairs_[locPost] = localWaitNode;
        }
    }

    for (auto it = seenPosts.begin(); it != seenPosts.end();) {
        auto post = *it;
        u32 topicId = GetTopicId(post);
        uint16_t curPostMask = topicId & waitCKEMask;
        if (curPostMask == 0) {
            ++it;
            continue;
        }

        if (post->task->GetType() == TaskTypeStub::LOCAL_POST_TO) {
            AddNodeRelation(post, localWaitNode);
        } else {
            AddNodeRelation(post, remoteWaitNode);
            auto waitTask = dynamic_cast<TaskStubWait *>(remoteWaitNode->task);
            waitTask->SetRemoteRank(post->rankIdx);
        }

        topicId = topicId & (~waitCKEMask);
        SetTopicId(post, topicId);
        if (topicId == 0) {
            it = seenPosts.erase(it);
        } else {
            ++it;
        }
    }
    return;
}

HcclResult ProcessWaitMask(RankId rankId, uint32_t dieId, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    uint16_t waitCKEId, uint16_t waitCKEMask, bool& isContinue)
{
    if (waitCKEMask != 0x0000) {
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, waitCKEId, ckeValue));
        // 条件不满足，继续等待
        if ((ckeValue & waitCKEMask) != waitCKEMask) {
            isContinue = 0;
            return HCCL_SUCCESS;
        } else {
            GenWaitNode(curCcuTask, queId, waitCKEId, waitCKEMask);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ProcessSetMask(RankId rankId, uint32_t dieId, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    uint16_t setCKEId, uint16_t setCKEMask, bool invalidPost = false)
{
    if (setCKEMask != 0x0000) {
        // TODO：这边的wait queId当前没有填，后续需要关注一下
        // TODO: 这边记录的pos为微码序列中的位置
        AddLocalPost(rankId, queId, dieId, curCcuTask, setCKEId, setCKEMask, invalidPost);

        // 将对应的bit位置一
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, setCKEId, ckeValue));
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(rankId, dieId, setCKEId, ckeValue | setCKEMask));
    }

    return HCCL_SUCCESS;
}

HcclResult ClearWaitMask(RankId rankId, uint32_t dieId, uint16_t waitCKEId, uint16_t waitCKEMask)
{
    // wait匹配之后清理状态位
    uint16_t ckeValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, waitCKEId, ckeValue));
    CHK_RET(AllRankParamRecorder::Global()->SetCKE(rankId, dieId, waitCKEId, ckeValue & (~waitCKEMask)));
    return HCCL_SUCCESS;
}

HcclResult TransformSetCKEInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t clearType   = instr->v1.setCKE.clearType;
    uint16_t setCKEId    = instr->v1.setCKE.setCKEId;
    uint16_t setCKEMask  = instr->v1.setCKE.setCKEMask;
    uint16_t waitCKEId   = instr->v1.setCKE.waitCKEId;
    uint16_t waitCKEMask = instr->v1.setCKE.waitCKEMask;

    // 当前只支持clearType为1的场景
    if (clearType != 0x0001) {
        HCCL_ERROR("do not support clearType[%hu]", clearType);
        return HCCL_E_INTERNAL;
    }

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Set CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, setCKEId,
                        setCKEMask, clearType);
    return HCCL_SUCCESS;
}

// 循环中CKE需要进行自动偏移
uint16_t UpdateCKEId(uint16_t CKEId, LoopGroupParam* loopGroupParam)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return CKEId;
    }

    // 不需要进行展开，不刷新CKE Id
    if (loopGroupParam->curLoopIdx < loopGroupParam->loopGroupXn.expandOffset) {
        return CKEId;
    }

    return CKEId + loopGroupParam->curExpandCnt * loopGroupParam->loopGroupXm.ckOffset;
}

// 循环中MS需要进行自动偏移
uint16_t UpdateMSId(uint16_t MSId, LoopGroupParam* loopGroupParam)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return MSId;
    }

    // 不需要进行展开，不刷新MS Id
    if (loopGroupParam->curLoopIdx < loopGroupParam->loopGroupXn.expandOffset) {
        return MSId;
    }

    return MSId + loopGroupParam->curExpandCnt * loopGroupParam->loopGroupXm.msOffset;
}

// 更新GSA的地址
uint64_t UpdateGSAValue(uint64_t gsaAddr, LoopGroupParam* loopGroupParam)
{
    // 不在循环中，不需要刷新
    if (loopGroupParam == nullptr) {
        return gsaAddr;
    }

    LoopXm& xm = loopGroupParam->loopXms[loopGroupParam->curLoopIdx];

    if (loopGroupParam->curLoopIdx < loopGroupParam->loopGroupXn.expandOffset) {
        return gsaAddr + loopGroupParam->curLoopCnt * xm.gsaStride;
    }

    return gsaAddr + loopGroupParam->curExpandCnt * loopGroupParam->loopGroupXm.gsaOffset \
        + loopGroupParam->curLoopCnt * xm.gsaStride;
}

// TODO: 这边可能支持reduce，需要处理一下
HcclResult TransformTransLocMemToRmtMemInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t rmtGSAId       = instr->v1.transLocMemToRmtMem.rmtGSAId;
    uint16_t rmtXnId        = instr->v1.transLocMemToRmtMem.rmtXnId;
    uint16_t locGSAId       = instr->v1.transLocMemToRmtMem.locGSAId;
    uint16_t locXnId        = instr->v1.transLocMemToRmtMem.locXnId;
    uint16_t lengthXnId     = instr->v1.transLocMemToRmtMem.lengthXnId;
    uint16_t channelId      = instr->v1.transLocMemToRmtMem.channelId;
    uint16_t reduceDataType = instr->v1.transLocMemToRmtMem.reduceDataType;
    uint16_t reduceOpCode   = instr->v1.transLocMemToRmtMem.reduceOpCode;
    uint16_t clearType      = instr->v1.transLocMemToRmtMem.clearType;
    uint16_t lengthEn       = instr->v1.transLocMemToRmtMem.lengthEn;
    uint16_t reduceEn       = instr->v1.transLocMemToRmtMem.reduceEn;
    uint16_t setCKEId       = instr->v1.transLocMemToRmtMem.setCKEId;
    uint16_t setCKEMask     = instr->v1.transLocMemToRmtMem.setCKEMask;
    uint16_t waitCKEId      = instr->v1.transLocMemToRmtMem.waitCKEId;
    uint16_t waitCKEMask    = instr->v1.transLocMemToRmtMem.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    // 获取远端rankId与dieId
    if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    uint64_t locAddr = 0x0;
    uint64_t rmtAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, locAddr);
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtAddr);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    CHK_PRT_RET(len > UB_MAX_SIZE,
        HCCL_ERROR("The size of data transfer is more than max transport size of UB."), HCCL_E_INTERNAL);
    checker::DataSlice srcSlice;
    checker::DataSlice dstSlice;
    RankId rId;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)locAddr, len, srcSlice));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)rmtAddr, len, dstSlice, &rId));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    // 说明transportId写错了
    if (rId != rmtRankId) {
        HCCL_ERROR("remoteRankId calculated by rmtAddr and channelId is not equal,"
            "remoteId is %d, but the addr is in rank %d", rmtRankId, rId);
        return HCCL_E_INTERNAL;
    }
    checker::CheckerReduceOp checkerReduceOp;
    checker::CheckerDataType checkerDataType;
    if (reduceEn) {
        checkerReduceOp = g_ReduceOp2CheckerReduceOp_ccu[reduceOpCode];
        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(reduceDataType, ccuReduceTypeMap[reduceOpCode], hcclDataType));
        checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
    }

    if (rmtRankId != rankId) {
        if (reduceEn) {
            AddWriteReduce(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
        } else {
            AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);
        }
    } else {
        if (reduceEn) {
            AddLocalReduce(rankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
        } else {
            AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);
        }
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));
    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO(
        "Wait CKE[%u:%04x], Trans LocMem[%u:%u] To RmtMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]",
        waitCKEId, waitCKEMask, locGSAId, locXnId, rmtGSAId, rmtXnId, lengthXnId, channelId, setCKEId, setCKEMask,
        clearType, lengthEn, reduceDataType, reduceOpCode, reduceEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMemToLocMemInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t dstGSAId    = instr->v1.transLocMemToLocMem.dstGSAId;
    uint16_t dstXnId     = instr->v1.transLocMemToLocMem.dstXnId;
    uint16_t srcGSAId    = instr->v1.transLocMemToLocMem.srcGSAId;
    uint16_t srcXnId     = instr->v1.transLocMemToLocMem.srcXnId;
    uint16_t lengthXnId  = instr->v1.transLocMemToLocMem.lengthXnId;
    uint16_t channelId   = instr->v1.transLocMemToLocMem.channelId;
    uint16_t clearType   = instr->v1.transLocMemToLocMem.clearType;
    uint16_t lengthEn    = instr->v1.transLocMemToLocMem.lengthEn;
    uint16_t setCKEId    = instr->v1.transLocMemToLocMem.setCKEId;
    uint16_t setCKEMask  = instr->v1.transLocMemToLocMem.setCKEMask;
    uint16_t waitCKEId   = instr->v1.transLocMemToLocMem.waitCKEId;
    uint16_t waitCKEMask = instr->v1.transLocMemToLocMem.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t srcAddr = 0x0;
    uint64_t dstAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, srcGSAId, srcAddr);
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, dstGSAId, dstAddr);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    CHK_PRT_RET(len > UB_MAX_SIZE,
        HCCL_ERROR("The size of data transfer is more than max transport size of UB."), HCCL_E_INTERNAL);

    checker::DataSlice srcSlice;
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)srcAddr, len, srcSlice));
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)dstAddr, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO(
        "Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId, waitCKEMask, srcGSAId, srcXnId, dstGSAId, dstXnId, lengthXnId, channelId, setCKEId, setCKEMask,
        clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformSyncCKEInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t rmtCKEId    = instr->v1.syncCKE.rmtCKEId;
    uint16_t locCKEId    = instr->v1.syncCKE.locCKEId;
    uint16_t locCKEMask  = instr->v1.syncCKE.locCKEMask;
    uint16_t channelId   = instr->v1.syncCKE.channelId;
    uint16_t clearType   = instr->v1.syncCKE.clearType;
    uint16_t setCKEId    = instr->v1.syncCKE.setCKEId;
    uint16_t setCKEMask  = instr->v1.syncCKE.setCKEMask;
    uint16_t waitCKEId   = instr->v1.syncCKE.waitCKEId;
    uint16_t waitCKEMask = instr->v1.syncCKE.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    // 获取远端rankId与dieId
    if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

    if (locCKEMask != 0x0000) {
        AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, rmtCKEId, locCKEMask);

        // 将对应的bit位置一
        uint16_t ckeValue = 0;
        uint16_t localCkeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rankId, dieId, locCKEId, localCkeValue));
        localCkeValue = localCkeValue & locCKEMask;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, rmtCKEId, ckeValue));
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, rmtCKEId, ckeValue | localCkeValue));
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Sync LocCKE[%u:%04x] To rmtCKE[%u:%04x] Use Channel[%u], Set "
                        "CKE[%u:%04x], clearType[%u]",
                        waitCKEId, waitCKEMask, locCKEId, locCKEMask, rmtCKEId, locCKEMask, channelId, setCKEId,
                        setCKEMask, clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformSyncXnInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t rmtXnId       = instr->v1.syncXn.rmtXnId;
    uint16_t locXnId       = instr->v1.syncXn.locXnId;
    uint16_t channelId     = instr->v1.syncXn.channelId;
    uint16_t setRmtCKEId   = instr->v1.syncXn.setRmtCKEId;
    uint16_t setRmtCKEMask = instr->v1.syncXn.setRmtCKEMask;
    uint16_t clearType     = instr->v1.syncXn.clearType;
    uint16_t setCKEId      = instr->v1.syncXn.setCKEId;
    uint16_t setCKEMask    = instr->v1.syncXn.setCKEMask;
    uint16_t waitCKEId     = instr->v1.syncXn.waitCKEId;
    uint16_t waitCKEMask   = instr->v1.syncXn.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    // 获取远端rankId与dieId
    if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

    uint64_t localXnValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, locXnId, localXnValue));
    CHK_RET(AllRankParamRecorder::Global()->SetXn(rmtRankId, rmtDieId, rmtXnId, localXnValue));

    if (setRmtCKEMask != 0x0000) {
        AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, setRmtCKEId, setRmtCKEMask);

        // 将对应的bit位置一
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue));
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue | setRmtCKEMask));
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Sync locXnId[%u] To rmtXnId[%u] Use Channel[%u], Set rmtCKE[%u:%04x], Set "
                        "CKE[%u:%04x], clearType[%u]",
                        waitCKEId, waitCKEMask, locXnId, rmtXnId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId,
                        setCKEMask, clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformJumpInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t conditionXnId = instr->v1.jmp.conditionXnId;
    uint64_t expectData    = instr->v1.jmp.expectData;
    uint16_t dstInstrXnId  = instr->v1.jmp.dstInstrXnId;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint64_t conditionValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, conditionXnId, conditionValue));
    if (conditionValue != expectData) {
        uint64_t nextInsIdx = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, dstInstrXnId, nextInsIdx));
        curCcuTask->microCodePosInQue[queId] = nextInsIdx - curCcuTask->startInstrIdInQue[queId];
    }

    HCCL_INFO("When conditionXn[%u][%llu] not equal to expectData[%llu], Jump To InstrIdXn[%u]", conditionXnId,
        conditionValue, expectData, dstInstrXnId);
    return HCCL_SUCCESS;
}

HcclResult GenSliceFromMs(uint16_t msId, uint64_t len, checker::DataSlice& slice)
{
    constexpr u32 MS_LEN = 4 * 1024;

    slice.SetBufferType(checker::BufferType::MS);
    slice.SetOffset(msId * MS_LEN);
    slice.SetSize(len);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMemToLocMSInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t locMSId     = UpdateMSId(instr->v1.transLocMemToLocMS.locMSId, loopGroupParam);
    uint16_t locGSAId    = instr->v1.transLocMemToLocMS.locGSAId;
    uint16_t locXnId     = instr->v1.transLocMemToLocMS.locXnId;
    uint16_t lengthXnId  = instr->v1.transLocMemToLocMS.lengthXnId;
    uint16_t channelId   = instr->v1.transLocMemToLocMS.channelId;
    uint16_t clearType   = instr->v1.transLocMemToLocMS.clearType;
    uint16_t lengthEn    = instr->v1.transLocMemToLocMS.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transLocMemToLocMS.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transLocMemToLocMS.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transLocMemToLocMS.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transLocMemToLocMS.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t locMemAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, locMemAddr);
    locMemAddr = UpdateGSAValue(locMemAddr, loopGroupParam);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    checker::DataSlice srcSlice;
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)locMemAddr, len, srcSlice));
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
               "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, locGSAId, locXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId,
               channelId, setCKEId, setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToLocMemInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t locGSAId    = instr->v1.transLocMSToLocMem.locGSAId;
    uint16_t locXnId     = instr->v1.transLocMSToLocMem.locXnId;
    uint16_t locMSId     = UpdateMSId(instr->v1.transLocMSToLocMem.locMSId, loopGroupParam);
    uint16_t lengthXnId  = instr->v1.transLocMSToLocMem.lengthXnId;
    uint16_t channelId   = instr->v1.transLocMSToLocMem.channelId;
    uint16_t clearType   = instr->v1.transLocMSToLocMem.clearType;
    uint16_t lengthEn    = instr->v1.transLocMSToLocMem.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transLocMSToLocMem.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transLocMSToLocMem.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transLocMSToLocMem.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transLocMSToLocMem.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    checker::DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));

    uint64_t locMemAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, locMemAddr);
    locMemAddr = UpdateGSAValue(locMemAddr, loopGroupParam);
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)locMemAddr, len, dstSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
               "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, locGSAId, locXnId, lengthXnId,
               channelId, setCKEId, setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToLocMSInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t dstMSId     = UpdateMSId(instr->v1.transLocMSToLocMS.dstMSId, loopGroupParam);
    uint16_t srcMSId     = UpdateMSId(instr->v1.transLocMSToLocMS.srcMSId, loopGroupParam);
    uint16_t lengthXnId  = instr->v1.transLocMSToLocMS.lengthXnId;
    uint16_t channelId   = instr->v1.transLocMSToLocMS.channelId;
    uint16_t clearType   = instr->v1.transLocMSToLocMS.clearType;
    uint16_t lengthEn    = instr->v1.transLocMSToLocMS.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transLocMSToLocMS.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transLocMSToLocMS.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transLocMSToLocMS.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transLocMSToLocMS.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    checker::DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(srcMSId, len, srcSlice));
    checker::DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(dstMSId, len, srcSlice));

    AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], "
               "Set CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, srcMSId / 0x8000, srcMSId % 0x8000, dstMSId / 0x8000, dstMSId % 0x8000,
               lengthXnId, channelId, setCKEId, setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToRmtMemInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t rmtGSAId    = instr->v1.transLocMSToRmtMem.rmtGSAId;
    uint16_t rmtXnId     = instr->v1.transLocMSToRmtMem.rmtXnId;
    uint16_t locMSId     = UpdateMSId(instr->v1.transLocMSToRmtMem.locMSId, loopGroupParam);
    uint16_t lengthXnId  = instr->v1.transLocMSToRmtMem.lengthXnId;
    uint16_t channelId   = instr->v1.transLocMSToRmtMem.channelId;
    uint16_t clearType   = instr->v1.transLocMSToRmtMem.clearType;
    uint16_t lengthEn    = instr->v1.transLocMSToRmtMem.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transLocMSToRmtMem.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transLocMSToRmtMem.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transLocMSToRmtMem.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transLocMSToRmtMem.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    checker::DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));

    uint64_t rmtMemAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtMemAddr);
    rmtMemAddr = UpdateGSAValue(rmtMemAddr, loopGroupParam);
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)rmtMemAddr, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
               "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtGSAId, rmtXnId, lengthXnId,
               channelId, setCKEId, setCKEMask, clearType, lengthEn);

    return HCCL_SUCCESS;
}

HcclResult TransformTransLocMSToRmtMSInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t rmtMSId       = UpdateMSId(instr->v1.transLocMSToRmtMS.rmtMSId, loopGroupParam);
    uint16_t locMSId       = UpdateMSId(instr->v1.transLocMSToRmtMS.locMSId, loopGroupParam);
    uint16_t lengthXnId    = instr->v1.transLocMSToRmtMS.lengthXnId;
    uint16_t channelId     = instr->v1.transLocMSToRmtMS.channelId;
    uint16_t setRmtCKEId   = UpdateCKEId(instr->v1.transLocMSToRmtMS.setRmtCKEId, loopGroupParam);
    uint16_t setRmtCKEMask = instr->v1.transLocMSToRmtMS.setRmtCKEMask;

    uint16_t clearType   = instr->v1.transLocMSToRmtMS.clearType;
    uint16_t lengthEn    = instr->v1.transLocMSToRmtMS.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transLocMSToRmtMS.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transLocMSToRmtMS.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transLocMSToRmtMS.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transLocMSToRmtMS.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    checker::DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, srcSlice));
    checker::DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(rmtMSId, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

    AddWrite(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    if (setRmtCKEMask != 0x0000) {
        AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, setRmtCKEId, setRmtCKEMask);

        // 将对应的bit位置一
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue));
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue | setRmtCKEMask));
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
               "RmtCKE[%u:%04x], Set CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, locMSId / 0x8000, locMSId % 0x8000, rmtMSId / 0x8000, rmtMSId % 0x8000,
               lengthXnId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransRmtMemToLocMSInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t locMSId     = UpdateMSId(instr->v1.transRmtMemToLocMS.locMSId, loopGroupParam);
    uint16_t rmtGSAId    = instr->v1.transRmtMemToLocMS.rmtGSAId;
    uint16_t rmtXnId     = instr->v1.transRmtMemToLocMS.rmtXnId;
    uint16_t lengthXnId  = instr->v1.transRmtMemToLocMS.lengthXnId;
    uint16_t channelId   = instr->v1.transRmtMemToLocMS.channelId;
    uint16_t clearType   = instr->v1.transRmtMemToLocMS.clearType;
    uint16_t lengthEn    = instr->v1.transRmtMemToLocMS.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transRmtMemToLocMS.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transRmtMemToLocMS.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transRmtMemToLocMS.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transRmtMemToLocMS.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t rmtAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtAddr);
    rmtAddr = UpdateGSAValue(rmtAddr, loopGroupParam);
    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    checker::DataSlice srcSlice;
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)rmtAddr, len, srcSlice));
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    // 获取远端rankId与dieId
    if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
               "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locMSId / 0x8000, locMSId % 0x8000, lengthXnId,
               channelId, setCKEId, setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransRmtMSToLocMemInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t locGSAId    = instr->v1.transRmtMSToLocMem.locGSAId;
    uint16_t locXnId     = instr->v1.transRmtMSToLocMem.locXnId;
    uint16_t rmtMSId     = UpdateMSId(instr->v1.transRmtMSToLocMem.rmtMSId, loopGroupParam);
    uint16_t lengthXnId  = instr->v1.transRmtMSToLocMem.lengthXnId;
    uint16_t channelId   = instr->v1.transRmtMSToLocMem.channelId;
    uint16_t clearType   = instr->v1.transRmtMSToLocMem.clearType;
    uint16_t lengthEn    = instr->v1.transRmtMSToLocMem.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transRmtMSToLocMem.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transRmtMSToLocMem.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transRmtMSToLocMem.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transRmtMSToLocMem.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    checker::DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));

    uint64_t localAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, localAddr);
    localAddr = UpdateGSAValue(localAddr, loopGroupParam);
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localAddr, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
               "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locGSAId, locXnId, lengthXnId,
               channelId, setCKEId, setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

HcclResult TransformTransRmtMSToLocMSInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t locMSId     = UpdateMSId(instr->v1.transRmtMSToLocMS.locMSId, loopGroupParam);
    uint16_t rmtMSId     = UpdateMSId(instr->v1.transRmtMSToLocMS.rmtMSId, loopGroupParam);
    uint16_t lengthXnId  = instr->v1.transRmtMSToLocMS.lengthXnId;
    uint16_t channelId   = instr->v1.transRmtMSToLocMS.channelId;
    uint16_t clearType   = instr->v1.transRmtMSToLocMS.clearType;
    uint16_t lengthEn    = instr->v1.transRmtMSToLocMS.lengthEn;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.transRmtMSToLocMS.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.transRmtMSToLocMS.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.transRmtMSToLocMS.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.transRmtMSToLocMS.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);

    checker::DataSlice srcSlice;
    CHK_RET(GenSliceFromMs(rmtMSId, len, srcSlice));
    checker::DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(locMSId, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use Channel[%u], "
               "Set CKE[%u:%04x], clearType[%u], lengthEn[%u]",
               waitCKEId, waitCKEMask, rmtMSId / 0x8000, rmtMSId % 0x8000, locMSId / 0x8000, locMSId % 0x8000,
               lengthXnId, channelId, setCKEId, setCKEMask, clearType, lengthEn);
    return HCCL_SUCCESS;
}

// TODO：这边待处理reduce信息
HcclResult TransformTransRmtMemToLocMemInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t locGSAId       = instr->v1.transRmtMemToLocMem.locGSAId;
    uint16_t locXnId        = instr->v1.transRmtMemToLocMem.locXnId;
    uint16_t rmtGSAId       = instr->v1.transRmtMemToLocMem.rmtGSAId;
    uint16_t rmtXnId        = instr->v1.transRmtMemToLocMem.rmtXnId;
    uint16_t lengthXnId     = instr->v1.transRmtMemToLocMem.lengthXnId;
    uint16_t channelId      = instr->v1.transRmtMemToLocMem.channelId;
    uint16_t reduceDataType = instr->v1.transRmtMemToLocMem.reduceDataType;
    uint16_t reduceOpCode   = instr->v1.transRmtMemToLocMem.reduceOpCode;
    uint16_t clearType      = instr->v1.transRmtMemToLocMem.clearType;
    uint16_t lengthEn       = instr->v1.transRmtMemToLocMem.lengthEn;
    uint16_t reduceEn       = instr->v1.transRmtMemToLocMem.reduceEn;
    uint16_t setCKEId       = instr->v1.transRmtMemToLocMem.setCKEId;
    uint16_t setCKEMask     = instr->v1.transRmtMemToLocMem.setCKEMask;
    uint16_t waitCKEId      = instr->v1.transRmtMemToLocMem.waitCKEId;
    uint16_t waitCKEMask    = instr->v1.transRmtMemToLocMem.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthXnId, len);
    CHK_PRT_RET(len == 0, HCCL_ERROR("The size of data transfer is 0."), HCCL_E_INTERNAL);
    CHK_PRT_RET(len > UB_MAX_SIZE,
        HCCL_ERROR("The size of data transfer is more than max transport size of UB."), HCCL_E_INTERNAL);

    uint64_t localAddr = 0x0;
    uint64_t rmtAddr = 0x0;
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, localAddr);
    AllRankParamRecorder::Global()->GetGSA(rankId, dieId, rmtGSAId, rmtAddr);

    checker::DataSlice srcSlice;
    RankId rId;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)rmtAddr, len, srcSlice, &rId));
    checker::DataSlice dstSlice;
    CHK_RET(MemLayout::Global()->GetSlice((checker::char_t*)localAddr, len, dstSlice));

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    // 说明transportId写错了
    if (rId != rmtRankId) {
        HCCL_ERROR("remoteRankId calculated by rmtAddr and channelId is not equal,"
            "remoteId is %d, but the addr is in rank %d", rmtRankId, rId);
        return HCCL_E_INTERNAL;
    }
    
    checker::CheckerReduceOp checkerReduceOp;
    checker::CheckerDataType checkerDataType;
    if (reduceEn) {
        checkerReduceOp = g_ReduceOp2CheckerReduceOp_ccu[reduceOpCode];
        DataType hcclDataType;
        CHK_RET(GetHcclDataTypeFromCCUDataType(reduceDataType, ccuReduceTypeMap[reduceOpCode], hcclDataType));
        checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
    }

    if (rmtRankId != rankId) {
        if (reduceEn) {
            AddReadReduce(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
        } else {
            AddRead(rankId, rmtRankId, queId, curCcuTask, srcSlice, dstSlice);
        }
    } else {
        if (reduceEn) {
            AddLocalReduce(rankId, queId, curCcuTask, srcSlice, dstSlice, checkerDataType, checkerReduceOp);
        } else {
            AddLocalCopy(rankId, queId, curCcuTask, srcSlice, dstSlice);
        }
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));
    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]",
        waitCKEId, waitCKEMask, rmtGSAId, rmtXnId, locGSAId, locXnId, lengthXnId, channelId, setCKEId, setCKEMask,
        clearType, lengthEn, reduceDataType, reduceOpCode, reduceEn);

    return HCCL_SUCCESS;
}

using TransformInstrFunc = HcclResult (*)(const CcuRep::CcuInstr*, TaskStubCcuGraph*, uint32_t, bool&, LoopGroupParam*);
extern std::unordered_map<uint16_t, TransformInstrFunc> g_transformInstrSqeMap;

// 处理Loop指令
HcclResult ProcessLoopIns(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask,
    uint32_t queId, bool& isContinue, LoopGroupParam* loopGroupParam, uint32_t loopGroupIdx)
{
    uint16_t startInstrId = instr->v1.loop.startInstrId;
    uint16_t endInstrId   = instr->v1.loop.endInstrId;
    uint16_t xnId         = instr->v1.loop.xnId;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    uint64_t value = 0;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, value);
    LoopXm xm{};
    xm.value = value;
    loopGroupParam->loopXms.push_back(xm);

    auto loopIdx = curCcuTask->loopIdx[loopGroupIdx]++;
    auto loopStart = AddLoopStartTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop前新增loopStart标记节点

    Hccl::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    for (u32 curLoopCnt = 0; curLoopCnt < xm.loopCnt; curLoopCnt++) {
        loopGroupParam->curLoopCnt = curLoopCnt;
        for (uint16_t insId = startInstrId; insId <= endInstrId; insId++) {
            // 获取当前要处理的指令
            const CcuRep::CcuInstr* instr = &microCodeQue.instrVec[insId - curCcuTask->startInstrIdInQue[queId]];
            if (g_transformInstrSqeMap.count(instr->header.header) == 0) {
                HCCL_ERROR("ins type not supported %hu", instr->header.header);
                return HCCL_E_INTERNAL;
            }
            CHK_RET(g_transformInstrSqeMap[instr->header.header](instr, curCcuTask, queId, isContinue, loopGroupParam));
        }
    }
    auto loopEnd = AddLoopEndTask(queId, loopIdx, loopGroupIdx, curCcuTask); // loop后新增loopEnd标记节点

    HCCL_INFO("Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]", startInstrId, endInstrId, xnId);
    return HCCL_SUCCESS;
}

HcclResult TransformLoopGroupInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopInfo)
{
    CHK_RET(IsLoopGroupParamNull(loopInfo));
    uint16_t startLoopInstrId = instr->v1.loopGroup.startLoopInstrId;
    uint16_t xnId             = instr->v1.loopGroup.xnId;
    uint16_t xmId             = instr->v1.loopGroup.xmId;
    uint16_t highPerfModeEn   = instr->v1.loopGroup.highPerfModeEn;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    LoopGroupParam loopGroupParam;
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, xnId, loopGroupParam.loopGroupXn.value);
    AllRankParamRecorder::Global()->GetXn(rankId, dieId, xmId, loopGroupParam.loopGroupXm.value);

    Hccl::CcuRep::CcuInstrInfo &microCodeQue = curCcuTask->instrInfo[queId];
    auto loopGroupIdx = curCcuTask->loopGroupIdx++;
    uint64_t loopCnt = loopGroupParam.loopGroupXn.loopInsCnt;
    for (u32 curLoopIdx = 0; curLoopIdx < loopCnt; curLoopIdx++) {
        loopGroupParam.curLoopIdx = curLoopIdx;
        // 计算当前Loop指令的位置
        u32 insPos = startLoopInstrId + curLoopIdx - curCcuTask->startInstrIdInQue[queId];
        if (curLoopIdx < loopGroupParam.loopGroupXn.expandOffset) {
            // 不用进行Loop展开
            CHK_RET(ProcessLoopIns(&microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, &loopGroupParam, loopGroupIdx));
        } else {
            // 需要进行loop展开
            for (u32 expandCnt = 0; expandCnt <= loopGroupParam.loopGroupXn.expandCnt; expandCnt++) {
                loopGroupParam.curExpandCnt = expandCnt;
                CHK_RET(ProcessLoopIns(&microCodeQue.instrVec[insPos], curCcuTask, queId, isContinue, &loopGroupParam, loopGroupIdx));
            }
        }
    }

    HCCL_INFO("LoopGroup From startLoopInstrId[%u] with loopGroupXn[%u], offsetXn[%u] and highPerfModeEn[%u]",
               startLoopInstrId, xnId, xmId, highPerfModeEn);
    return HCCL_SUCCESS;
}

HcclResult TransformLoopInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t startInstrId = instr->v1.loop.startInstrId;
    uint16_t endInstrId   = instr->v1.loop.endInstrId;
    uint16_t xnId         = instr->v1.loop.xnId;
    HCCL_ERROR("Loop From startInstrId[%u] to endInstrId[%u] with loopXn[%u]", startInstrId, endInstrId, xnId);
    // 当前Loop都需要通过LoopGoup来触发，暂不支持单独解析Loop命令
    return HCCL_E_INTERNAL;
}

HcclResult TransformClearCKEInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t clearType   = instr->v1.clearCKE.clearType;
    uint16_t clearCKEId  = UpdateCKEId(instr->v1.clearCKE.clearCKEId, loopGroupParam);
    uint16_t clearMask   = instr->v1.clearCKE.clearMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.clearCKE.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.clearCKE.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    if (clearMask != 0x0000) {
        HCCL_ERROR("do not support Clear CKE[%u:%04x]", clearCKEId, clearMask);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO("Wait CKE[%u:%04x], Clear CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, clearCKEId,
               clearMask, clearType);
    return HCCL_SUCCESS;
}

HcclResult TransformSyncGSAInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    CHK_RET(IsLoopGroupParamNull(loopGroupParam));
    uint16_t rmtGSAId      = instr->v1.syncGSA.rmtGSAId;
    uint16_t locGSAId      = instr->v1.syncGSA.locGSAId;
    uint16_t channelId     = instr->v1.syncGSA.channelId;
    uint16_t setRmtCKEId   = instr->v1.syncGSA.setRmtCKEId;
    uint16_t setRmtCKEMask = instr->v1.syncGSA.setRmtCKEMask;
    uint16_t clearType     = instr->v1.syncGSA.clearType;
    uint16_t setCKEId      = instr->v1.syncGSA.setCKEId;
    uint16_t setCKEMask    = instr->v1.syncGSA.setCKEMask;
    uint16_t waitCKEId     = instr->v1.syncGSA.waitCKEId;
    uint16_t waitCKEMask   = instr->v1.syncGSA.waitCKEMask;

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    // 获取远端rankId与dieId
    if (g_allRankChannelInfo[rankId][dieId].count(channelId) == 0) {
        HCCL_ERROR("channel id[%hu] in rank[%u] is not found", channelId, rankId);
        return HCCL_E_INTERNAL;
    }

    RankId rmtRankId = g_allRankChannelInfo[rankId][dieId][channelId].dstRank;
    uint32_t rmtDieId = g_allRankChannelInfo[rankId][dieId][channelId].remoteDieId;

    uint64_t localGSAValue = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetGSA(rankId, dieId, locGSAId, localGSAValue));
    CHK_RET(AllRankParamRecorder::Global()->SetGSA(rmtRankId, rmtDieId, rmtGSAId, localGSAValue));

    if (setRmtCKEMask != 0x0000) {
        AddPost(rankId, rmtRankId, queId, curCcuTask, rmtDieId, setRmtCKEId, setRmtCKEMask);

        // 将对应的bit位置一
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue));
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(rmtRankId, rmtDieId, setRmtCKEId, ckeValue | setRmtCKEMask));
    }

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask, true));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO(
        "Wait CKE[%u:%04x], Sync locGSAId[%u] To rmtGSAId[%u] Use Channel[%u], Set rmtCKE[%u:%04x], Set "
        "CKE[%u:%04x], clearType[%u]",
        waitCKEId, waitCKEMask, locGSAId, rmtGSAId, channelId, setRmtCKEId, setRmtCKEMask, setCKEId, setCKEMask,
        clearType);

    return HCCL_SUCCESS;
}

std::string ParseMSList(const CcuRep::CcuInstr *instr)
{
    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    uint16_t count = instr->v1.add.count;
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = instr->v1.add.msId[index];
    }

    std::string res = "MS[";
    for (uint16_t i = 0; i < count + 2; i++) { // 循环范围 0~count + 2
        if (i == count + 1) {
            res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + "]";
        } else {
            res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + ", ";
        }
    }
    return res;
}

HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType)
{
    static std::map<uint16_t, DataType> ccuSumDataTypeMap = {
        {0, DataType::FP32},    {1, DataType::FP16}, {2, DataType::BFP16}, {3, DataType::HIF8},  {4, DataType::FP8E4M3},
        {5, DataType::FP8E5M2}, {6, DataType::INT8}, {7, DataType::UINT8}, {8, DataType::INT16}, {9, DataType::INT32},
    };

    static std::map<uint16_t, DataType> ccuMaxMinDataTypeMap = {
        {0, DataType::FP32},  {1, DataType::FP16},  {2, DataType::BFP16}, {6, DataType::INT8},
        {7, DataType::UINT8}, {8, DataType::INT16}, {9, DataType::INT32},

    };

    if (ccuReduceType == CcuRep::CCU_REDUCE_SUM) {
        if (ccuSumDataTypeMap.find(ccuDataType) == ccuSumDataTypeMap.end()) {
            HCCL_ERROR("Unsupported CCUDataType[%hu] for Ccu SUM", ccuDataType);
            return HCCL_E_INTERNAL;
        }
        dataType = ccuSumDataTypeMap[ccuDataType];
    } else if (ccuReduceType == CcuRep::CCU_REDUCE_MAX || ccuReduceType == CcuRep::CCU_REDUCE_MIN) {
        if (ccuMaxMinDataTypeMap.find(ccuDataType) == ccuMaxMinDataTypeMap.end()) {
            HCCL_ERROR("Unsupported CCUDataType[%hu] for Ccu MAX/MIN", ccuDataType);
            return HCCL_E_INTERNAL;
        }
        dataType = ccuMaxMinDataTypeMap[dataType];
    } else {
        HCCL_ERROR("Unsupported ccuReduceType[%hu]", ccuReduceType);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult TransformAddInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t count       = instr->v1.add.count;
    uint16_t castEn      = instr->v1.add.castEn;
    uint16_t dataType    = instr->v1.add.dataType;
    uint16_t clearType   = instr->v1.add.clearType;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.add.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.add.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.add.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.add.waitCKEMask;
    uint16_t lengthId    = instr->v1.add.XnIdLength;

    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = UpdateMSId(instr->v1.add.msId[index], loopGroupParam);
    }
    // TODO: 对于不支持的数据类型需要进行报错处理

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthId, len));
    checker::DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(msId[0], len, dstSlice));

    std::vector<checker::DataSlice> srcSlices;
    for (uint16_t i = 1; i < count + 2; i++) {
        checker::DataSlice tmp;
        CHK_RET(GenSliceFromMs(msId[i], len, tmp));
        srcSlices.push_back(tmp);
    }

    DataType hcclDataType;
    CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, CcuRep::CCU_REDUCE_SUM, hcclDataType));
    AddLocalBatchReduce(rankId, queId, curCcuTask, srcSlices, dstSlice, hcclDataType);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO(
        "Wait CKE[%u:%04x], Add %s with Count[%u], DataType[%u] and CastEn[%u], Set CKE[%u:%04x], clearType[%u]",
        waitCKEId, waitCKEMask, ParseMSList(instr).c_str(), count, dataType, castEn, setCKEId, setCKEMask, clearType);

    return HCCL_SUCCESS;
}

HcclResult TransformMaxInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t count       = instr->v1.max.count;
    uint16_t dataType    = instr->v1.max.dataType;
    uint16_t clearType   = instr->v1.max.clearType;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.max.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.max.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.max.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.max.waitCKEMask;
    uint16_t lengthId    = instr->v1.max.XnIdLength;

    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = UpdateMSId(instr->v1.max.msId[index], loopGroupParam);
    }
    // TODO: 对于不支持的数据类型需要进行报错处理

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthId, len));
    checker::DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(msId[0], len, dstSlice));

    std::vector<checker::DataSlice> srcSlices;
    for (uint16_t i = 1; i < count + 2; i++) {
        checker::DataSlice tmp;
        CHK_RET(GenSliceFromMs(msId[i], len, tmp));
        srcSlices.push_back(tmp);
    }

    DataType hcclDataType;
    CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, CcuRep::CCU_REDUCE_MAX, hcclDataType));
    AddLocalBatchReduce(rankId, queId, curCcuTask, srcSlices, dstSlice, hcclDataType);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO(
        "Wait CKE[%u:%04x], Max %s with Count[%u], DataType[%u] and CastEn[%u], Set CKE[%u:%04x], clearType[%u]",
        waitCKEId, waitCKEMask, ParseMSList(instr).c_str(), count, dataType, setCKEId, setCKEMask, clearType);

    return HCCL_SUCCESS;
}

HcclResult TransformMinInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId,
    bool& isContinue, LoopGroupParam* loopGroupParam)
{
    uint16_t count       = instr->v1.min.count;
    uint16_t dataType    = instr->v1.min.dataType;
    uint16_t clearType   = instr->v1.min.clearType;
    uint16_t setCKEId    = UpdateCKEId(instr->v1.min.setCKEId, loopGroupParam);
    uint16_t setCKEMask  = instr->v1.min.setCKEMask;
    uint16_t waitCKEId   = UpdateCKEId(instr->v1.min.waitCKEId, loopGroupParam);
    uint16_t waitCKEMask = instr->v1.min.waitCKEMask;
    uint16_t lengthId    = instr->v1.min.XnIdLength;

    uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
    for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
        msId[index] = UpdateMSId(instr->v1.min.msId[index], loopGroupParam);
    }
    // TODO: 对于不支持的数据类型需要进行报错处理

    RankId rankId = curCcuTask->GetRankId();
    uint32_t dieId;
    curCcuTask->GetDieId(queId, dieId);

    CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, waitCKEId, waitCKEMask, isContinue));
    if (!isContinue) {
        return HCCL_SUCCESS;
    }

    uint64_t len = 0;
    CHK_RET(AllRankParamRecorder::Global()->GetXn(rankId, dieId, lengthId, len));
    checker::DataSlice dstSlice;
    CHK_RET(GenSliceFromMs(msId[0], len, dstSlice));

    std::vector<checker::DataSlice> srcSlices;
    for (uint16_t i = 1; i < count + 2; i++) {
        checker::DataSlice tmp;
        CHK_RET(GenSliceFromMs(msId[i], len, tmp));
        srcSlices.push_back(tmp);
    }

    DataType hcclDataType;
    CHK_RET(GetHcclDataTypeFromCCUDataType(dataType, CcuRep::CCU_REDUCE_MIN, hcclDataType));
    AddLocalBatchReduce(rankId, queId, curCcuTask, srcSlices, dstSlice, hcclDataType);

    CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, setCKEId, setCKEMask));

    CHK_RET(ClearWaitMask(rankId, dieId, waitCKEId, waitCKEMask));

    HCCL_INFO(
        "Wait CKE[%u:%04x], Min %s with Count[%u], DataType[%u] and CastEn[%u], Set CKE[%u:%04x], clearType[%u]",
        waitCKEId, waitCKEMask, ParseMSList(instr).c_str(), count, dataType, setCKEId, setCKEMask, clearType);
    return HCCL_SUCCESS;
}

std::unordered_map<uint16_t, TransformInstrFunc> g_transformInstrSqeMap = {
    {CcuRep::InstrHeader(LOAD_TYPE, LOADSQEARGSTOGSA_CODE).header, &TransformLoadSqeArgsToGSAInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, LOADSQEARGSTOXN_CODE).header, &TransformLoadSqeArgsToXnInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, LOADIMDTOGSA_CODE).header, &TransformLoadImdToGSAInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, LOADIMDTOXN_CODE).header, &TransformLoadImdToXnInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, LOADGSAXN_CODE).header, &TransformLoadGSAXnInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, LOADGSAGSA_CODE).header, &TransformLoadGSAGSAInstr},
    {CcuRep::InstrHeader(LOAD_TYPE, LOADXX_CODE).header, &TransformLoadXXInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, LOOPGROUP_CODE).header, &TransformLoopGroupInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, LOOP_CODE).header, &TransformLoopInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, SETCKE_CODE).header, &TransformSetCKEInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, CLEARCKE_CODE).header, &TransformClearCKEInstr},
    {CcuRep::InstrHeader(CTRL_TYPE, JMP_CODE).header, &TransformJumpInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMS_CODE).header, &TransformTransLocMemToLocMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMS_CODE).header, &TransformTransRmtMemToLocMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMEM_CODE).header, &TransformTransLocMSToLocMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMEM_CODE).header, &TransformTransLocMSToRmtMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMEM_CODE).header, &TransformTransRmtMSToLocMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMS_CODE).header, &TransformTransLocMSToLocMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMSTOLOCMS_CODE).header, &TransformTransRmtMSToLocMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMSTORMTMS_CODE).header, &TransformTransLocMSToRmtMSInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSRMTMEMTOLOCMEM_CODE).header, &TransformTransRmtMemToLocMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMEMTORMTMEM_CODE).header, &TransformTransLocMemToRmtMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE).header, &TransformTransLocMemToLocMemInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, SYNCCKE_CODE).header, &TransformSyncCKEInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, SYNCGSA_CODE).header, &TransformSyncGSAInstr},
    {CcuRep::InstrHeader(TRANS_TYPE, SYNCXN_CODE).header, &TransformSyncXnInstr},
    {CcuRep::InstrHeader(REDUCE_TYPE, ADD_CODE).header, &TransformAddInstr},
    {CcuRep::InstrHeader(REDUCE_TYPE, MAX_CODE).header, &TransformMaxInstr},
    {CcuRep::InstrHeader(REDUCE_TYPE, MIN_CODE).header, &TransformMinInstr},
};

HcclResult TransformInstr(const CcuRep::CcuInstr *instr, TaskStubCcuGraph *curCcuTask, uint32_t queId, bool& isContinue)
{
    if (g_transformInstrSqeMap.count(instr->header.header) == 0) {
        HCCL_ERROR("ins type not supported %hu", instr->header.header);
        return HCCL_E_INTERNAL;
    }

    return g_transformInstrSqeMap[instr->header.header](instr, curCcuTask, queId, isContinue, nullptr);
}

void addChildNode(std::deque<TaskNode*> &queue, std::set<TaskNode*> &isVisitedNode, TaskNode* curNode) {
    for (auto node = curNode->children.begin(); node != curNode->children.end(); node++) {
        TaskNode* child = *node;
        if (isVisitedNode.count(child) == 0) {
            isVisitedNode.insert(child);
            queue.push_back(child);
        }
    }
    return;
}

HcclResult TransformInstrQue(TaskNodePtr node, TaskStubCcuGraph *curCcuTask, uint32_t queId)
{
    // 表示当前微码序列已经完成成图
    if (curCcuTask->instrQueStatus[queId]) {
        return HCCL_SUCCESS;
    }

    bool isContinue = true;
    Hccl::CcuRep::CcuInstrInfo& microCodeQue = curCcuTask->instrInfo[queId];
    u32& pos = curCcuTask->microCodePosInQue[queId];
    while (pos < microCodeQue.instrVec.size()) {
        u32 prePos = pos;
        CHK_RET(TransformInstr(&microCodeQue.instrVec[pos], curCcuTask, queId, isContinue));
        if (!isContinue) {
            return HCCL_SUCCESS;
        }
        // 相等表示当前指令未刷新pos，则进行递增
        if (pos == prePos) {
            pos++;
        }
    }

    curCcuTask->instrQueStatus[queId] = true;
    curCcuTask->isGenGraphed = std::all_of(curCcuTask->instrQueStatus.begin(), curCcuTask->instrQueStatus.end(),
        [](bool b) { return b; });

    // 如果整图匹配完成，添加一个结束节点
    if (curCcuTask->isGenGraphed) {
        AddCcuSubGraphEnd(node, curCcuTask);
    }

    return HCCL_SUCCESS;
}

HcclResult ProcessCcuNode(TaskNodePtr node, TaskStubCcuGraph *curCcuTask)
{
    curCcuTask->queueNum_ = curCcuTask->instrInfo.size();
    uint32_t rankSize = RankInfoRecorder::Global()->GetRankSize();
    curCcuTask->bilateralPart1_.resize(curCcuTask->queueNum_);
    curCcuTask->bilateralPart2_.resize(curCcuTask->queueNum_);
    curCcuTask->bilateralNodes_.resize(curCcuTask->queueNum_);
    curCcuTask->waitInfoTmp_.resize(rankSize);
    curCcuTask->postInfoTmp_.resize(rankSize);
    for (uint32_t queId = 0; queId < curCcuTask->queueNum_; queId++) {
        CHK_RET(TransformInstrQue(node, curCcuTask, queId));
        CollectBilateralPostInfo(curCcuTask, queId, nullptr, true);
    }
    return HCCL_SUCCESS;
}

void PrintCcuSingleQue(TaskNodePtr head, u32 rankId, u32 queueIdx)
{
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;
    std::set<TaskNode*> alreadyPrintedNodes;

    // 打印首节点
    printf("ccu single queue head[%p]%s\n", head, head->task->Describe().c_str());
    alreadyPrintedNodes.insert(head);

    printf("children[");
    for (auto& child : head->children) {
        printf("%p,", child);
        if (isVisitedNode.count(child) == 0 && child->rankIdx == rankId && child->queIdx == queueIdx) {
            candNode.push_back(child);
            isVisitedNode.insert(child);
        }
    }
    printf("]\n");

    while(!candNode.empty()) {
        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        for (auto& child : curNode->children) {
            if (isVisitedNode.count(child) == 0 && child->rankIdx == rankId && child->queIdx == queueIdx) {
                candNode.push_back(child);
                isVisitedNode.insert(child);
            }
        }

        bool parentsAllPrint = true;
        for (auto& parent : curNode->parents) {
            if (parent->rankIdx == rankId && parent->queIdx == queueIdx) {
                if (alreadyPrintedNodes.count(parent) == 0) {
                    parentsAllPrint = false;
                    break;
                }
            }
        }

        if (parentsAllPrint) {
            printf("[%p]%s\n", curNode, curNode->task->Describe().c_str());
            alreadyPrintedNodes.insert(curNode);
        } else {
            candNode.push_back(curNode);
        }
    }
    return;
}

void PrintCcuGraph(TaskNodePtr dummyStart)
{
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;

    for (auto node = dummyStart->children.begin(); node != dummyStart->children.end(); node++) {
        TaskNode* child = *node;
        isVisitedNode.insert(child);
        candNode.push_back(child);
    }

    while(!candNode.empty()) {
        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        addChildNode(candNode, isVisitedNode, curNode);

        if (curNode->task->GetType() != TaskTypeStub::CCU_GRAPH) {
            continue;
        }

        printf("=======================================================\n");
        printf("rank id is %d\n", curNode->rankIdx);

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(curNode->task);
        for (auto& child : curCcuTask->ccuHeadTaskNode->children) {
            u32 queueIdx = child->queIdx;
            printf("-------------------------------------------------------\n");
            printf("stream/queue id is %d\n", queueIdx);
            PrintCcuSingleQue(child, curNode->rankIdx, queueIdx);
        }
    }
    printf("-------------------------------------------------------\n");
    return;
}

HcclResult GenCcuGraph(TaskNode* dummyStart) {
    std::deque<TaskNode*> candNode;
    std::set<TaskNode*> isVisitedNode;
    HCCL_INFO("[GenCcuGraph] dummyStart children size: %u", dummyStart->children.size());

    bool isPrintCcuGraph = std::getenv("CCU_TASK_PRINT");
    for (auto node = dummyStart->children.begin(); node != dummyStart->children.end(); node++) {
        TaskNode* child = *node;
        isVisitedNode.insert(child);
        candNode.push_back(child);
    }

    u32 unmatchedCnt = 0;
    while(!candNode.empty()) {
        // 先判断是否存在死锁的情况
        if (unmatchedCnt >= candNode.size()) {
            for (auto &node : candNode) {
                node->unmatch = true;
            }
            HCCL_ERROR("deadLocking occurs due to mismatch.");
            return HcclResult::HCCL_E_INTERNAL;
        }

        TaskNodePtr curNode = candNode.front();
        candNode.pop_front();
        if (curNode->task->GetType() != TaskTypeStub::CCU_GRAPH) {
            addChildNode(candNode, isVisitedNode, curNode);
            unmatchedCnt = 0;
            continue;
        }

        TaskStubCcuGraph *curCcuTask = dynamic_cast<TaskStubCcuGraph *>(curNode->task);
        std::vector<u32> microCodePosInQuePre = curCcuTask->microCodePosInQue;
        HcclResult ret = ProcessCcuNode(curNode, curCcuTask);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("ProcessCcuNode failed");
            if (isPrintCcuGraph) {
                PrintCcuGraph(dummyStart);
            }
            return ret;
        }

        if (microCodePosInQuePre != curCcuTask->microCodePosInQue) {
            // 表示当前子图有匹配到部分微码
            unmatchedCnt = 0;
        } else {
            unmatchedCnt++;
        }

        // 如果整个CCU子图已经匹配完成
        if (curCcuTask->isGenGraphed) {
            addChildNode(candNode, isVisitedNode, curNode);
            continue;
        }

        // CCU子图未完成匹配，需要放回到队列中，等待下次处理
        candNode.push_back(curNode);
    }

    // 检查是否有未匹配的post节点
    CHK_RET(AllRankParamRecorder::Global()->CheckAllPostMatch());

    if (isPrintCcuGraph) {
        PrintCcuGraph(dummyStart);
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
