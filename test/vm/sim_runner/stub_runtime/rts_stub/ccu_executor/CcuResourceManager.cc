/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor resource manager
 * Author: z00445483
 */

#include <cstring>
#include <iostream>
#include <queue>
#include "log.h"
#include "sal.h"
#include "CcuResourceManager.h"
#include "CcuExecutorManager.h"
#include "ccu_microcode.h"
#include "SimRunnerMgr.h"

using namespace std;
using namespace Hccl::CcuRep;

void CcuResouceManager::Init(int rankId, int rankSize, CcuVersion version)
{
    ccuResData_.version = version;
    rankId_ = rankId;
    if (ccuResData_.version == CCU_V1) {
        if (ccuResData_.v1Res == nullptr) {
            ccuResData_.v1Res = std::make_unique<CcuResouceV1>(rankId, rankSize);
        }
    } else {
        std::cout<<"[ERROR][CcuResouceManager][Init] ccu version "<<static_cast<int>(ccuResData_.version)<<" not supported"<<std::endl;
    }
}

const static char *tomlFileFormatStr  = "./%s/ccu_%u_%u_%u_sq.toml";
const static char *instrFileFormatStr = "./%s/instruction_0_%u_%u.bin";
const static char *chanelFileFormatStr = "./%s/channel_0_%u_%u.txt";
HcclResult Tomlfile(const tagRtCcuTaskInfo &taskInfo, int  devId, int index)
{
    char instrFilename[256];
    sprintf(instrFilename, tomlFileFormatStr ,CcuResouceManager::GetInstance().GetCcuInfoFilePath().c_str(), devId, taskInfo.dieId + 2, index);
        FILE *fp = fopen(instrFilename, "w");
    if (fp == nullptr) {
        HCCL_WARNING("Open File Failed!");
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;
    }
    fprintf(fp, "name = \" ccu \"\r\n");
    fprintf(fp, "[[tasks]]\r\n");
    fprintf(fp, "sqe_type = \"CCU\"\r\n");
    fprintf(fp, "mission_id = %u\r\n", taskInfo.missionId);
    fprintf(fp, "iodie_id = %u\r\n", taskInfo.dieId + 2);   
    fprintf(fp, "inst_start_id = %u\r\n", taskInfo.instStartId);
    fprintf(fp, "inst_cnt = %u\r\n", taskInfo.instCnt);
    fprintf(fp, "inst_addr_key_value = %u\r\n", taskInfo.key);
    fprintf(fp, "args = [");
    fprintf(fp, "%lu, ", 0x90D2000000 + devId * 0x10000000000); //打桩适配ca model的sendBuffer,只适用于1D的
    fprintf(fp, "%lu, ", 0x90E2000000 + devId * 0x10000000000); //打桩适配ca model的recvBuffer,只适用于1D的
    for (int i = 2; i < taskInfo.argSize - 1; i++) {
        fprintf(fp, "%lu, ", taskInfo.args[i]);
    }
    fprintf(fp, "%lu]", taskInfo.args[taskInfo.argSize - 1]);
    fclose(fp);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GeneCCUToml(std::map<int, std::vector<FakeSqe>> sqeQueues)
{
    for (auto &stream : sqeQueues) {
        int streamIndex = 0;
        for (auto &sqe : stream.second) {
            auto taskInfo = sqe.ccuTaskInfo;
            for (u32 i = 0; i < taskInfo.argSize; i++) {
                HCCL_DEBUG("task arg index=[%u], value=[%lu].", i, taskInfo.args[i]);
            }
            HcclResult ret = Tomlfile(taskInfo, static_cast<int>(sqe.devId), streamIndex++);
            if  (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_WARNING("GeneCCUToml failed");
                return ret;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GeneCCUInstr(const CcuInstrData &cccuInstrInfo, int rankId, int dieId)
{
    char tomlFilename[256];
    static int cnt = 0;
    sprintf(tomlFilename, instrFileFormatStr, CcuResouceManager::GetInstance().GetCcuInfoFilePath().c_str(), rankId, dieId + 2);
    if (cccuInstrInfo.instrCnt == 0) {
        HCCL_WARNING("rankId[%d] dieId[%d] instr count is zero!", rankId, dieId);
        return HcclResult::HCCL_E_PARA;
    }
    FILE *fp = fopen(tomlFilename, "w");
    if (fp == nullptr) {
        HCCL_WARNING("Open instr File Failed!");
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;
    }
    uint64_t *ptr = (uint64_t *)(cccuInstrInfo.instrData.data());
    for (uint32_t i = 0; i < cccuInstrInfo.instrCnt; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            if (j == 3) {
                fprintf(fp, "%lu\r\n", ptr[i * 4 + j]);
            } else {
                fprintf(fp, "%lu\t", ptr[i * 4 + j]);
            }
        }
    }
    fclose(fp);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult GeneChannelMap(array<CcuInfo, SimCcuV1::MAX_CCU_CHANNEL_NUM> oneDiechannelMap, int rankId, int dieId)
{
    if (oneDiechannelMap[0].rankId == 0x7FFFFFFF) {
        HCCL_DEBUG("GeneChannelMap::die[%d] channel not config!", dieId);
        return HcclResult::HCCL_SUCCESS;
    }
    uint32_t channelId = 0;
    uint32_t rmtRankId = 0;
    uint32_t rmtDieId = 0;
    uint32_t locaRankId = rankId;
    uint32_t locaDieid = dieId;
    char chanelFilename[256];
    sprintf(chanelFilename, chanelFileFormatStr, CcuResouceManager::GetInstance().GetCcuInfoFilePath().c_str(), locaRankId, locaDieid + 2);
    printf("GeneChannelMap::rankId[%d], dieId[%d]\n", locaRankId, locaDieid);
    FILE *fp = fopen(chanelFilename, "w");
    if (fp == nullptr) {
        HCCL_WARNING("Open channel File Failed!");
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;;
    }
    for (int i = 0; i < oneDiechannelMap.size(); i++) {
        channelId = i;
        rmtRankId = oneDiechannelMap[i].rankId;
        rmtDieId = oneDiechannelMap[i].dieId;
        if (rmtRankId == 0x7FFFFFFF && rmtDieId == 0x7FFFFFFF) {
            HCCL_DEBUG("GeneChannelMap::channelId[%d] not config", channelId);
            break;
        }
        fprintf(fp, "%u %u %u\r\n", channelId, rmtRankId, rmtDieId + 2);
        printf("GeneChannelMap channelId[%u], {%u:%u -> %u:%u}\n", channelId, locaRankId, locaDieid, rmtRankId, rmtDieId);
    }
    fclose(fp);
    return HcclResult::HCCL_SUCCESS;
}

void CcuResouceManager::GeneCaModelFile(const array<CcuInstrData, DIE_NUM> &ccuInstrInfo, std::map<int, std::vector<FakeSqe>> sqeQueues) const
{
    HcclResult res = HcclResult::HCCL_E_RESERVED;
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        res = GeneCCUInstr(ccuInstrInfo[dieId], rankId_, dieId);
        if (res == HcclResult::HCCL_E_OPEN_FILE_FAILURE) {
            HCCL_ERROR("GeneCCUInstr failed");
        }
        res = GeneChannelMap(ccuResData_.v1Res->channelId2RmtRankMap_[dieId], rankId_, dieId);
        if (res != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("GeneChannelMap failed");
        }
    }
    res = GeneCCUToml(sqeQueues);
    if (res != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("GeneCcuTaskInfo failed");
    }
}

void CcuResouceManager::InitInstrInfo(const array<CcuInstrData, DIE_NUM> &ccuInstrInfo, std::map<int, std::vector<FakeSqe>> sqeQueues)
{
    if (ccuResData_.version == CCU_V1) {
        ccuResData_.v1Res->InitInstrInfo(ccuInstrInfo);
    } else {
        std::cout<<"[ERROR][CcuResouceManager][InitInstrInfo] ccu version "<<static_cast<int>(ccuResData_.version)<<" not supported"<<std::endl;
    }
    GeneCaModelFile(ccuInstrInfo, sqeQueues);
    for (int i = 0; i < DIE_NUM; i++) {
        DumpChannelId2RmtRank(i);
    }
    DumpCcuInstructions();
}

void CcuResouceManager::InitChannelId2RmtRankMap(int rankId, int dieId, uint16_t channelId, int rmtRank, uint16_t rmtDieId)
{
    if (dieId >= DIE_NUM || rmtDieId >= DIE_NUM || channelId >= SimCcuV1::MAX_CCU_CHANNEL_NUM) {
        return;
    }
    CcuInfo ccuInfo;
    ccuInfo.rankId = rmtRank;
    ccuInfo.dieId  = rmtDieId;
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        ccuResData_.v1Res->channelId2RmtRankMap_[dieId][channelId] = ccuInfo;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][InitChannelId2RmtRankMap] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
    }
}

void CcuResouceManager::AddTaskInfo(int dieId, const rtCcuTaskInfo_t &ccuTaskInfo)
{
    if (dieId >= DIE_NUM) {
        HCCL_ERROR("[CcuResouceManager][AddTaskInfo] invalid dieId[%d].", dieId);
        return;
    }
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        ccuResData_.v1Res->ccuTaskInfos_[dieId] = ccuTaskInfo;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][AddTaskInfo] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
    }
}

uint64_t CcuResouceManager::GetSqeArgValue(int rankId, int dieId, uint16_t argId) const
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return ccuResData_.v1Res->ccuTaskInfos_[dieId].args[argId];
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetSqeArgValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
        return U64_INVALID;
    }
}

uint64_t CcuResouceManager::GetXnValue(int rankId, int dieId, uint16_t xnId) const
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.xn[dieId][xnId];
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetXnValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
         return U64_INVALID;
    }
}

uint64_t CcuResouceManager::GetGsaValue(int rankId, int dieId, uint16_t gsaId) const
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.gsa[dieId][gsaId];
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetGsaValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
         return U64_INVALID;
    }
}

uint16_t CcuResouceManager::GetCkeValue(int rankId, int dieId, uint16_t ckeId) const
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.cke[dieId][ckeId];
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetCkeValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
         return U16_INVALID;
    }
}

std::pair<int, int> CcuResouceManager::GetRmtCcu(int dieId, uint16_t channelId) const
{
    int rmtRank = S32_INVALID;
    int rmtDie  = S32_INVALID;

    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        rmtRank = ccuResData_.v1Res->channelId2RmtRankMap_[dieId][channelId].rankId;
        rmtDie  = ccuResData_.v1Res->channelId2RmtRankMap_[dieId][channelId].dieId;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetRmtCcu] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    if (dieId >= DIE_NUM || rmtRank == S32_INVALID || rmtDie == S32_INVALID) {
        return std::make_pair(S32_INVALID, S32_INVALID);
    }

    return std::make_pair(rmtRank, rmtDie);
}

void CcuResouceManager::UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value)
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.xn[dieId][xnId] = value;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][UpdateXnValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
    }
}

void CcuResouceManager::UpdateGsaValue(int rankId, int dieId, uint16_t gsaId, uint64_t value)
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.gsa[dieId][gsaId] = value;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][UpdateGsaValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
    }
}

void CcuResouceManager::UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value)
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.cke[dieId][ckeId] = value;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][UpdateCkeValue] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
    }
}

void CcuResouceManager::TransMemToMem(void *srcBuf, void *dstBuf, uint64_t length)
{
    if (srcBuf == nullptr) {
        HCCL_ERROR("[CcuResouceManager][TransMemToMem] srcBuf invalid...");
        return;
    }

    if (dstBuf == nullptr) {
        HCCL_ERROR("[CcuResouceManager][TransMemToMem] dstBuf invalid...");
        return;
    }

    memcpy(dstBuf, srcBuf, length);
}

void CcuResouceManager::TransMSToMS(int srcRank, int srcDie, int dstRank, int dstDie, uint16_t srcMsId, uint16_t dstMsId, uint16_t length)
{
    memcpy(GetMsAddr(dstRank, dstDie, dstMsId), GetMsAddr(srcRank, srcDie, srcMsId), length);
}

void CcuResouceManager::TransMSToMem(int rankId, int dieId, uint16_t msId, void *buf, uint16_t length)
{
    if (buf == nullptr) {
        HCCL_ERROR("[CcuResouceManager][TransMSToMem] param invalid...");
        return;
    }

    memcpy(buf, GetMsAddr(rankId, dieId, msId), length);
}

void CcuResouceManager::TransMemToMS(int rankId, int dieId, uint16_t msId, void *buf, uint16_t length)
{
    if (buf == nullptr) {
        HCCL_ERROR("[CcuResouceManager][TransMemToMS] param invalid...");
        return;
    }

    memcpy(GetMsAddr(rankId, dieId, msId), buf, length);

    auto msValuePtr0 = reinterpret_cast<int32_t*>(GetMsAddr(rankId, dieId, msId));
}

char *CcuResouceManager::GetMsAddr(int rankId, int dieId, uint16_t msId) const
{
    auto version = ccuResData_.version;
    uint32_t offset = static_cast<uint32_t>(msId) * BYTE_NUM_4K;  // 一个MS容量为4KB
    if (version == CCU_V1) {
        return (ccuResData_.v1Res->allShmBase_[rankId]->head.ccu.ms[dieId].data() + offset);
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetMsAddr] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
        return nullptr;
    }
}

uint16_t CcuResouceManager::GetInstrCnt(int dieId) const
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return ccuResData_.v1Res->instrSpace_[dieId].instrCnt;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetInstrCnt] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
        return U16_INVALID;
    }
}

std::vector<Hccl::CcuRep::CcuInstr> CcuResouceManager::GetInstrData(int dieId) const
{
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return ccuResData_.v1Res->instrSpace_[dieId].instrData;
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetInstrCnt] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
        return {};
    }
}

std::string CcuResouceManager::GetInstrDescribe(int dieId, int instrId) const
{
#ifndef DEVICE_STUB
    auto version = ccuResData_.version;
    if (version == CCU_V1) {
        return Hccl::CcuRep::ParseInstr(&(ccuResData_.v1Res->instrSpace_[dieId].instrData[instrId]));
    } else {
        std::cout<<"[ERROR][CcuResouceManager][GetInstrDescribe] ccu version "<<static_cast<int>(version)<<" not supported"<<std::endl;
        return "";
    }
#endif
    return "";
}

void CcuResouceManager::DumpCcuInstructions() const
{
    if (!enableDump_) {
        return;
    }
#ifndef DEVICE_STUB
    // 打印instr信息
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        std::cout<<"==========================print ccu instructions start: rank["<<rankId_<<"], dieId["<<dieId<<"]=========================="<<std::endl;
        auto instrInfoCnt = GetInstrCnt(dieId);
        std::cout<<"*******************rankId=["<<rankId_<<"], dieId=["<<dieId<<"], instrCnt=["<<instrInfoCnt<<"]"<<std::endl;

        for (uint32_t i = 0; i < instrInfoCnt; i++) {
            std::cout<<"ccu instruction info: "<<i<<": "<<GetInstrDescribe(dieId, i)<<std::endl;
        }
        std::cout<<"==========================print ccu instructions end: rank["<<rankId_<<"], dieId["<<dieId<<"]=========================="<<std::endl;
    }
#endif
}

void CcuResouceManager::DumpCcuXnResouceInfo() const
{
    for (u32 dieId = 0; dieId < DIE_NUM; dieId++) {
        std::cout<<"==========================All CCU XN Resouce Info Start: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
        std::cout<<"-------------------------DieId["<<dieId<<"] XN Resouce Info Start-------------------------" << std::endl;
        for (u32 xnId = 0; xnId < SimCcuV1::CCU_RESOURCE_XN_MAX; xnId++) {
            auto xnValue = GetXnValue(rankId_, dieId, xnId);
            if (xnValue != 0) {
                std::cout<<"[XN: Id = " << xnId <<", Value = " <<std::hex<<xnValue<<"]" << std::endl;
            }
        }
        std::cout<<"-------------------------DieId["<<dieId<<"] XN Resouce Info End-------------------------" << std::endl;
        std::cout<<"==========================All CCU XN Resouce Info End: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
    }
}

void CcuResouceManager::DumpCcuGsaResouceInfo() const
{
    for (u32 dieId = 0; dieId < DIE_NUM; dieId++) {
        std::cout<<"==========================All CCU GSA Resouce Info Start: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
        std::cout<<"-------------------------DieId["<<dieId<<"] GSA Resouce Info Start-------------------------" << std::endl;
        for (u32 gsaId = 0; gsaId < SimCcuV1::CCU_RESOURCE_GSA_MAX; gsaId++) {
            auto gsaValue = GetGsaValue(rankId_, dieId, gsaId);
            if (gsaValue != 0) {
                std::cout<<"[GSA: Id = "<<gsaId<<", Value = " <<std::hex<<gsaValue<<"]" << std::endl;
            }
        }
        std::cout<<"-------------------------DieId["<<dieId<<"] GSA Resouce Info End-------------------------" << std::endl;
        std::cout<<"==========================All CCU GSA Resouce Info End: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
    }
}

void CcuResouceManager::DumpCcuCkeResouceInfo() const
{
    for (u32 dieId = 0; dieId < DIE_NUM; dieId++) {
        std::cout<<"==========================All CCU CKE Resouce Info Start: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
        std::cout<<"-------------------------DieId["<<dieId<<"] CKE Resouce Info Start-------------------------" << std::endl;
        for (u32 ckeId = 0; ckeId < SimCcuV1::CCU_RESOURCE_MS_NUM; ckeId++) {
            auto ckeValue = GetGsaValue(rankId_, dieId, ckeId);
            if (ckeValue != 0) {
                std::cout<<"[CKE: Id = "<<ckeId<<", Value = " <<std::hex<<ckeValue<<"]" << std::endl;
            }
        }
        std::cout<<"-------------------------DieId["<<dieId<<"] CKE Resouce Info End-------------------------" << std::endl;
        std::cout<<"==========================All CCU CKE Resouce Info End: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
    }
}

void CcuResouceManager::DumpCcuChannelResouceInfo() const
{
    for (int dieId = 0; dieId < DIE_NUM; dieId++) {
        std::cout<<"==========================All CCU CHANNEL Resouce Info Start: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
        std::cout<<"-------------------------DieId["<<dieId<<"] CHANNEL Resouce Info Start-------------------------" << std::endl;
        for (u32 chId = 0; chId < SimCcuV1::MAX_CCU_CHANNEL_NUM; chId++) {
            auto rmtCcu = GetRmtCcu(dieId, chId);
            if (rmtCcu.first != S32_INVALID && rmtCcu.second != S32_INVALID) {
                std::cout<<"[CHANNEL: Id = "<<chId<<", Local["<<rankId_<<":"<<dieId<<"] -> Remote["<<rmtCcu.first<<":"<<rmtCcu.second<<"]]" << std::endl;
            }
        }
        std::cout<<"-------------------------DieId["<<dieId<<"] CHANNEL Resouce Info End-------------------------" << std::endl;
        std::cout<<"==========================All CCU CHANNEL Resouce Info End: rank["<<rankId_<<"], dieId["<<dieId<<"]===========================" << std::endl;
    }
}

void CcuResouceManager::DumpChannelId2RmtRank(int dieId) const
{
    if (!enableDump_) {
        return;
    }

    auto version = ccuResData_.version;
    for (u32 i = 0; i < SimCcuV1::MAX_CCU_CHANNEL_NUM; i++) {
        auto rmtCcu = GetRmtCcu(dieId, i);
        if (rmtCcu.first == INT32_MAX || rmtCcu.second == INT32_MAX) {
            continue;
        }
        HCCL_INFO("[CcuResouceManager][DumpChannelId2RmtRank] channelId[%d], ccu[%d:%d --> %d:%d]", i, rankId_, dieId, rmtCcu.first, rmtCcu.second);
    }
}

void CcuResouceManager::DumpCcuAllResouceInfo() const
{
    DumpCcuXnResouceInfo();
    DumpCcuGsaResouceInfo();
    DumpCcuCkeResouceInfo();
    DumpCcuChannelResouceInfo();
}

string& CcuResouceManager::GetCcuInfoFilePath()
{
    return GeneCcuInfoFilePath;
}